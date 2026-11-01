"""E8 click effect (research/47): on the synthetic development set holdout4 a simulated reviewer clicks once per review
window on the true trace where the red part starts; SAM 3 film2 re-tracks from that point exactly as the review page
does (research/45: 480-column window opened 40 columns before the first disputed column, track to the window end + 128,
at least one 256-column SAM window, click snapped to the darkest row within +-5); every re-track is accepted, and the
re-tracked path is scored against the ground truth on the window's disputed columns.

usage: python -I scripts/exp_click.py run       # once (start marker): oracle clicks, then clicks with +-3 px noise
       python -I scripts/exp_click.py score     # tables and event-level bootstrap from the saved rows
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")                     # the model is loaded offline only
import hashlib                                                   # noqa: E402
import json                                                      # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402
from pathlib import Path                                         # noqa: E402

import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ml_consensus as mc                                        # noqa: E402

OUT = ROOT / "out" / "ml" / "click"
DOC47 = ROOT.parent / "47-click-effect.md"                       # internal notes, not published; hashed only if present
SAM3_CKPT = ROOT / "out" / "ml" / "sam3_film2_modal" / "ckpt_sam3_film2_best.pt"
SAM3_FROZEN = ROOT / "out" / "ml" / "sam3_film2" / "frozen_method.json"
SPLIT = "holdout4"
VIEW, LEAD, MARGIN, SNAP, GAP, MIN_SPAN, WARN = 480, 40, 128, 5, 30, 15, 0.25   # research/45 review page
NOISE_PX, SEED = 3.0, 0
NONE, AUTO, DISPUTED, HUMAN = 0, 1, 2, 3
VARIANTS = ("oracle", "noisy")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(e[::2].tolist(), e[1::2].tolist()))


def spans(dis):
    """the review page's spans: disputed runs merged across gaps of <= 30 columns, kept if >= 15 columns"""
    out, cur = [], None
    for a, b in runs(dis):
        if cur and a - cur[1] <= GAP:
            cur[1] = b
        else:
            if cur:
                out.append(cur)
            cur = [a, b]
    if cur:
        out.append(cur)
    return [(a, b) for a, b in out if b - a >= MIN_SPAN]


def snap(img, x, y):
    """review_server.retrack: the darkest row within +-5 of the click, averaged over columns x-1..x+1"""
    H, W = img.shape
    yc = int(round(y))
    lo, hi = max(0, yc - SNAP), min(H, yc + SNAP + 1)
    col = img[lo:hi, max(0, x - 1):min(W, x + 2)].astype(float).mean(1)
    return float(lo + int(np.argmin(col)))


def image(it):
    return cv2.imread(str(mc.SETS[SPLIT][0] / it["record"] / "film" / f"heli_g{it['gain']:g}.png"), 0)


def simulate(m3, model, dev, fz, it, rule, variant, rng):
    """every review window of every line of one image -> rows (columns are image columns)"""
    img = image(it)
    (H, W), S = img.shape, it["S"]
    x0 = (W - it["P1"].shape[1]) // 2                             # load_pairs slices [x0, W - x0)
    x1 = W - x0
    acc = mc.accept(it, rule["tau_S"], rule["min_run_cols"])
    rows = []
    for k in range(it["n"]):
        ok, pen, P1, P2 = it["ok"][k], it["pen"][k], it["P1"][k], it["P2"][k]
        with np.errstate(invalid="ignore"):                       # a reviewer can only click on a trace on the image
            visible = ok & (pen >= 0) & (pen <= H - 1)            # (research/47 §5.1: clipped high-gain traces)
        stat = np.where(acc[k], AUTO, np.where(ok, DISPUTED, NONE)).astype(np.int8)
        cur = P1.copy()
        for a, b in spans(ok & ~acc[k]):
            pos = a
            while pos < b:
                idx = np.flatnonzero((stat[pos:b] == DISPUTED) & visible[pos:b])
                left = (stat[pos:b] == DISPUTED) & ok[pos:b]
                if not len(idx):                                  # only off-image disputed columns are left
                    if left.any():
                        rows.append({"record": it["record"], "group": it["group"], "gain": it["gain"], "line": k,
                                     "quake": k == it["q"], "c": int(x0 + pos), "w1": int(x0 + b), "n": int(left.sum()),
                                     "unreached": True, "reason": "off_image"})
                    break
                c = pos + int(idx[0])
                skipped = int(((stat[pos:c] == DISPUTED) & ok[pos:c]).sum())
                if skipped:                                       # off-image disputed columns before the click
                    rows.append({"record": it["record"], "group": it["group"], "gain": it["gain"], "line": k,
                                 "quake": k == it["q"], "c": int(x0 + pos), "w1": int(x0 + c), "n": skipped,
                                 "unreached": True, "reason": "off_image"})
                w1 = min(c - LEAD + VIEW, len(stat))
                xc, x_end = x0 + c, min(x1, max(x0 + w1 + MARGIN, x0 + c + m3.S.WIN_W))
                score_cols = np.flatnonzero((stat[c:w1] == DISPUTED) & ok[c:w1]) + c
                row = {"record": it["record"], "group": it["group"], "gain": it["gain"], "line": k, "quake": k == it["q"],
                       "c": int(xc), "w1": int(x0 + w1), "n": int(len(score_cols))}
                if x_end - xc < m3.S.WIN_W:                       # the page refuses: too close to the line end
                    rows.append(row | {"unreached": True, "reason": "line_end"})
                    pos = w1
                    continue
                y_click = float(pen[c]) + (rng.uniform(-NOISE_PX, NOISE_PX) if variant == "noisy" else 0.0)
                y_click = float(np.clip(y_click, 0, H - 1))       # a click is always on the image
                ys = snap(img, xc, y_click)
                t = time.time()
                with m3.amp_ctx(dev, bool(fz["amp_bf16"])):
                    probb, _ = m3.S.segment_image(model, img, np.array([ys]), xc, x_end, tops=np.array([0]), dev=dev)
                    P = m3.S.decode_paths(probb, np.array([ys]), xc, x_end, float(fz["lam_final"]), dev, tops=np.array([0]))
                new = P[0, xc:x_end]                              # sliced columns c .. c + len(new)
                j = np.arange(c, c + len(new))
                held = np.isin(stat[j], (AUTO, HUMAN)) & np.isfinite(new) & np.isfinite(cur[j])
                max_dev = float(np.max(np.abs(new[held] - cur[j][held])) / S) if held.any() else None
                e_new = np.abs(new[score_cols - c] - pen[score_cols]) / S
                e_old = np.abs(P1[score_cols] - pen[score_cols]) / S
                e_ra2 = np.abs(P2[score_cols] - pen[score_cols]) / S
                with np.errstate(invalid="ignore"):
                    row.update({
                        "unreached": False, "y_true": float(pen[c]), "y_click": y_click, "y_snapped": ys, "x_end": int(x_end),
                        "warn": max_dev is not None and max_dev > WARN, "max_dev_S": max_dev, "secs": round(time.time() - t, 3),
                        "new4": int((e_new <= 0.25).sum()), "new2": int((e_new <= 0.5).sum()),
                        "old4": int((e_old <= 0.25).sum()), "old2": int((e_old <= 0.5).sum()),
                        "ra4": int((e_ra2 <= 0.25).sum()), "ra2": int((e_ra2 <= 0.5).sum()),
                        "fixed": int(((e_old > 0.25) & (e_new <= 0.25)).sum()), "broken": int(((e_old <= 0.25) & (e_new > 0.25)).sum())})
                m = stat[c:w1] == DISPUTED                        # accept: the window's disputed columns take the re-track
                seg = cur[c:w1]
                seg[m] = new[:w1 - c][m]
                stat[c:w1][m] = HUMAN
                rows.append(row)
                pos = w1
    return rows


def cmd_run():
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / "run_started.json"
    if marker.exists():
        raise SystemExit(f"E8 already started at {json.loads(marker.read_text())['started']}: one run only (research/47 §5)")
    fz, rule_file = json.loads(SAM3_FROZEN.read_text()), mc.OUT / "frozen_rule.json"
    marker.write_text(json.dumps({"started": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "split": SPLIT,
                                  "code_sha256": sha_file(__file__), "doc_sha256": sha_file(DOC47) if DOC47.exists() else None,
                                  "rule_sha256": sha_file(rule_file), "ckpt_sha256": fz["ckpt_sha256"]}, indent=1))
    import ml_sam3 as m3
    assert m3.sha256(SAM3_CKPT) == fz["ckpt_sha256"], "SAM 3 checkpoint is not the frozen file"
    dev = m3.get_device("auto")
    model, _ = m3.load_finetuned(SAM3_CKPT, dev)
    rule = json.loads(rule_file.read_text())
    items = mc.load_pairs(SPLIT)
    for variant in VARIANTS:
        rng = np.random.default_rng(SEED)
        t0 = time.time()
        with (OUT / f"rows_{variant}.jsonl").open("w") as f:
            for i, it in enumerate(items):
                for row in simulate(m3, model, dev, fz, it, rule, variant, rng):
                    f.write(json.dumps(row) + "\n")
                f.flush()
                if (i + 1) % 10 == 0:
                    print(f"[{variant}] {i + 1}/{len(items)} images, {time.time() - t0:.0f} s", flush=True)
    cmd_score()


def tally(rows):
    keys = ("n", "new4", "new2", "old4", "old2", "ra4", "ra2", "fixed", "broken")
    done = [r for r in rows if not r["unreached"]]
    t = {k: sum(r[k] for r in done) for k in keys}
    f = lambda a, b: round(a / b, 4) if b else None              # noqa: E731
    un = [r for r in rows if r["unreached"]]
    return {"windows": len(done), "unreached_line_end_windows": sum(r.get("reason") == "line_end" for r in un),
            "unreached_cols": sum(r["n"] for r in un),
            "unreached_line_end_cols": sum(r["n"] for r in un if r.get("reason") == "line_end"),
            "unreached_off_image_cols": sum(r["n"] for r in un if r.get("reason") == "off_image"), "cols_scored": t["n"],
            "precision_S4_retrack": f(t["new4"], t["n"]), "precision_S4_sam3_before": f(t["old4"], t["n"]),
            "precision_S4_ra2": f(t["ra4"], t["n"]), "precision_S2_retrack": f(t["new2"], t["n"]),
            "precision_S2_sam3_before": f(t["old2"], t["n"]), "fixed": t["fixed"], "broken": t["broken"],
            "cols_per_click": f(t["n"], len(done)),
            "warn_share": f(sum(r["warn"] for r in done), len(done)),
            "precision_S4_no_warn": f(sum(r["new4"] for r in done if not r["warn"]), sum(r["n"] for r in done if not r["warn"])),
            "precision_S4_warn": f(sum(r["new4"] for r in done if r["warn"]), sum(r["n"] for r in done if r["warn"])),
            "median_secs": float(np.median([r["secs"] for r in done])) if done else None}


def boot(rows, reps=10000, seed=0):
    """event-level percentile 95% interval of the re-track precision (S/4), as research/39 and 41"""
    groups = sorted({r["group"] for r in rows})
    G = np.array([[sum(r["new4"] for r in rows if r["group"] == g and not r["unreached"]),
                   sum(r["n"] for r in rows if r["group"] == g and not r["unreached"])] for g in groups], float)
    draw = np.random.default_rng(seed).integers(0, len(groups), (reps, len(groups)))
    T = G[draw].sum(1)
    return [round(float(v), 4) for v in np.percentile(T[:, 0] / T[:, 1], [2.5, 97.5])]


def after_review(rows):
    """auto-accepted columns (E2 rule) plus the reviewed columns: coverage of ok columns and precision at S/4"""
    rule = json.loads((mc.OUT / "frozen_rule.json").read_text())
    items = mc.load_pairs(SPLIT)
    C, _ = mc.table(items, rule["tau_S"], rule["min_run_cols"])
    out = {}
    for cat in ("quake", "all"):
        j = mc.CATS.index(cat)
        tot = C[:, j].sum(0)
        ok, acc, acc4 = int(tot[0]), int(tot[1]), int(tot[2])
        sel = [r for r in rows if not r["unreached"] and (cat == "all" or r["quake"])]
        rn, r4 = sum(r["n"] for r in sel), sum(r["new4"] for r in sel)
        dis = ok - acc
        out[cat] = {"ok_cols": ok, "auto_accepted": round(acc / ok, 4), "auto_precision_S4": round(acc4 / acc, 4),
                    "disputed_cols": dis, "reviewed_cols": rn, "reviewed_share_of_disputed": round(rn / dis, 4) if dis else None,
                    "after_review_coverage": round((acc + rn) / ok, 4), "after_review_precision_S4": round((acc4 + r4) / (acc + rn), 4)}
    return out


def cmd_score():
    res = {"split": SPLIT, "what": "research/47 E8: one simulated click per review window, SAM 3 film2 re-track, scored on the window's disputed columns"}
    for variant in VARIANTS:
        f = OUT / f"rows_{variant}.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(line) for line in f.read_text().splitlines()]
        res[variant] = {"all": tally(rows), "quake": tally([r for r in rows if r["quake"]]),
                        "ci95_precision_S4": {"all": boot(rows), "quake": boot([r for r in rows if r["quake"]])},
                        "by_gain": {f"g{g:g}": {"all": tally([r for r in rows if r["gain"] == g]),
                                                "quake": tally([r for r in rows if r["gain"] == g and r["quake"]])}
                                    for g in sorted({r["gain"] for r in rows})},
                        "after_review": after_review(rows),
                        "clicks_per_image": round(sum(not r["unreached"] for r in rows) / len({(r["record"], r["gain"]) for r in rows}), 2)}
    (OUT / "e8_results.json").write_text(json.dumps(res, indent=1))
    for variant in VARIANTS:
        if variant in res:
            for cat in ("quake", "all"):
                t = res[variant][cat]
                print(f"[{variant} {cat}] windows {t['windows']} cols {t['cols_scored']}: re-track {t['precision_S4_retrack']} "
                      f"{res[variant]['ci95_precision_S4'][cat]} vs SAM 3 before {t['precision_S4_sam3_before']}, RA2 {t['precision_S4_ra2']} | "
                      f"fixed {t['fixed']} broken {t['broken']} | warn {t['warn_share']} (prec {t['precision_S4_warn']} vs "
                      f"{t['precision_S4_no_warn']}) | unreached {t['unreached_cols']}", flush=True)
            print(f"[{variant}] after review: {res[variant]['after_review']}", flush=True)


if __name__ == "__main__":
    {"run": cmd_run, "score": cmd_score}[sys.argv[1]]()
