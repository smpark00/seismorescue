"""Row-anchor model (research/27 architecture, unchanged) retrained on the calibrated film2 renders (research/31).

usage: python -I scripts/ml_sim2real_train.py split | train [minutes] | val | freeze | cacheB | holdout4 | holdout3 | calib | real | real-alq

- data: out/ml/sim2real/render/train/<record>/film/heli_v<j>.png (6 random film2 variants per record: film_train's
  43 records + extra events); validation = the same four event-date groups as research/27 (split.json there), all
  other records train. holdout4 = out/ml/sim2real/render/holdout4 (fresh events, gains 1, 2, 4, 6, 8), used once.
- init: fine-tuned from out/ml/rowanchor/best.pt (RA v1). Same losses, crops, composites and augmentation.
- selection: checkpoint and decoding are chosen on validation only, by the quake-line r (research/29 definition:
  path deviation vs. GT sig, missing -> 0), fail % reported alongside. frozen.json is written before any test run.
- real: KIP and MAT exactly as ml_rowanchor.cmd_real (research/25 layout, alignment, metric code), with flat and the
  published A/B/judge/DP numbers; RA v1 is re-run with its own frozen config for the same table.
"""
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ml_rowanchor as ra  # noqa: E402
from track_joint import score as tj_score  # noqa: E402

OUT = ROOT / "out" / "ml" / "sim2real" / "ra2"
RENDER = ROOT / "out" / "ml" / "sim2real" / "render"
TRAIN_DIR, HOLD4 = RENDER / "train", RENDER / "holdout4"
HOLD4_CACHE = ROOT / "out" / "ml" / "sim2real" / "cache_holdout4"
G_HOLD = (1, 2, 4, 6, 8)
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
ra.DEV = DEV


def metr(P, pen, sig, base, k, x0, x1, S):
    """research/29-30 r: deviation (base - path)/S with missing -> 0 vs (base - sig)/S where GT exists"""
    a = (base[k] - P[k, x0:x1]) / S
    apen, asig = (base[k] - pen[k, x0:x1]) / S, (base[k] - sig[k, x0:x1]) / S
    ok = np.isfinite(apen) & np.isfinite(asig)
    an = np.nan_to_num(a)[ok]
    t = asig[ok]
    r = float(np.corrcoef(an, t)[0, 1]) if an.std() > 0 and t.std() > 0 else 0.0
    return r


def load_item(rec_dir, tag):
    d = rec_dir / "film"
    meta = json.loads((d / "meta.json").read_text())
    img = cv2.imread(str(d / f"heli_{tag}.png"), 0)
    z = np.load(d / f"gt_{tag}.npz")
    x0 = meta["margin_x"]
    x1 = img.shape[1] - x0
    D, fy, fx = ra.prep(img, x0, x1)
    S = float(meta["spacing"])
    return {"name": rec_dir.name, "tag": tag, "gain": tag, "dir": str(d), "D": D, "fy": fy, "fx": fx,
            "pen": z["pen"].astype(np.float32), "sig": z["sig"].astype(np.float32), "S": S, "x0": x0, "x1": x1,
            "W": img.shape[1], "bases": meta["margin_y"] + S * np.arange(meta["n_lines"]), "act": meta["active_line"],
            "P": meta["variants"][tag]["P"]}


def split():
    f = OUT / "split.json"
    if f.exists():
        return json.loads(f.read_text())
    v1 = json.loads((ra.OUT / "split.json").read_text())
    recs = sorted(p.name for p in TRAIN_DIR.iterdir() if (p / "film" / "meta.json").exists())
    sp = {"rule": "research/27 validation date groups; every other record trains", "val_groups": v1["val_groups"],
          "train": [r for r in recs if r[:8] not in v1["val_groups"]], "val": [r for r in recs if r[:8] in v1["val_groups"]]}
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(sp, indent=1))
    return sp


def tags_of(rec):
    return sorted(json.loads((TRAIN_DIR / rec / "film" / "meta.json").read_text())["variants"])


def load_set(names, root=TRAIN_DIR, tags=None):
    out = []
    for n in names:
        for t in (tags or tags_of(n)):
            out.append(load_item(root / n, t))
    return out


def eval_r(model, items, configs, logT, B_of=None, keep=None):
    """per image: quake / neighbour / whole r and fail % for each decoding (and B if given)"""
    rows = []
    for it in items:
        y0s = it["pen"][:, it["x0"]]
        lp = ra.infer(model, it["D"], it["fy"], it["fx"], it["bases"], y0s, it["x0"], it["x1"])
        u0 = [ra.to_bin(y, b, float(it["x0"])) for y, b in zip(y0s, it["bases"])]
        ests = {}
        for name, a in configs.items():
            if a == "soft":
                ests[name] = decode_soft(lp, it["bases"], it["x0"], it["x1"], it["W"])
                continue
            al, eps = (a, 0.0) if a is None or np.isscalar(a) else a
            ests[name] = ra.decode(lp, it["bases"], it["x0"], it["x1"], it["W"], u0, None if a is None else ra.robust(logT, eps), al or 1.0)
        if B_of is not None:
            ests.update(B_of(it))
        n, q, S, x0, x1 = len(it["bases"]), it["act"], it["S"], it["x0"], it["x1"]
        row = {"name": it["name"], "gain": it["gain"], "P": it["P"], "act": int(q)}
        for m, est in ests.items():
            nb = [k for k in (q - 1, q + 1) if 0 <= k < n]
            rr = [metr(est, it["pen"], it["sig"], it["bases"], k, x0, x1, S) for k in range(n)]
            row[m] = {"r_q": rr[q], "r_nb": float(np.mean([rr[k] for k in nb])), "r_whole": float(np.mean(rr)),
                      "fail_q": tj_score(est, it["pen"], [q], x0, x1, S)[0], "fail_w": tj_score(est, it["pen"], range(n), x0, x1, S)[0]}
        if keep is not None:
            keep[(it["name"], it["gain"])] = ests
        rows.append(row)
    return rows


def decode_soft(logp, bases, x0, x1, W):
    """expected row bin under the per-column distribution (no argmax / Viterbi), interpolated to full-res columns"""
    N, nb, Wc = logp.shape
    p = np.exp(logp)
    u = (p * np.arange(nb)[None, :, None]).sum(1) / np.clip(p.sum(1), 1e-12, None)
    yc = ra.from_bin(u, np.asarray(bases)[:, None])
    xc = x0 + 2 * np.arange(Wc) + 0.5
    est = np.full((N, W), np.nan)
    xs = np.arange(x0, x1)
    for k in range(N):
        est[k, x0:x1] = np.interp(xs, xc, yc[k])
    return est


def summ(rows, methods, key="gain"):
    out = {}
    groups = sorted({r[key] for r in rows}, key=lambda v: (isinstance(v, str), v)) + ["all"]
    for g in groups:
        rr = [r for r in rows if g == "all" or r[key] == g]
        out[str(g)] = {m: {k: round(float(np.mean([r[m][k] for r in rr])), 3 if k.startswith("r") else 1)
                           for k in ("r_q", "r_nb", "r_whole", "fail_q", "fail_w")} for m in methods if m in rr[0]}
        out[str(g)]["n_images"] = len(rr)
    return out


def pbin(P):
    return "P<2" if P < 2 else "P2-4" if P < 4 else "P4-8" if P < 8 else "P>=8"


def cmd_train(minutes):
    sp = split()
    t0 = time.time()
    items = load_set(sp["train"])
    vitems = load_set(sp["val"])
    for r in [*items, *vitems]:
        r["pbin"] = pbin(r["P"])
    idx_tr = list(range(len(items)))
    logT = ra.transition_prior(items, idx_tr)
    np.save(OUT / "logT.npy", logT)
    print(f"loaded {len(items)} train / {len(vitems)} val images in {time.time() - t0:.0f}s; device {DEV}", flush=True)
    base_plan = [(i, k) for i, it in enumerate(items) for k in range(len(it["bases"]))]
    base_plan += [(i, it["act"]) for i, it in enumerate(items)] * 2
    torch.manual_seed(0)
    model = ra.RowAnchor().to(DEV)
    ck = torch.load(ra.OUT / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    LR = 1e-3
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    budget = minutes * 60.0
    rng = np.random.default_rng(11)
    pool = ThreadPoolExecutor(3)
    log, best, step, seed = [], -1e9, 0, 1000

    def plans():
        while True:
            p = rng.permutation(len(base_plan))
            for s in range(0, len(p) - ra.BATCH + 1, ra.BATCH):
                yield [base_plan[q] for q in p[s:s + ra.BATCH]]
    gen = plans()
    fut = [pool.submit(ra.make_batch, items, idx_tr, next(gen), seed + q) for q in range(3)]
    seed += 3

    def validate(el, final=False):
        model.eval()
        rows = eval_r(model, vitems, {"argmax": None}, logT)
        sm = summ(rows, ["argmax"])["all"]["argmax"]
        rec = {"step": step, "minutes": round(el / 60, 2), "train_loss": round(float(np.mean(run_loss[-200:])), 4) if run_loss else None,
               "val_r_q": sm["r_q"], "val_r_nb": sm["r_nb"], "val_r_whole": sm["r_whole"], "val_fail_q": sm["fail_q"],
               "val_fail_w": sm["fail_w"], "final": final}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "step": step, "rec": rec}, OUT / "last.pt")
        (OUT / "train_log.json").write_text(json.dumps(log, indent=1))
        return rec
    run_loss = []
    rec0 = validate(0.0)                      # the RA v1 weights on film2 validation (starting point)
    best = rec0["val_r_q"]
    torch.save({"model": model.state_dict(), "step": 0, "rec": rec0}, OUT / "best.pt")
    tstart, next_val = time.time(), 360.0
    while True:
        el = time.time() - tstart
        if el > budget:
            break
        frac = el / budget
        lr = LR * min(1.0, frac / 0.03) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
        for g_ in opt.param_groups:
            g_["lr"] = max(lr, 1e-5)
        X, U, Wt = fut.pop(0).result()
        fut.append(pool.submit(ra.make_batch, items, idx_tr, next(gen), seed))
        seed += 1
        model.train()
        loss = ra.soft_ce(model(X.to(DEV)), U.to(DEV), Wt.to(DEV))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss.append(float(loss.detach()))
        step += 1
        if el > next_val:
            next_val += 360.0
            rec = validate(el)
            if rec["val_r_q"] > best:
                best = rec["val_r_q"]
                torch.save({"model": model.state_dict(), "step": step, "rec": rec}, OUT / "best.pt")
    rec = validate(time.time() - tstart, final=True)
    if rec["val_r_q"] > best:
        torch.save({"model": model.state_dict(), "step": step, "rec": rec}, OUT / "best.pt")
    pool.shutdown(wait=False, cancel_futures=True)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ra.RowAnchor().to(DEV)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


ALPHAS = {"argmax": None, "viterbi_a0.5": 0.5, "viterbi_a1": 1.0, "viterbi_a2": 2.0, "viterbi_a4": 4.0,
          "viterbi_a1_eps0.1": (1.0, 0.1), "viterbi_a2_eps0.1": (2.0, 0.1), "softmean": "soft"}


def cmd_val():
    sp = split()
    model, ck = load_model(OUT / "best.pt")
    logT = np.load(OUT / "logT.npy")
    vitems = load_set(sp["val"])
    v1, _ = load_model(ra.OUT / "best.pt")
    fz1 = json.loads((ra.OUT / "frozen.json").read_text())
    logT1 = ra.robust(np.load(ra.OUT / "logT.npy"), fz1.get("eps", 0.0))
    rows = eval_r(model, vitems, ALPHAS, logT)
    rows1 = eval_r(v1, vitems, {"RA1_frozen": (fz1["alpha"], 0.0)}, logT1)
    for r, r1 in zip(rows, rows1):
        r["RA1_frozen"] = r1["RA1_frozen"]
        r["pbin"] = pbin(r["P"])
    res = {"checkpoint": ck["rec"], "rows": rows, "summary": summ(rows, list(ALPHAS) + ["RA1_frozen"]),
           "summary_by_P": summ(rows, list(ALPHAS) + ["RA1_frozen"], key="pbin")}
    (OUT / "val_results.json").write_text(json.dumps(res, indent=1))
    for g, v in res["summary"]["all"].items():
        print(g, v)
    for b, v in res["summary_by_P"].items():
        print(b, {m: (v[m]["r_q"], v[m]["fail_q"]) for m in ("argmax", "viterbi_a4", "RA1_frozen") if m in v})


def cmd_freeze():
    """decoding chosen on validation only: highest validation quake-line r (ties -> lower quake fail)"""
    v = json.loads((OUT / "val_results.json").read_text())
    sm = v["summary"]["all"]
    cand = {k: (sm[k]["r_q"], -sm[k]["fail_q"]) for k in ALPHAS}
    best = max(cand, key=cand.get)
    a = ALPHAS[best]
    alpha, eps = (None, 0.0) if a is None or a == "soft" else ((a, 0.0) if np.isscalar(a) else a)
    fz = {"method": "RA2", "config": best, "soft": a == "soft", "alpha": alpha, "eps": float(eps), "checkpoint": "out/ml/sim2real/ra2/best.pt",
          "checkpoint_step": v["checkpoint"]["step"], "rule": "decoding with the highest validation quake-line r (research/29 r); "
          "ties broken by lower quake fail %; chosen before any holdout4 / real run",
          "val_r_q": {k: sm[k]["r_q"] for k in ALPHAS}, "val_fail_q": {k: sm[k]["fail_q"] for k in ALPHAS},
          "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (OUT / "frozen.json").write_text(json.dumps(fz, indent=1))
    print(json.dumps(fz, indent=1))


def frozen_models():
    fz = json.loads((OUT / "frozen.json").read_text())
    m2, _ = load_model(OUT / "best.pt")
    lt2 = ra.robust(np.load(OUT / "logT.npy"), fz["eps"])
    fz1 = json.loads((ra.OUT / "frozen.json").read_text())
    m1, _ = load_model(ra.OUT / "best.pt")
    lt1 = ra.robust(np.load(ra.OUT / "logT.npy"), fz1.get("eps", 0.0))
    return fz, m2, lt2, fz1, m1, lt1


def cmd_cacheB():
    from arbiter import cache_proposals
    cache_proposals(HOLD4, HOLD4_CACHE, [("film", g) for g in G_HOLD])


def run_synth(root, cache, tag, gains):
    fz, m2, lt2, fz1, m1, lt1 = frozen_models()
    names = sorted(p.name for p in root.iterdir() if (p / "film" / "meta.json").exists())
    rows = []
    for n in names:
        for g in gains:
            it = load_item_gain(root / n, g)
            r2 = eval_r(m2, [it], {"RA2": "soft" if fz.get("soft") else fz["alpha"] if fz["eps"] == 0 else (fz["alpha"], 0.0)}, lt2)[0]
            def B_of(it_):
                flat = np.full((len(it_["bases"]), it_["W"]), np.nan)
                flat[:, it_["x0"]:it_["x1"]] = np.asarray(it_["bases"], float)[:, None]
                return {"B": np.load(cache / f"{it_['name']}_film_g{g:g}.npz")["B"], "flat": flat}
            r1 = eval_r(m1, [it], {"RA1": fz1["alpha"]}, lt1, B_of=B_of)[0]
            r2.update({"RA1": r1["RA1"], "B": r1["B"], "flat": r1["flat"]})
            rows.append(r2)
            print(n, g, {m: round(r2[m]["r_q"], 3) for m in ("RA2", "RA1", "B")}, flush=True)
    res = {"frozen_RA2": fz, "frozen_RA1": fz1, "rows": rows, "summary": summ(rows, ["RA2", "RA1", "B", "flat"])}
    (OUT / f"{tag}_results.json").write_text(json.dumps(res, indent=1))
    for g, v in res["summary"].items():
        print(g, json.dumps(v))


def load_item_gain(rec_dir, g):
    """holdout sets: heli_g<gain>.png; film2 meta has variants, film-mode (v1) meta does not"""
    d = rec_dir / "film"
    meta = json.loads((d / "meta.json").read_text())
    if "variants" in meta:
        return load_item(rec_dir, f"g{g:g}")
    it = ra.load_item(rec_dir, g)
    it["sig"] = np.load(d / f"gt_g{g:g}.npz")["sig"].astype(np.float32)
    it["P"] = float(g)
    return it


def real_eval(names, scan_root, tag):
    """ml_rowanchor.cmd_real with RA2 + RA1 + flat, research/25 layout/alignment/metric code; published A/B/judge/DP"""
    import real_paired_eval as rpe
    from obspy import UTCDateTime
    fz, m2, lt2, fz1, m1, lt1 = frozen_models()
    allres = {}
    for name in names:
        cfg = rpe.SCANS[name]
        res = json.loads((scan_root / f"{name}_results.json").read_text())
        img_s = cv2.imread(str(scan_root / name / f"{name}_rescaled.png"), 0)
        z = np.load(scan_root / name / f"{name}_paths.npz")
        bases, y0 = z["bases"], z["y0"]
        L = res["layout"]
        x0, x1, xa, S = L["x0"], L["x1"], L["xa"], rpe.TARGET_S
        W = img_s.shape[1]
        D, fy, fx = ra.prep(img_s, x0, x1)
        u0 = [ra.to_bin(y, b, float(x0)) for y, b in zip(y0, bases)]
        lp2 = ra.infer(m2, D, fy, fx, bases, y0, x0, x1)
        lp1 = ra.infer(m1, D, fy, fx, bases, y0, x0, x1)
        paths = {"RA2": decode_soft(lp2, bases, x0, x1, W) if fz.get("soft") else
                 ra.decode(lp2, bases, x0, x1, W, u0, None if fz["alpha"] is None else lt2, fz["alpha"] or 1.0),
                 "RA2_argmax": ra.decode(lp2, bases, x0, x1, W, u0),
                 "RA1": ra.decode(lp1, bases, x0, x1, W, u0, lt1, fz1["alpha"]),
                 "B_recomputed": z["B"], "flat": np.repeat(np.asarray(bases, float)[:, None], W, 1)}
        np.savez_compressed(OUT / f"{tag}_{name}_paths.npz", **paths)
        al, a_glob = res["alignment"]["chosen_dp_fit"], res["amplitude"]["global_scale_px_per_m"]
        t_on, t_off = UTCDateTime(cfg["on"]), UTCDateTime(cfg["off"])
        t_ref, ref, _ = rpe.reference(cfg, t_on)
        cat = {r["line"]: r["category"] for r in res["lines"]}
        per = {m: {} for m in paths}
        for k in cat:
            idx, _, _ = rpe.sample_path(paths["B_recomputed"], k, x0, x1, xa, al, t_ref, 0.0, float(t_off - t_on))
            tk, rf = t_ref[idx], ref[idx]
            for m, P in paths.items():
                _, y, _ = rpe.sample_path(P, k, x0, x1, xa, al, t_ref, 0.0, float(t_off - t_on))
                per[m][k] = rpe.line_metrics(-(y - bases[k]) * al["sign"], rf, tk, S, a_glob)
        keys = ("r_band", "in_0.1S", "lost_0.5S", "in_0.1S_global", "lost_0.5S_global")
        summ_ = {}
        for c in ("quake", "neighbour", "other_active", "quiet"):
            ks = [k for k, v in cat.items() if v == c]
            if not ks:
                continue
            summ_[c] = {"lines": ks}
            for m in paths:
                summ_[c][m] = {key: round(float(np.nanmean([per[m][k][key] for k in ks])), 3) for key in keys}
            for m in ("A", "B", "judge", "DP", "DP38"):
                if m in res["summary"].get(c, {}):
                    summ_[c][m + "_published"] = {key: res["summary"][c][m][key] for key in keys}
        allres[name] = {"summary": summ_, "per_line": {m: {str(k): v for k, v in d.items()} for m, d in per.items()},
                        "main_shock_line": res["main_shock_line"]}
        for c, v in summ_.items():
            print(name, c, v["lines"], " | ".join(f"{m}: r {v[m]['r_band']:+.2f} in {v[m]['in_0.1S_global']:.2f} lost {v[m]['lost_0.5S_global']:.2f}"
                                                  for m in ("RA2", "RA2_argmax", "RA1", "B_recomputed", "flat")), flush=True)
    (OUT / f"{tag}_results.json").write_text(json.dumps({"frozen_RA2": fz, "frozen_RA1": fz1, "scans": allres}, indent=1))


def cmd_calib():
    """the calibration pair CAL_ALQ73 (used to fit the renderer's exposure model; not a test, not used for selection)"""
    import ml_sim2real_calib as cal
    import real_paired_eval as rpe
    rpe.SCANS.update(cal.PAIRED)
    real_eval(["CAL_ALQ73"], ROOT / "out" / "ml" / "sim2real" / "calib_paired", "calib")


def cmd_real():
    """the TEST pair: run once, after frozen.json exists"""
    assert (OUT / "frozen.json").exists(), "freeze first"
    assert not (OUT / "real_results.json").exists(), "real test already run once"
    real_eval(["KIP", "MAT"], ROOT / "out" / "real_paired", "real")


def cmd_real_alq():
    """research/49: the ALQ 1975 LHZ pair, once, with the frozen model"""
    assert (OUT / "frozen.json").exists(), "freeze first"
    assert not (OUT / "realalq_results.json").exists(), "ALQ already run once"
    real_eval(["ALQ"], ROOT / "out" / "real_paired", "realalq")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "split":
        print(json.dumps(split(), indent=1))
    elif cmd == "train":
        cmd_train(float(sys.argv[2]) if len(sys.argv) > 2 else 60.0)
    elif cmd == "val":
        cmd_val()
    elif cmd == "freeze":
        cmd_freeze()
    elif cmd == "cacheB":
        cmd_cacheB()
    elif cmd == "holdout4":
        assert (OUT / "frozen.json").exists(), "freeze first"
        run_synth(HOLD4, HOLD4_CACHE, "holdout4", G_HOLD)
    elif cmd == "holdout3":
        assert (OUT / "frozen.json").exists(), "freeze first"
        run_synth(ra.HOLD_DIR, ra.HOLD_CACHE, "holdout3", ra.G_HOLD)
    elif cmd == "calib":
        cmd_calib()
    elif cmd == "real":
        cmd_real()
    elif cmd == "real-alq":
        cmd_real_alq()
