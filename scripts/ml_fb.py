"""research/55: the second opinion with no non-Meta model at runtime. The one frozen SAM 3 film2 tracker traces every line from
both ends; a column is accepted where the two traces agree.

usage (GPU commands on the workstation: HF_HUB_OFFLINE=1 .venv-cuda/bin/python -I scripts/ml_fb.py ...):
  check                      first holdout4 image: forward reproduces the stored path, start points, agreement
  dump holdout4|6|7          forward and backward paths of a rendered split -> out/ml/fb/paths/<split>/ (resumes)
  select                     the research/39 grid on holdout4, scored without the minute marks (research/55 §4.1 amendment);
                             writes out/ml/fb/frozen_rule.json once
  score holdout4|6|7         the frozen rule with the flat line beside it -> out/ml/fb/score_<split>.json
  real MAT KIP ALQ           backward on the stored real frames, then research/43's metric -> out/ml/fb/real_results.json
                             (never published: per-line ALQ) and real_results_public.json

Backward is the same model and decoding on the horizontally mirrored image, started at the right edge. Both passes start
from the layout baselines (real_paired_eval.snap_start), as the product does, never from the true pen.
holdout7 is a test split: its start marker is written before the first image and a resumed run must have the same hashes.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("SEISMO_SAM3_OUT", "sam3_film2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ml_consensus as mc                                         # noqa: E402

OUT = ROOT / "out" / "ml" / "fb"
RULE = OUT / "frozen_rule.json"
SPLITS = {"holdout4": ROOT / "out/film2_holdout4", "holdout6": ROOT / "out/film2_holdout6",
          "holdout7": ROOT / "out/film2_holdout7"}
TEST_SPLITS = ("holdout7",)
START_COLS = 120                                                  # as l0_geometry and real_paired_eval
TICK_S = 0.1                                                      # a column is a minute mark where |pen - sig| > 0.1 S


def load_model(device="auto"):
    import ml_sam3 as m3
    fz = json.loads(m3.FROZEN.read_text())
    ckpt = ROOT / fz["ckpt"]
    assert m3.sha256(ckpt) == fz["ckpt_sha256"], "SAM 3 checkpoint is not the frozen file"
    dev = m3.get_device(device)
    model, _ = m3.load_finetuned(ckpt, dev)
    return m3, fz, model, dev


def trace(tracker, img, bases, x0, x1, spacing, tops=None, mirror=False, y0=None):
    """one pass of the frozen tracker over [x0, x1). mirror: the mirrored image, started at x1 - 1; the path is mirrored back.
    y0 overrides the start rows (only to reproduce a stored run)."""
    import real_paired_eval as rpe
    m3, fz, model, dev = tracker
    W = img.shape[1]
    im, a, b = (img[:, ::-1].copy(), W - x1, W - x0) if mirror else (img, x0, x1)
    y = rpe.snap_start(im, np.asarray(bases, float), spacing, a, START_COLS) if y0 is None else y0
    with m3.amp_ctx(dev, bool(fz["amp_bf16"])):
        prob, _ = m3.S.segment_image(model, im, y, a, b, tops=tops, dev=dev)
    P = m3.S.decode_paths(prob, y, a, b, float(fz["lam_final"]), dev, tops=tops)
    return P[:, ::-1] if mirror else P


def film_items(split):
    root = SPLITS[split]
    for rec in sorted(p for p in root.iterdir() if p.is_dir()):
        for g in mc.GAINS:
            yield rec, g


def film_image(rec, g):
    import cv2
    meta = json.loads((rec / "film" / "meta.json").read_text())
    img = cv2.imread(str(rec / "film" / f"heli_g{g:g}.png"), 0)
    S, x0, n = float(meta["spacing"]), int(meta["margin_x"]), int(meta["n_lines"])
    return img, meta["margin_y"] + S * np.arange(n), x0, img.shape[1] - x0, S


def pairs(split):
    d = OUT / "paths" / split
    mc.SETS[split] = (SPLITS[split], d, "F", d, "B")
    return mc.load_pairs(split)


def without_marks(items):
    """the same images with the minute-mark columns left out of the score (research/55 §4.1 amendment). Accept masks do not change."""
    out = []
    for it in items:
        with np.errstate(invalid="ignore"):
            out.append({**it, "ok": it["ok"] & ~(np.abs(it["pen"] - it["sig"]) > TICK_S * it["S"])})
    return out


def cmd_check():
    tracker = load_model()
    rec, g = next(film_items("holdout4"))
    img, bases, x0, x1, S = film_image(rec, g)
    pen = np.load(rec / "film" / f"gt_g{g:g}.npz")["pen"].astype(float)
    stored = np.load(ROOT / "out/ml/sam3_film2/paths/holdout4" / f"{rec.name}_g{g}.npz")["SAM"].astype(float)
    t = time.time()
    F_gt = trace(tracker, img, bases, x0, x1, S, y0=pen[:, x0])
    print(f"{rec.name} g{g}: image {img.shape}, S {S}, one pass {time.time() - t:.1f} s")
    print(f"same call as the stored run: |new - stored| median {np.nanmedian(np.abs(F_gt - stored)):.4f} max {np.nanmax(np.abs(F_gt - stored)):.3f} px")
    F, B = trace(tracker, img, bases, x0, x1, S), trace(tracker, img, bases, x0, x1, S, mirror=True)
    ef, eb = np.abs(F[:, x0:x1] - pen[:, x0:x1]) / S, np.abs(B[:, x0:x1] - pen[:, x0:x1]) / S
    d = np.abs(F[:, x0:x1] - B[:, x0:x1]) / S
    q = json.loads((rec / "film" / "meta.json").read_text())["active_line"]
    print(f"forward within S/4: all {100 * np.nanmean(ef <= .25):.1f}% quake line {q} {100 * np.nanmean(ef[q] <= .25):.1f}%; "
          f"backward: all {100 * np.nanmean(eb <= .25):.1f}% quake {100 * np.nanmean(eb[q] <= .25):.1f}%")
    print(f"|F - B| < 0.15 S: all {100 * np.nanmean(d < .15):.1f}% quake {100 * np.nanmean(d[q] < .15):.1f}%")
    with np.errstate(invalid="ignore"):
        for tau in (.05, .1, .15, .25):
            a = d[q] < tau
            print(f"  quake tau {tau}: accepted {100 * a.mean():.1f}%, forward within S/4 on accepted {100 * np.nanmean(ef[q][a] <= .25):.1f}%")
    print("per line |F-B|<0.15 S:", " ".join(f"{k}:{100 * np.nanmean(d[k] < .15):.0f}" for k in range(len(bases))))


def code_hashes():
    import ml_sam3 as m3
    fz = json.loads(m3.FROZEN.read_text())
    return {"ml_fb": mc.sha_file(__file__), "ml_sam": mc.sha_file(HERE / "ml_sam.py"), "ml_sam3": mc.sha_file(HERE / "ml_sam3.py"),
            "real_paired_eval": mc.sha_file(HERE / "real_paired_eval.py"), "ckpt": fz["ckpt_sha256"], "lam": fz["lam_final"],
            "rule": mc.sha_file(RULE) if RULE.exists() else None}


def cmd_dump(split):
    d = OUT / "paths" / split
    d.mkdir(parents=True, exist_ok=True)
    if split in TEST_SPLITS and not RULE.exists():
        raise SystemExit(f"{split} is a test split: freeze the rule on holdout4 first")
    if not SPLITS[split].is_dir():
        raise SystemExit(f"{split}: {SPLITS[split]} is missing, run make_film2_train.py {split}")
    n_images = sum(1 for _ in film_items(split))
    if split in TEST_SPLITS:
        mark, h = OUT / f"{split}_started.json", code_hashes()
        if not n_images:
            raise SystemExit(f"{split}: no records in {SPLITS[split]}")
        if mark.exists():
            old = json.loads(mark.read_text())
            if old["hashes"] != h or old["n_images"] != n_images:
                raise SystemExit(f"{split} was started with other code, another rule or other images: {old} vs {h}, {n_images} images")
        else:
            if any(d.glob("*.npz")):
                raise SystemExit(f"{split}: path files exist with no start marker in {d}")
            body = json.dumps({"started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "n_images": n_images, "hashes": h}, indent=1)
            with open(mark, "x") as f:
                f.write(body)
    tracker = load_model()
    t0, done = time.time(), 0
    for rec, g in film_items(split):
        f = d / f"{rec.name}_g{g}.npz"
        if f.exists():
            continue
        img, bases, x0, x1, S = film_image(rec, g)
        t = time.time()
        F = trace(tracker, img, bases, x0, x1, S)
        B = trace(tracker, img, bases, x0, x1, S, mirror=True)
        tmp = f.with_name(f.name + ".tmp")                        # not *.npz: mc.sha_dir must not see it, and a killed run leaves no short file
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, F=F.astype(np.float32), B=B.astype(np.float32))
        os.replace(tmp, f)
        done += 1
        print(f"{rec.name} g{g}: {time.time() - t:.1f} s", flush=True)
    print(f"{split}: {done} images in {(time.time() - t0) / 60:.1f} min", flush=True)


def cmd_select():
    if RULE.exists():
        raise SystemExit("frozen_rule.json exists: the rule was already chosen (research/55 §4.1)")
    items = pairs("holdout4")
    assert len(items) == 100, len(items)
    free = without_marks(items)
    n_all, n_free = sum(int(it["ok"].sum()) for it in items), sum(int(it["ok"].sum()) for it in free)
    print(f"minute-mark columns left out of the score: {100 * (n_all - n_free) / n_all:.2f}%")
    ref = mc.summary(free, mc.table(free, None, 1)[0])
    ref_marks = mc.summary(items, mc.table(items, None, 1)[0])
    print("reference (forward alone):", "quake", mc.fmt(ref["gall"]["quake"]), "| all", mc.fmt(ref["gall"]["all"]))
    print("  the same with the marks:", "quake", mc.fmt(ref_marks["gall"]["quake"]), "| all", mc.fmt(ref_marks["gall"]["all"]))
    grid = []
    for tau in mc.TAUS:
        for L in mc.RUNS:
            s = mc.summary(free, mc.table(free, tau, L)[0])["gall"]
            m = mc.summary(items, mc.table(items, tau, L)[0])["gall"]
            feas = s["quake"]["precision"] >= mc.PREC_Q and s["all"]["precision"] >= mc.PREC_ALL
            grid.append({"tau": tau, "L": L, "feasible": bool(feas), **{c: s[c] for c in mc.CATS}, "with_marks": {c: m[c] for c in mc.CATS}})
            print(f"tau {tau:.2f} L {L:2d} | quake {mc.fmt(s['quake'])} | all {mc.fmt(s['all'])} | {'OK' if feas else '--'}"
                  f" | with marks: quake prec {100 * m['quake']['precision']:.2f}% all prec {100 * m['all']['precision']:.2f}%", flush=True)
    feas = [g for g in grid if g["feasible"]]
    if feas:
        best = max(g["quake"]["accept"] for g in feas)
        tied = [g for g in feas if g["quake"]["accept"] >= best - mc.TIE]
        pick = sorted(tied, key=lambda g: (-g["all"]["accept"], g["tau"], -g["L"]))[0]
    else:
        pick = next(g for g in grid if g["tau"] == 0.05 and g["L"] == 60)
    C, accs = mc.table(free, pick["tau"], pick["L"])
    Cm, _ = mc.table(items, pick["tau"], pick["L"])
    constraints = {"quake_precision": mc.PREC_Q, "all_precision": mc.PREC_ALL, "scored_without_minute_marks": f"|pen - sig| > {TICK_S} S"}
    out = {"split": "holdout4", "role": "development (rule selection)", "constraints": constraints,
           "minute_mark_columns_left_out": (n_all - n_free) / n_all, "feasible_any": bool(feas), "selected": {"tau": pick["tau"], "L": pick["L"]},
           "reference_forward_alone": ref, "reference_forward_alone_with_marks": ref_marks, "selected_summary": mc.summary(free, C),
           "selected_summary_with_marks": mc.summary(items, Cm), "bootstrap": mc.bootstrap(free, C), "bootstrap_with_marks": mc.bootstrap(items, Cm),
           "r_accepted_quake": mc.r_accepted(items, accs), "r_accepted_quake_reference": mc.r_accepted(items, [np.isfinite(it["P1"]) for it in items]),
           "grid": grid}
    (OUT / "select_holdout4.json").write_text(json.dumps(out, indent=1))
    rule = {"tau_S": pick["tau"], "min_run_cols": pick["L"], "feasible": bool(feas),
            "accept": "both passes present, |forward - backward| / S < tau_S, inside a run of >= min_run_cols such columns (per line)",
            "export": "the forward path on accepted columns, missing elsewhere",
            "passes": "SAM 3 film2 (out/ml/sam3_film2/frozen_method.json): forward, and backward on the mirrored image",
            "selected_on": "holdout4 (development), research/55 §4.1 with its amendment", "constraints": constraints,
            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "inputs_sha256": {"paths": mc.sha_dir(OUT / "paths" / "holdout4")},
            "code_sha256": mc.sha_file(__file__)}
    with open(RULE, "x") as f:
        f.write(json.dumps(rule, indent=1))
    print(f"SELECTED tau {pick['tau']} L {pick['L']} (feasible: {bool(feas)})")
    for name, summ in (("without marks", out["selected_summary"]), ("with marks", out["selected_summary_with_marks"])):
        print(f" {name}:")
        for c in mc.CATS:
            print(f"  {c:9s} {mc.fmt(summ['gall'][c])}")
        print("  by gain (quake):", " | ".join(f"g{g} {mc.fmt(summ[f'g{g}']['quake'])}" for g in mc.GAINS))


def flat_side(items, accs):
    """the frozen rule beside the flat line on the same columns, with the intervals over events"""
    import flat_baseline as flat
    tot = {c: dict.fromkeys(flat.FIELDS, 0) for c in ("quake", "all")}
    group = {c: {} for c in tot}
    for it, acc in zip(items, accs):
        for k in range(it["n"]):
            n = flat.counts(it, k, acc)
            for c in ("quake", "all") if k == it["q"] else ("all",):
                pg = group[c].setdefault(it["group"], dict.fromkeys(flat.FIELDS, 0))
                for f in flat.FIELDS:
                    tot[c][f] += int(n[f])
                    pg[f] += int(n[f])
    ci = {c: {"accepted_share": flat.boot_ci(group[c], lambda t: 100 * t["acc"] / t["cols"]),
              "accepted_sam3": flat.boot_ci(group[c], lambda t: 100 * t["sam_acc"] / t["acc"]),
              "accepted_flat": flat.boot_ci(group[c], lambda t: 100 * t["flat_acc"] / t["acc"]),
              "accepted_sam3_minus_flat": flat.boot_ci(group[c], lambda t: 100 * (t["sam_acc"] - t["flat_acc"]) / t["acc"]),
              "pen_moved_sam3_alone": flat.boot_ci(group[c], lambda t: 100 * t["moved_sam"] / t["moved"]),
              "pen_moved_share_accepted": flat.boot_ci(group[c], lambda t: 100 * t["moved_acc"] / t["moved"]),
              "pen_moved_accepted": flat.boot_ci(group[c], lambda t: 100 * t["moved_acc_sam"] / t["moved_acc"]),
              "pen_moved_accepted_minus_alone": flat.boot_ci(group[c], lambda t: 100 * (t["moved_acc_sam"] / t["moved_acc"] - t["moved_sam"] / t["moved"]))}
          for c in tot}
    return {c: flat.report(tot[c]) for c in tot}, ci


def cmd_score(split):
    if split in TEST_SPLITS and not (OUT / f"{split}_started.json").exists():
        raise SystemExit(f"{split} has no start marker: its paths did not come from cmd_dump")
    rule = json.loads(RULE.read_text())
    items = pairs(split)
    C, accs = mc.table(items, rule["tau_S"], rule["min_run_cols"])
    summ = mc.summary(items, C)
    with_flat, ci = flat_side(items, accs)
    free = without_marks(items)
    Cf, _ = mc.table(free, rule["tau_S"], rule["min_run_cols"])
    summ_f = mc.summary(free, Cf)
    with_flat_f, ci_f = flat_side(free, accs)
    rq = {m: float(np.mean([mc.r_quake(it, it[k]) for it in items])) for m, k in (("forward", "P1"), ("backward", "P2"))}
    r_acc = mc.r_accepted(items, accs)                            # images with too few accepted columns drop out of r_acc: compare on the same images
    r_same = float(np.mean([mc.r_quake(it, it["P1"]) for it in items if f"{it['record']}_g{it['gain']}" in r_acc["per_image"]]))
    out = {"split": split, "role": "test, once (research/55 §4.2)" if split in TEST_SPLITS else "development or descriptive (research/55)",
           "rule": rule, "n_images": len(items), "n_events": len({it["group"] for it in items}), "summary": summ,
           "with_flat": with_flat, "ci95_over_events": ci, "bootstrap": mc.bootstrap(items, C),
           "r_quake_alone": rq, "r_accepted_quake": r_acc, "r_forward_on_the_accepted_r_images": r_same, "paths_sha256": mc.sha_dir(OUT / "paths" / split),
           "without_minute_marks": {"definition": f"columns with |pen - sig| > {TICK_S} S left out", "summary": summ_f, "with_flat": with_flat_f,
                                    "ci95_over_events": ci_f, "bootstrap": mc.bootstrap(free, Cf)}}
    (OUT / f"score_{split}.json").write_text(json.dumps(out, indent=1))
    print(f"{split}: {len(items)} images, {out['n_events']} events, rule tau {rule['tau_S']} L {rule['min_run_cols']}")
    for name, sm, wf, cc in (("as registered (minute marks scored)", summ, with_flat, ci), ("without the minute marks", summ_f, with_flat_f, ci_f)):
        print(f" {name}:")
        for g in [*(f"g{x}" for x in mc.GAINS), "gall"]:
            print(f"  {g:5s} quake {mc.fmt(sm[g]['quake'])} | all {mc.fmt(sm[g]['all'])}")
        for c in ("quake", "all"):
            print(c, json.dumps({k: v for k, v in wf[c].items() if k != "columns"}))
            print(c, "ci", json.dumps(cc[c]))
    print("r alone", json.dumps(rq), "r on accepted", r_acc["mean"], f"(n={r_acc['n_images']}), forward alone on those images", r_same)


def real_frame(name):
    """the research/25 frame of a scan, as ml_sam3.eval_real reads it"""
    import cv2
    rp = ROOT / "out" / "real_paired"
    img = cv2.imread(str(rp / name / f"{name}_rescaled.png"), 0)
    z = np.load(rp / name / f"{name}_paths.npz")
    res = json.loads((rp / f"{name}_results.json").read_text())
    return img, z, res


def cmd_real(names):
    """research/43 metric code on the stored frozen forward path and a new backward path. Line sampling and rescoring only."""
    import real_paired_eval as rpe
    from obspy import UTCDateTime
    import ml_sam3 as m3
    rule = json.loads(RULE.read_text())
    mark = OUT / "real_started.json"
    if not mark.exists():
        mark.write_text(json.dumps({"started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rule": rule, "code_sha256": mc.sha_file(__file__)}, indent=1))
    tracker = None
    results = OUT / "real_results.json"
    out = json.loads(results.read_text()) if results.exists() else {"frame": "research/25 layout, chosen_dp_fit alignment, global amplitude", "scans": {}}
    out["rule"] = rule
    for name in names:
        img, z, res = real_frame(name)
        bases, lay = z["bases"], res["layout"]
        x0, x1, xa, S = lay["x0"], lay["x1"], lay["xa"], rpe.TARGET_S
        H = img.shape[0]
        tops = np.clip(np.round(bases).astype(int) - m3.S.WIN_H // 2, 0, H - m3.S.WIN_H)
        P1 = np.load(ROOT / "out" / "ml" / "sam3_film2" / f"real_{name}_paths.npz")["SAM"].astype(float)
        cfg = rpe.SCANS[name]
        t_on, t_off = UTCDateTime(cfg["on"]), UTCDateTime(cfg["off"])
        t_ref, ref, _ = rpe.reference(cfg, t_on)
        al, a_glob = res["alignment"]["chosen_dp_fit"], res["amplitude"]["global_scale_px_per_m"]
        pf = OUT / f"real_{name}_paths.npz"
        if pf.exists():
            P2 = np.load(pf)["B"].astype(float)
        else:
            tracker = tracker or load_model()
            t = time.time()
            P2 = trace(tracker, img, bases, x0, x1, S, tops=tops, mirror=True)
            last = len(bases) - 1                                 # a partial last line: start where its trace ends
            idx, _, xcol = rpe.sample_path(P1, last, x0, x1, xa, al, t_ref, 0.0, float(t_off - t_on))
            if idx.size and x1 - 1 - float(xcol[idx].max()) > 100:
                xe = int(round(float(xcol[idx].max()))) + 1
                P2[last] = trace(tracker, img, bases[last:], x0, xe, S, tops=tops[last:], mirror=True)[0]
                print(f"[{name}] last line re-started at x {xe}")
            print(f"[{name}] backward {time.time() - t:.0f} s", flush=True)
            np.savez_compressed(pf, B=P2.astype(np.float32), tops=tops)
        assert P1.shape == P2.shape == z["B"].shape, (P1.shape, P2.shape, z["B"].shape)
        with np.errstate(invalid="ignore"):
            d = np.where(np.isfinite(P1[:, x0:x1]) & np.isfinite(P2[:, x0:x1]), np.abs(P1[:, x0:x1] - P2[:, x0:x1]) / S, np.inf)
        acc = np.stack([mc.runs_ge(a, rule["min_run_cols"]) for a in d < rule["tau_S"]])
        cats = {ln["line"]: ln["category"] for ln in res["lines"]}
        per, repro = {}, []
        for k in sorted(cats):
            idx, y, xcol = rpe.sample_path(P1, k, x0, x1, xa, al, t_ref, 0.0, float(t_off - t_on))
            if idx.size < 1200:                                   # same skip rule as ml_sam.real_metrics
                continue
            rf, tk = ref[idx], t_ref[idx]
            dev = -(y - bases[k]) * al["sign"]
            m_acc = acc[k, np.clip(np.round(xcol[idx]).astype(int), x0, x1 - 1) - x0]
            full = rpe.line_metrics(dev, rf, tk, S, a_glob)
            db, rb, dl, rl = rpe.bandpass(dev), rpe.bandpass(rf), rpe.lowpass(dev), rpe.lowpass(rf)
            tt = tk - tk.mean()
            Xg = np.c_[np.ones_like(tt), tt]
            eg = dl - a_glob * rl - Xg @ np.linalg.lstsq(Xg, dl - a_glob * rl, rcond=None)[0]

            def part(m):
                if m.sum() < 300:
                    return None
                return {"n": int(m.sum()), "in_0.1S_global": float(np.mean(np.abs(eg[m]) < 0.1 * S)),
                        "lost_0.5S_global": float(np.mean(np.abs(eg[m]) > 0.5 * S)), "r_band": float(rpe.pearson(db[m], rb[m]))}
            allp = part(np.ones_like(m_acc))
            repro.append(max(abs(allp[key] - full[key]) for key in ("in_0.1S_global", "lost_0.5S_global", "r_band")))
            per[k] = {"category": cats[k], "accept_frac": float(m_acc.mean()), "n": int(idx.size), "all": allp,
                      "accepted": part(m_acc), "disputed": part(~m_acc)}
        summ = {}
        for cat in ("quake", "neighbour", "other_active", "quiet"):
            ks = [k for k, v in per.items() if v["category"] == cat]
            if not ks:
                continue
            summ[cat] = {"lines": ks, "accept_frac": float(np.mean([per[k]["accept_frac"] for k in ks]))}
            for prt in ("all", "accepted", "disputed"):
                vals = [per[k][prt] for k in ks if per[k][prt]]
                summ[cat][prt] = ({key: float(np.mean([v[key] for v in vals])) for key in ("in_0.1S_global", "lost_0.5S_global", "r_band")}
                                  | {"n_lines": len(vals)}) if vals else None
        out["scans"][name] = {"summary": summ, "per_line": {str(k): v for k, v in per.items()}, "main_shock_line": res["main_shock_line"],
                              "all_vs_research25_metric_max_abs_diff": float(max(repro))}
        print(f"[{name}] 'all' subset vs research/25 line_metrics: max |diff| {max(repro):.4f} (rounding 1e-3)")
        for cat, v in summ.items():
            f3 = lambda e: "-" if e is None else f"in {e['in_0.1S_global']:.2f} lost {e['lost_0.5S_global']:.2f} r {e['r_band']:+.2f} ({e['n_lines']})"  # noqa: E731
            print(f"[{name}] {cat:12s} lines {v['lines']} accept {100 * v['accept_frac']:5.1f}% | all {f3(v['all'])} | "
                  f"accepted {f3(v['accepted'])} | disputed {f3(v['disputed'])}", flush=True)
        results.write_text(json.dumps(out, indent=1))
        # what may go in the repository: ALQ by line type only, no line numbers (data rule, research/49)
        (OUT / "real_results_public.json").write_text(json.dumps({**out, "scans": {
            n: ({"summary": {c: {k: v for k, v in sm.items() if k != "lines"} for c, sm in sc["summary"].items()}} if n == "ALQ" else sc)
            for n, sc in out["scans"].items()}}, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "check":
        cmd_check()
    elif cmd == "dump":
        cmd_dump(args[0])
    elif cmd == "select":
        cmd_select()
    elif cmd == "score":
        cmd_score(args[0])
    elif cmd == "real":
        cmd_real(args)
    else:
        raise SystemExit(__doc__)
