"""E2 (research/39): two-model consensus rule, chosen on holdout4 (development) and frozen for E3 (holdout6).

usage: python -I scripts/ml_consensus.py dump | select | describe | e3 | real

- dump: RA2 paths on holdout4 and RA v1 paths on holdout3 (research/31 and /27 saved none), regenerated with the frozen
  models through the same eval code; the saved float32 paths must reproduce the stored per-image metrics.
- select: a (line, column) is accepted iff both paths exist, |SAM 3 - RA2| / S < tau and it lies in a run of >= L such
  columns; the exported value is SAM 3's. The research/39 §4 grid is scored on holdout4 and the chosen rule is written
  once to out/ml/consensus/frozen_rule.json.
- describe: the frozen rule, unchanged, on the v1 pairs (SAM 3 v1 + RA v1) of holdout5 and holdout3 (descriptive only).
- e3 (research/41): once, after the SAM 3 film2 holdout6 run. RA2 paths on holdout6 with the frozen model, then the
  frozen rule, with per-gain targets and the quake-line r of each model alone.
- real (research/43 §3): once. The frozen rule on the stored real SAM 3 film2 / RA2 paths of MAT and KIP (research/25
  frame); research/25's reference samples split into accepted / disputed, with its metric code on each subset.
- real-alq (research/49 §4): once. The same code on the ALQ 1975 LHZ pair, which no model or rule was developed on.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

OUT = ROOT / "out" / "ml" / "consensus"
TAUS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)    # in line spacings S
RUNS = (1, 20, 60)                                          # minimum accepted run, columns
PREC_Q, PREC_ALL, TIE = 0.95, 0.99, 0.001
GAINS = (1, 2, 4, 6, 8)
CATS = ("quake", "neighbour", "other", "all")
FIELDS = ("ok", "acc", "acc_c4", "acc_c2", "w4", "w4_disp", "w2", "w2_disp", "disp_c4")
SETS = {  # split -> GT renders, primary (SAM 3) paths and key, second-opinion paths and key
    "holdout4": (ROOT / "out/film2_holdout4", ROOT / "out/ml/sam3_film2/paths/holdout4", "SAM", OUT / "paths/holdout4_RA2", "RA2"),
    "holdout5": (ROOT / "out/film_holdout5", ROOT / "out/ml/sam3_modal/paths/holdout5", "SAM", ROOT / "out/ml/rowanchor/paths/holdout5", "RA"),
    "holdout3": (ROOT / "out/film_holdout3", ROOT / "out/ml/sam3_modal/paths/holdout3", "SAM", OUT / "paths/holdout3_RA1", "RA"),
    "holdout6": (ROOT / "out/film2_holdout6", ROOT / "out/ml/sam3_film2/paths/holdout6", "SAM", OUT / "paths/holdout6_RA2", "RA2"),
}
R_TOL, FAIL_TOL = 1e-3, 0.2


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_dir(d):
    h = hashlib.sha256()
    for f in sorted(Path(d).glob("*.npz")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- dump: regenerate the second-opinion paths
def cmd_dump():
    import ml_sim2real_train as s2r
    import ml_rowanchor as ra
    fz, m2, lt2, fz1, m1, lt1 = s2r.frozen_models()
    assert fz.get("soft"), "research/31 froze soft-mean RA2"

    # RA2 on holdout4: exactly run_synth's call, scored again from the float32 array that E2 reads
    stored = {(r["name"], r["gain"]): r["RA2"] for r in json.loads((s2r.OUT / "holdout4_results.json").read_text())["rows"]}
    pdir = OUT / "paths" / "holdout4_RA2"
    pdir.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in s2r.HOLD4.iterdir() if (p / "film" / "meta.json").exists())
    devs = []
    for n in names:
        for g in s2r.G_HOLD:
            it = s2r.load_item_gain(s2r.HOLD4 / n, g)
            keep = {}
            s2r.eval_r(m2, [it], {"RA2": "soft"}, lt2, keep=keep)
            np.savez_compressed(pdir / f"{n}_g{g}.npz", RA2=keep[(n, it["gain"])]["RA2"].astype(np.float32))
            P = np.load(pdir / f"{n}_g{g}.npz")["RA2"]
            q, nl, S, x0, x1 = it["act"], len(it["bases"]), it["S"], it["x0"], it["x1"]
            rr = [s2r.metr(P, it["pen"], it["sig"], it["bases"], k, x0, x1, S) for k in range(nl)]
            new = {"r_q": rr[q], "r_nb": float(np.mean([rr[k] for k in (q - 1, q + 1) if 0 <= k < nl])), "r_whole": float(np.mean(rr)),
                   "fail_q": s2r.tj_score(P, it["pen"], [q], x0, x1, S)[0], "fail_w": s2r.tj_score(P, it["pen"], range(nl), x0, x1, S)[0]}
            old = stored[(n, it["gain"])]
            devs.append({"image": f"{n}_g{g}", **{k: abs(new[k] - old[k]) for k in new}})
    mx = {k: max(d[k] for d in devs) for k in ("r_q", "r_nb", "r_whole", "fail_q", "fail_w")}
    ok = all(mx[k] <= R_TOL for k in ("r_q", "r_nb", "r_whole")) and all(mx[k] <= FAIL_TOL for k in ("fail_q", "fail_w"))
    rep = {"what": "RA2 holdout4 paths regenerated vs research/31 stored per-image metrics", "n_images": len(devs),
           "max_abs_diff": mx, "tolerance": {"r": R_TOL, "fail_pp": FAIL_TOL}, "pass": ok, "frozen": fz,
           "ckpt_sha256": sha_file(s2r.OUT / "best.pt"), "paths_sha256": sha_dir(pdir), "device": str(s2r.DEV),
           "at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "per_image": devs}
    (OUT / "repro_holdout4_RA2.json").write_text(json.dumps(rep, indent=1))
    print(f"RA2 holdout4: {len(devs)} images, max |diff| {mx} -> {'PASS' if ok else 'FAIL'}", flush=True)

    # RA v1 on holdout3: exactly ml_rowanchor.cmd_holdout's call
    res3 = json.loads((ra.OUT / "holdout3_results.json").read_text())
    for k in ("method", "alpha", "eps", "checkpoint_step"):
        assert res3["frozen"].get(k) == fz1.get(k), (k, res3["frozen"].get(k), fz1.get(k))
    stored1 = {(r["name"], r["gain"]): r[fz1["method"]]["per_line"] for r in res3["rows"]}
    pdir1 = OUT / "paths" / "holdout3_RA1"
    pdir1.mkdir(parents=True, exist_ok=True)
    devs1 = []
    for n in sorted(p.name for p in ra.HOLD_DIR.iterdir() if p.is_dir()):
        for g in ra.G_HOLD:
            it = ra.load_item(ra.HOLD_DIR / n, g)
            keep = {}
            ra.eval_items(m1, [it], {fz1["method"]: (fz1["alpha"], 0.0)}, lt1, keep=keep)
            np.savez_compressed(pdir1 / f"{n}_g{g}.npz", RA=keep[(n, g)][fz1["method"]].astype(np.float32))
            P = np.load(pdir1 / f"{n}_g{g}.npz")["RA"]
            new = [ra.tj_score(P, it["pen"], [k], it["x0"], it["x1"], it["S"])[0] for k in range(len(it["bases"]))]
            devs1.append({"image": f"{n}_g{g}", "max_line_fail_diff": max(abs(a - b) for a, b in zip(new, stored1[(n, g)]))})
    mx1 = max(d["max_line_fail_diff"] for d in devs1)
    rep1 = {"what": "RA v1 holdout3 paths regenerated vs research/27 stored per-line fail %", "n_images": len(devs1),
            "max_abs_line_fail_diff_pp": mx1, "tolerance_pp": FAIL_TOL, "pass": mx1 <= FAIL_TOL, "frozen": fz1,
            "paths_sha256": sha_dir(pdir1), "device": str(s2r.DEV), "at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "per_image": devs1}
    (OUT / "repro_holdout3_RA1.json").write_text(json.dumps(rep1, indent=1))
    print(f"RA v1 holdout3: {len(devs1)} images, max per-line fail |diff| {mx1:.2f} pp -> {'PASS' if mx1 <= FAIL_TOL else 'FAIL'}")


# ---------------------------------------------------------------- consensus metrics
def load_pairs(split):
    """per image: GT, geometry, the two paths, and d = |P1 - P2| / S, e = |P1 - pen| / S on [x0, x1)"""
    gt_root, d1, k1, d2, k2 = SETS[split]
    items = []
    for rec in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        meta = json.loads((rec / "film" / "meta.json").read_text())
        S, x0, n = float(meta["spacing"]), meta["margin_x"], meta["n_lines"]
        for g in GAINS:
            gt = np.load(rec / "film" / f"gt_g{g}.npz")
            pen, sig = gt["pen"].astype(float), gt["sig"].astype(float)
            P1 = np.load(d1 / f"{rec.name}_g{g}.npz")[k1].astype(float)
            P2 = np.load(d2 / f"{rec.name}_g{g}.npz")[k2].astype(float)
            assert P1.shape == P2.shape == pen.shape, (rec.name, g, P1.shape, P2.shape, pen.shape)
            sl = slice(x0, pen.shape[1] - x0)
            with np.errstate(invalid="ignore"):
                d = np.where(np.isfinite(P1[:, sl]) & np.isfinite(P2[:, sl]), np.abs(P1[:, sl] - P2[:, sl]) / S, np.inf)
                e = np.where(np.isfinite(P1[:, sl]), np.abs(P1[:, sl] - pen[:, sl]) / S, np.inf)   # NaN where pen is missing
            items.append({"record": rec.name, "group": rec.name[:8], "gain": g, "S": S, "q": meta["active_line"], "n": n,
                          "base": meta["margin_y"] + S * np.arange(n), "P1": P1[:, sl], "P2": P2[:, sl], "pen": pen[:, sl], "sig": sig[:, sl],
                          "d": d, "e": e, "ok": np.isfinite(pen[:, sl])})
    return items


def runs_ge(mask, L):
    """drop True runs shorter than L columns (1-D)"""
    if L <= 1:
        return mask
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    out = mask.copy()
    for s, t in zip(e[::2], e[1::2]):
        if t - s < L:
            out[s:t] = False
    return out


def accept(it, tau, L):
    if tau is None:                                   # reference: no consensus, every column where SAM 3 has a value
        return np.isfinite(it["P1"])
    return np.stack([runs_ge(a, L) for a in it["d"] < tau])


def cat_of(k, q):
    return "quake" if k == q else "neighbour" if abs(k - q) == 1 else "other"


def table(items, tau, L):
    """(n_images, CATS, FIELDS) column counts and the accept masks"""
    C = np.zeros((len(items), len(CATS), len(FIELDS)), np.int64)
    accs = []
    for i, it in enumerate(items):
        acc = accept(it, tau, L)
        ok, e = it["ok"], it["e"]
        with np.errstate(invalid="ignore"):
            c4, c2 = ok & (e <= 0.25), ok & (e <= 0.5)
        w4, w2, a = ok & ~c4, ok & ~c2, acc & ok
        pl = np.stack([ok.sum(1), a.sum(1), (a & c4).sum(1), (a & c2).sum(1), w4.sum(1), (w4 & ~acc).sum(1),
                       w2.sum(1), (w2 & ~acc).sum(1), (ok & ~acc & c4).sum(1)], 1)
        for k in range(it["n"]):
            C[i, CATS.index(cat_of(k, it["q"]))] += pl[k]
        C[i, CATS.index("all")] = pl.sum(0)
        accs.append(acc)
    return C, accs


def rates(c):
    ok, acc, a4, a2, w4, w4d, w2, w2d, d4 = (int(v) for v in c)
    f = lambda a, b: a / b if b else float("nan")  # noqa: E731
    return {"accept": f(acc, ok), "precision": f(a4, acc), "precision_S2": f(a2, acc), "capture": f(w4d, w4),
            "capture_S2": f(w2d, w2), "abw_line": f(acc - a2, acc), "abw_near": f(a2 - a4, acc), "review": f(ok - acc, ok),
            "wasted_review": f(d4, ok - acc), "n_cols": ok, "n_acc": acc, "n_err": w4}


def summary(items, C):
    out = {}
    for g in [*GAINS, "all"]:
        tot = C[[i for i, it in enumerate(items) if g == "all" or it["gain"] == g]].sum(0)
        out[f"g{g}"] = {c: rates(tot[j]) for j, c in enumerate(CATS)}
    return out


def r_accepted(items, accs, min_cols=200):
    """quake-line r on accepted columns only: (base - P1)/S vs (base - sig)/S, per image with >= min_cols columns"""
    rs = {}
    for it, acc in zip(items, accs):
        k = it["q"]
        m = acc[k] & np.isfinite(it["pen"][k]) & np.isfinite(it["sig"][k])
        if m.sum() < min_cols:
            continue
        a, t = (it["base"][k] - it["P1"][k][m]) / it["S"], (it["base"][k] - it["sig"][k][m]) / it["S"]
        rs[f"{it['record']}_g{it['gain']}"] = float(np.corrcoef(a, t)[0, 1]) if a.std() > 0 and t.std() > 0 else 0.0
    by_gain = {f"g{g}": [v for n, v in rs.items() if n.endswith(f"_g{g}")] for g in GAINS}
    return {"mean": float(np.mean(list(rs.values()))) if rs else None, "n_images": len(rs), "n_total": len(items),
            "min_cols": min_cols, "by_gain": {g: {"mean": float(np.mean(v)) if v else None, "n": len(v)} for g, v in by_gain.items()},
            "per_image": rs}


def bootstrap(items, C, reps=10000, seed=0):
    """event-level percentile 95% intervals for accept / precision / capture (quake and all lines)"""
    groups = sorted({it["group"] for it in items})
    G = np.stack([C[[i for i, it in enumerate(items) if it["group"] == g]].sum(0) for g in groups])
    draw = np.random.default_rng(seed).integers(0, len(groups), (reps, len(groups)))
    T = G[draw].sum(1).astype(float)                                  # (reps, CATS, FIELDS)
    F = {f: T[..., i] for i, f in enumerate(FIELDS)}
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = {"accept": F["acc"] / F["ok"], "precision": F["acc_c4"] / F["acc"], "capture": F["w4_disp"] / F["w4"],
                "abw": 1 - F["acc_c4"] / F["acc"]}
    out = {}
    for c in ("quake", "all"):
        j = CATS.index(c)
        for k, v in vals.items():
            lo, hi = np.nanpercentile(v[:, j], [2.5, 97.5])
            out[f"{c} {k}"] = [float(lo), float(hi)]
    return {"n_events": len(groups), "reps": reps, "seed": seed, "ci95": out}


def fmt(r):
    return f"acc {100 * r['accept']:5.1f}%  prec {100 * r['precision']:5.2f}%  capt {100 * r['capture']:5.1f}%"


# ---------------------------------------------------------------- select (holdout4, once) and describe
def cmd_select():
    fr = OUT / "frozen_rule.json"
    if fr.exists():
        raise SystemExit("frozen_rule.json exists: the rule was already chosen (research/39 §4)")
    rep = json.loads((OUT / "repro_holdout4_RA2.json").read_text())
    assert rep["pass"], "RA2 reproduction check failed: E2 stops (research/39 §2)"
    items = load_pairs("holdout4")
    assert len(items) == 100, len(items)
    ref_C, _ = table(items, None, 1)
    ref = summary(items, ref_C)
    print("reference (SAM 3 alone):", "quake", fmt(ref["gall"]["quake"]), "| all", fmt(ref["gall"]["all"]))
    grid = []
    for tau in TAUS:
        for L in RUNS:
            C, _ = table(items, tau, L)
            s = summary(items, C)["gall"]
            feas = s["quake"]["precision"] >= PREC_Q and s["all"]["precision"] >= PREC_ALL
            grid.append({"tau": tau, "L": L, "feasible": bool(feas), **{c: s[c] for c in CATS}})
            print(f"tau {tau:.2f} L {L:2d} | quake {fmt(s['quake'])} | all {fmt(s['all'])} | {'OK' if feas else '--'}", flush=True)
    feas = [g for g in grid if g["feasible"]]
    if feas:
        best = max(g["quake"]["accept"] for g in feas)
        tied = [g for g in feas if g["quake"]["accept"] >= best - TIE]
        pick = sorted(tied, key=lambda g: (-g["all"]["accept"], g["tau"], -g["L"]))[0]
    else:
        pick = next(g for g in grid if g["tau"] == 0.05 and g["L"] == 60)
    C, accs = table(items, pick["tau"], pick["L"])
    out = {"split": "holdout4", "role": "development (rule selection)", "constraints": {"quake_precision": PREC_Q, "all_precision": PREC_ALL},
           "feasible_any": bool(feas), "selected": {"tau": pick["tau"], "L": pick["L"]},
           "tied_candidates": [(g["tau"], g["L"]) for g in feas if g["quake"]["accept"] >= max(x["quake"]["accept"] for x in feas) - TIE] if feas else [],
           "reference_sam3_alone": ref, "selected_summary": summary(items, C), "bootstrap": bootstrap(items, C),
           "r_accepted_quake": r_accepted(items, accs), "r_accepted_quake_reference": r_accepted(items, [np.isfinite(it["P1"]) for it in items]),
           "grid": grid}
    rule = {"tau_S": pick["tau"], "min_run_cols": pick["L"], "feasible": bool(feas),
            "accept": "both paths present, |SAM 3 - RA2| / S < tau_S, inside a run of >= min_run_cols such columns (per line)",
            "export": "SAM 3 path on accepted columns, missing elsewhere", "primary": "SAM 3 film2 (out/ml/sam3_film2/frozen_method.json)",
            "second": "RA2 soft-mean (out/ml/sim2real/ra2/frozen.json)", "selected_on": "holdout4 (development), research/39 §4",
            "constraints": out["constraints"], "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "inputs_sha256": {"sam3_paths": sha_dir(SETS["holdout4"][1]), "ra2_paths": sha_dir(SETS["holdout4"][3]),
                              "sam3_frozen_method": sha_file(ROOT / "out/ml/sam3_film2/frozen_method.json"),
                              "ra2_frozen": sha_file(ROOT / "out/ml/sim2real/ra2/frozen.json")},
            "code_sha256": sha_file(__file__)}
    (OUT / "e2_holdout4.json").write_text(json.dumps(out, indent=1))
    with open(fr, "x") as f:
        f.write(json.dumps(rule, indent=1))
    s = out["selected_summary"]["gall"]
    print(f"SELECTED tau {pick['tau']} L {pick['L']} (feasible: {bool(feas)})")
    for c in CATS:
        print(f"  {c:9s} {fmt(s[c])}  abw line {100 * s[c]['abw_line']:.2f}% near {100 * s[c]['abw_near']:.2f}%  wasted {100 * s[c]['wasted_review']:.1f}%")
    print("  by gain (quake):", " | ".join(f"g{g} {fmt(out['selected_summary'][f'g{g}']['quake'])}" for g in GAINS))
    print("  bootstrap:", json.dumps(out["bootstrap"]["ci95"]))
    ra_, rr = out["r_accepted_quake"], out["r_accepted_quake_reference"]
    print(f"  quake r on accepted columns: {ra_['mean']} (n={ra_['n_images']}/{ra_['n_total']}); SAM 3 alone same r: {rr['mean']} (n={rr['n_images']})")


def cmd_describe():
    rule = json.loads((OUT / "frozen_rule.json").read_text())
    out = {"rule": rule, "note": "descriptive only: frozen rule unchanged on the v1 pairs (SAM 3 v1 + RA v1), v1 renderer"}
    for split in ("holdout5", "holdout3"):
        if split == "holdout3" and not json.loads((OUT / "repro_holdout3_RA1.json").read_text())["pass"]:
            out[split] = "skipped: RA v1 reproduction check failed (research/39 §5)"
            print(split, out[split])
            continue
        items = load_pairs(split)
        C, accs = table(items, rule["tau_S"], rule["min_run_cols"])
        ref_C, _ = table(items, None, 1)
        out[split] = {"n_images": len(items), "summary": summary(items, C), "reference_sam3_alone": summary(items, ref_C),
                      "bootstrap": bootstrap(items, C), "r_accepted_quake": r_accepted(items, accs)}
        s, r0 = out[split]["summary"]["gall"], out[split]["reference_sam3_alone"]["gall"]
        print(f"{split} ({len(items)} images) quake {fmt(s['quake'])} | all {fmt(s['all'])} || SAM 3 v1 alone: quake {fmt(r0['quake'])}")
    (OUT / "e2_v1_descriptive.json").write_text(json.dumps(out, indent=1))


def r_quake(it, P):
    """research/29 r on the scored span (= ml_sim2real_train.metr / E1): (base - P)/S, missing -> 0, vs (base - sig)/S"""
    k, S = it["q"], it["S"]
    a, apen, asig = (it["base"][k] - P[k]) / S, (it["base"][k] - it["pen"][k]) / S, (it["base"][k] - it["sig"][k]) / S
    ok = np.isfinite(apen) & np.isfinite(asig)
    an = np.nan_to_num(a)[ok]
    return float(np.corrcoef(an, asig[ok])[0, 1]) if an.std() > 0 and asig[ok].std() > 0 else 0.0


def cmd_e3():
    import ml_sim2real_train as s2r
    rule = json.loads((OUT / "frozen_rule.json").read_text())
    gt_root, sam_dir = SETS["holdout6"][0], SETS["holdout6"][1]
    names = sorted(p.name for p in gt_root.iterdir() if p.is_dir())
    missing = [f"{n}_g{g}" for n in names for g in GAINS if not (sam_dir / f"{n}_g{g}.npz").exists()]
    assert not missing, f"SAM 3 film2 holdout6 paths missing: {missing[:3]}"
    with open(OUT / "e3_started.json", "x") as f:
        f.write(json.dumps({"started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rule": rule, "n_images": len(names) * len(GAINS),
                            "code_sha256": sha_file(__file__)}, indent=1))
    fz, m2, lt2, *_ = s2r.frozen_models()
    pdir = SETS["holdout6"][3]
    pdir.mkdir(parents=True, exist_ok=True)
    for n in names:                                    # same call as research/31 run_synth (soft-mean RA2)
        for g in GAINS:
            it = s2r.load_item_gain(s2r.RENDER / "holdout6" / n, g)
            keep = {}
            s2r.eval_r(m2, [it], {"RA2": "soft"}, lt2, keep=keep)
            np.savez_compressed(pdir / f"{n}_g{g}.npz", RA2=keep[(n, it["gain"])]["RA2"].astype(np.float32))
    items = load_pairs("holdout6")
    C, accs = table(items, rule["tau_S"], rule["min_run_cols"])
    ref_C, _ = table(items, None, 1)
    summ = summary(items, C)
    targets = {g: {"quake_precision_ge_0.95": bool(summ[g]["quake"]["precision"] >= PREC_Q),
                   "all_precision_ge_0.99": bool(summ[g]["all"]["precision"] >= PREC_ALL)} for g in summ}
    rq = {m: {f"g{g}": float(np.mean([r_quake(it, it[k]) for it in items if it["gain"] == g])) for g in GAINS}
          | {"gall": float(np.mean([r_quake(it, it[k]) for it in items]))} for m, k in (("SAM3_film2", "P1"), ("RA2", "P2"))}
    out = {"split": "holdout6", "role": "test (research/41), once", "rule": rule, "n_images": len(items),
           "n_events": len({it["group"] for it in items}), "summary": summ, "targets": targets,
           "reference_sam3_alone": summary(items, ref_C), "bootstrap": bootstrap(items, C), "r_quake_alone": rq,
           "r_accepted_quake": r_accepted(items, accs), "r_accepted_quake_reference": r_accepted(items, [np.isfinite(it["P1"]) for it in items]),
           "inputs_sha256": {"sam3_paths": sha_dir(sam_dir), "ra2_paths": sha_dir(pdir)}}
    (OUT / "e3_holdout6.json").write_text(json.dumps(out, indent=1))
    print(f"E3 holdout6: {len(items)} images, {out['n_events']} events, rule tau {rule['tau_S']} L {rule['min_run_cols']}")
    for g in [*(f"g{x}" for x in GAINS), "gall"]:
        q, a = summ[g]["quake"], summ[g]["all"]
        print(f"  {g:5s} quake {fmt(q)} | all {fmt(a)} | targets {targets[g]}")
    print("  bootstrap:", json.dumps(out["bootstrap"]["ci95"]))
    print("  reference SAM 3 alone (gall): quake", fmt(out["reference_sam3_alone"]["gall"]["quake"]), "| all", fmt(out["reference_sam3_alone"]["gall"]["all"]))
    print("  quake r alone:", json.dumps({m: round(v["gall"], 3) for m, v in rq.items()}),
          "| r on accepted columns", out["r_accepted_quake"]["mean"], f"(n={out['r_accepted_quake']['n_images']})")


def cmd_real(names=("MAT", "KIP"), ra2="real_{}_paths.npz", started="real_started.json", results="real_results.json"):
    import real_paired_eval as rpe
    from obspy import UTCDateTime
    rule = json.loads((OUT / "frozen_rule.json").read_text())
    with open(OUT / started, "x") as f:
        f.write(json.dumps({"started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "rule": rule, "code_sha256": sha_file(__file__)}, indent=1))
    out = {"rule": rule, "frame": "research/25 layout, chosen_dp_fit alignment, global amplitude", "scans": {}}
    for name in names:
        cfg, rp = rpe.SCANS[name], ROOT / "out" / "real_paired"
        res = json.loads((rp / f"{name}_results.json").read_text())
        z = np.load(rp / name / f"{name}_paths.npz")
        bases, lay = z["bases"], res["layout"]
        x0, x1, xa, S = lay["x0"], lay["x1"], lay["xa"], rpe.TARGET_S
        P1 = np.load(ROOT / "out" / "ml" / "sam3_film2" / f"real_{name}_paths.npz")["SAM"].astype(float)
        P2 = np.load(ROOT / "out" / "ml" / "sim2real" / "ra2" / ra2.format(name))["RA2"].astype(float)
        assert P1.shape == P2.shape == z["B"].shape, (P1.shape, P2.shape, z["B"].shape)
        with np.errstate(invalid="ignore"):
            d = np.where(np.isfinite(P1[:, x0:x1]) & np.isfinite(P2[:, x0:x1]), np.abs(P1[:, x0:x1] - P2[:, x0:x1]) / S, np.inf)
        acc = np.stack([runs_ge(a, rule["min_run_cols"]) for a in d < rule["tau_S"]])
        t_on, t_off = UTCDateTime(cfg["on"]), UTCDateTime(cfg["off"])
        t_ref, ref, _ = rpe.reference(cfg, t_on)
        al, a_glob = res["alignment"]["chosen_dp_fit"], res["amplitude"]["global_scale_px_per_m"]
        cats = {l["line"]: l["category"] for l in res["lines"]}
        per, repro = {}, []
        for k in sorted(cats):
            idx, y, xcol = rpe.sample_path(P1, k, x0, x1, xa, al, t_ref, 0.0, float(t_off - t_on))
            if idx.size < 1200:                                    # same skip rule as ml_sam.real_metrics
                continue
            rf, tk = ref[idx], t_ref[idx]
            dev = -(y - bases[k]) * al["sign"]
            m_acc = acc[k, np.clip(np.round(xcol[idx]).astype(int), x0, x1 - 1) - x0]
            full = rpe.line_metrics(dev, rf, tk, S, a_glob)          # research/25 metric, all samples
            db, rb, dl, rl = rpe.bandpass(dev), rpe.bandpass(rf), rpe.lowpass(dev), rpe.lowpass(rf)
            tt = tk - tk.mean()
            Xg = np.c_[np.ones_like(tt), tt]
            eg = dl - a_glob * rl - Xg @ np.linalg.lstsq(Xg, dl - a_glob * rl, rcond=None)[0]   # = line_metrics' eg

            def part(m):
                if m.sum() < 300:
                    return None
                return {"n": int(m.sum()), "in_0.1S_global": float(np.mean(np.abs(eg[m]) < 0.1 * S)),
                        "lost_0.5S_global": float(np.mean(np.abs(eg[m]) > 0.5 * S)), "r_band": float(rpe.pearson(db[m], rb[m]))}
            allp = part(np.ones_like(m_acc))
            repro.append(max(abs(allp[key] - full[key]) for key in ("in_0.1S_global", "lost_0.5S_global", "r_band")))
            per[k] = {"category": cats[k], "accept_frac": float(m_acc.mean()), "n": int(idx.size),
                      "all": allp, "accepted": part(m_acc), "disputed": part(~m_acc)}
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
    (OUT / results).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"dump": cmd_dump, "select": cmd_select, "describe": cmd_describe, "e3": cmd_e3, "real": cmd_real,
     "real-alq": lambda: cmd_real(("ALQ",), "realalq_{}_paths.npz", "real_alq_started.json", "real_alq_results.json")}[sys.argv[1]]()
