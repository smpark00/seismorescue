"""Post hoc and descriptive: how would a flat line at the baseline score on the numbers we quote? (research/54)
A flat line has no skill, so a metric it scores well on says little. No model runs here and there is no start marker;
this only reads stored paths and ground truth.

SEISMO_FLAT_RULE=fb reads the research/55 both-directions masks instead (click, real, mat6): out/flat_fb/.

usage: python -I scripts/flat_baseline.py synth   # E2 (holdout4), E3 (holdout6): frozen accept masks, pen-moved columns
       python -I scripts/flat_baseline.py val     # film2 validation: fine-tuned, untuned and flat, research/37
       python -I scripts/flat_baseline.py click   # E8 click experiment: the same scored columns, research/47
       python -I scripts/flat_baseline.py real    # MAT, KIP, ALQ: frozen masks and metric of research/43 and research/49
       python -I scripts/flat_baseline.py mat6    # MAT line 6 r: lag, circular shift, phase scramble, other lines
       python -I scripts/flat_baseline.py all

Each command first reproduces the stored SAM 3 numbers and stops if they differ, then adds the flat line.
Needs the rendered holdouts (synth, click), film2_train (val) and the digital reference records (real, mat6).
Results: out/flat/flat_<command>.json, category totals only (no per-line numbers for ALQ, no images).

mat6 passes only if all of these hold (written before the first run, research/54 §1):
  1. the highest r within +-600 s of lag lies within +-5 s of lag 0;
  2. the accepted-span r at lag 0 beats the largest |r| of 1500 circular shifts of the reference;
  3. it beats the largest |r| of 2000 phase scrambles of the reference;
  4. it beats the largest |r| against the reference of any other line.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ml_consensus as mc                                         # noqa: E402
import ml_fb as fb                                                # noqa: E402

BOTH = os.environ.get("SEISMO_FLAT_RULE") == "fb"
OUT = ROOT / "out" / ("flat_fb" if BOTH else "flat")
TOL = 0.25                                                        # S/4, the synthetic precision metric
RULE = None


def rule():
    global RULE
    if RULE is None:
        RULE = json.loads(((fb.RULE if BOTH else mc.OUT / "frozen_rule.json")).read_text())
    return RULE


def pct(a, b):
    return round(100 * a / b, 2) if b else None


def save(cmd, res):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"flat_{cmd}.json").write_text(json.dumps(res, indent=1))


def same(name, got, want, tol=1e-9):
    if abs(got - want) > tol:
        raise SystemExit(f"{name}: recomputed {got} but stored {want}; stored paths and results no longer match")


# ---------------------------------------------------------------- synthetic: frozen accept masks
FIELDS = ("cols", "acc", "sam_acc", "flat_acc", "sam_all", "flat_all", "rej", "sam_rej", "flat_rej",
          "moved", "moved_sam", "moved_acc", "moved_acc_sam")


def counts(it, k, acc):
    ok = it["ok"][k]
    a, rej = acc[k] & ok, ok & ~acc[k]
    with np.errstate(invalid="ignore"):
        dev = np.abs(it["pen"][k] - it["base"][k]) / it["S"]
        good, flat, moved = it["e"][k] <= TOL, dev <= TOL, dev > TOL
    return dict(cols=(ok).sum(), acc=a.sum(), sam_acc=(a & good).sum(), flat_acc=(a & flat).sum(),
                sam_all=(ok & good).sum(), flat_all=(ok & flat).sum(), rej=rej.sum(), sam_rej=(rej & good).sum(),
                flat_rej=(rej & flat).sum(), moved=(ok & moved).sum(), moved_sam=(ok & moved & good).sum(),
                moved_acc=(a & moved).sum(), moved_acc_sam=(a & moved & good).sum())


def report(t):
    return {"accepted_share": pct(t["acc"], t["cols"]),
            "accepted_within_S4": {"sam3": pct(t["sam_acc"], t["acc"]), "flat": pct(t["flat_acc"], t["acc"])},
            "all_columns_within_S4": {"sam3": pct(t["sam_all"], t["cols"]), "flat": pct(t["flat_all"], t["cols"])},
            "disputed_within_S4": {"sam3": pct(t["sam_rej"], t["rej"]), "flat": pct(t["flat_rej"], t["rej"])},
            "pen_moved": {"share_of_columns": pct(t["moved"], t["cols"]), "sam3_alone_within_S4": pct(t["moved_sam"], t["moved"]),
                          "share_accepted": pct(t["moved_acc"], t["moved"]),
                          "accepted_within_S4": pct(t["moved_acc_sam"], t["moved_acc"])},
            "columns": {k: int(t[k]) for k in FIELDS}}


def boot_ci(per_group, fn, reps=10000, seed=0):
    """95% interval of fn(totals) over events, the same resampling unit as the frozen tables"""
    rng = np.random.default_rng(seed)
    groups = sorted(per_group)
    arr = np.array([[per_group[g][f] for f in FIELDS] for g in groups], float)
    vals = []
    for _ in range(reps):
        tot = arr[rng.integers(0, len(groups), len(groups))].sum(0)
        vals.append(fn(dict(zip(FIELDS, tot))))
    return [round(float(v), 2) for v in np.nanpercentile(vals, [2.5, 97.5])]   # 2 digits: research/55 §4.4 asks whether the lower bound is above 0


def cmd_synth():
    res = {"what": "research/54: flat line on the frozen consensus masks; pen moved = |pen - base| > S/4", "tol_S": TOL}
    stored = {"holdout6": ("e3_holdout6.json", "summary"), "holdout4": ("e2_holdout4.json", "selected_summary")}
    for split, (fname, key) in stored.items():
        items = mc.load_pairs(split)
        C, accs = mc.table(items, rule()["tau_S"], rule()["min_run_cols"])
        s, want = mc.summary(items, C)["gall"], json.loads((mc.OUT / fname).read_text())[key]["gall"]
        for cat in ("quake", "all"):
            for f in ("accept", "precision"):
                same(f"{split} {cat} {f}", s[cat][f], want[cat][f])
        tot = {c: dict.fromkeys(FIELDS, 0) for c in ("quake", "all")}
        gain = {g: {c: dict.fromkeys(FIELDS, 0) for c in ("quake", "all")} for g in mc.GAINS}
        group = {c: {} for c in ("quake", "all")}
        for it, acc in zip(items, accs):
            for k in range(it["n"]):
                n = counts(it, k, acc)
                for c in ("quake", "all") if k == it["q"] else ("all",):
                    pg = group[c].setdefault(it["group"], dict.fromkeys(FIELDS, 0))
                    for f in FIELDS:
                        tot[c][f] += int(n[f]); gain[it["gain"]][c][f] += int(n[f]); pg[f] += int(n[f])
        res[split] = {c: report(tot[c]) for c in tot}
        res[split]["by_gain"] = {f"g{g}": {c: report(gain[g][c]) for c in tot} for g in mc.GAINS}
        res[split]["ci95_over_events"] = {c: {
            "accepted_sam3": boot_ci(group[c], lambda t: 100 * t["sam_acc"] / t["acc"]),
            "accepted_flat": boot_ci(group[c], lambda t: 100 * t["flat_acc"] / t["acc"]),
            "pen_moved_sam3_alone": boot_ci(group[c], lambda t: 100 * t["moved_sam"] / t["moved"]),
            "pen_moved_accepted": boot_ci(group[c], lambda t: 100 * t["moved_acc_sam"] / t["moved_acc"]),
            "pen_moved_share_accepted": boot_ci(group[c], lambda t: 100 * t["moved_acc"] / t["moved"])} for c in tot}
        print(split, json.dumps({c: {k: v for k, v in res[split][c].items() if k != "columns"} for c in tot}))
        print(split, "ci", json.dumps(res[split]["ci95_over_events"]))
    save("synth", res)


# ---------------------------------------------------------------- film2 validation (research/37)
def cmd_val():
    from track_joint import score
    tuned = json.loads((ROOT / "out/ml/sam3_film2_modal/eval_val_sam3_film2.json").read_text())
    zero = json.loads((ROOT / "out/ml/sam3_film2_modal/eval_zeroshot-val.json").read_text())
    zrows = {(r["record"], r["gain"]): r for r in zero["rows"]}
    root, rows = ROOT / "out" / "film2_train", []
    for r in tuned["rows"]:
        meta = json.loads((root / r["record"] / "film" / "meta.json").read_text())
        pen = np.load(root / r["record"] / "film" / f"gt_g{r['gain']:g}.npz")["pen"].astype(float)
        S, x0, n, act = float(meta["spacing"]), meta["margin_x"], pen.shape[0], meta["active_line"]
        x1 = pen.shape[1] - x0
        flat = np.tile((meta["margin_y"] + S * np.arange(n))[:, None], (1, pen.shape[1]))
        z = zrows[(r["record"], r["gain"])]
        rows.append({"record": r["record"], "gain": r["gain"], "sam3_all": r["SAM"], "sam3_quake": r["SAM_quake"],
                     "untuned_all": z["SAM"], "untuned_quake": z["SAM_quake"], "b_all": r["B"], "b_quake": r["B_quake"],
                     "flat_all": score(flat, pen, range(n), x0, x1, S)[0], "flat_quake": score(flat, pen, [act], x0, x1, S)[0]})
    keys = ("sam3_all", "sam3_quake", "untuned_all", "untuned_quake", "b_all", "b_quake", "flat_all", "flat_quake")
    mean = {k: round(float(np.mean([r[k] for r in rows])), 1) for k in keys}
    same("val quake", mean["sam3_quake"], 20.5, 0.051)
    same("val all", mean["sam3_all"], 5.9, 0.051)
    same("zero-shot quake", mean["untuned_quake"], 62.7, 0.051)
    res = {"what": "research/54: film2 validation, % of drawn columns missing or |path - pen| > S/2 (track_joint.score)",
           "n_images": len(rows), "lam": tuned["decode"]["lam"], "mean_fail_pct": mean,
           "by_gain": {f"g{g:g}": {k: round(float(np.mean([r[k] for r in rows if r["gain"] == g])), 1) for k in keys}
                       for g in sorted({r["gain"] for r in rows})}}
    print(json.dumps(res))
    save("val", res)


# ---------------------------------------------------------------- E8 click experiment (research/47)
def cmd_click():
    click = ROOT / "out/ml" / ("click_fb" if BOTH else "click")
    stored = json.loads((click / "e8_results.json").read_text())["oracle"]
    rows = [json.loads(line) for line in (click / "rows_oracle.jsonl").read_text().splitlines()]
    items = {(it["record"], it["gain"]): it for it in (fb.pairs("holdout4") if BOTH else mc.load_pairs("holdout4"))}
    gt_root = mc.SETS["holdout4"][0]
    acc = {key: mc.accept(it, rule()["tau_S"], rule()["min_run_cols"]) for key, it in items.items()}
    tot = {c: dict(n=0, new=0, old=0, ra=0, flat=0) for c in ("all", "quake")}
    for r in rows:
        if r["unreached"]:
            continue
        it, key = items[(r["record"], r["gain"])], (r["record"], r["gain"])
        x0, k = json.loads((gt_root / r["record"] / "film" / "meta.json").read_text())["margin_x"], r["line"]
        c, w1 = r["c"] - x0, r["w1"] - x0
        cols = np.flatnonzero(it["ok"][k][c:w1] & ~acc[key][k][c:w1]) + c
        if len(cols) != r["n"]:
            raise SystemExit(f"scored columns differ for {key} line {k} at {c}: {len(cols)} vs {r['n']}")
        pen, S = it["pen"][k][cols], it["S"]
        old = int((np.abs(it["P1"][k][cols] - pen) / S <= TOL).sum())
        ra = int((np.abs(it["P2"][k][cols] - pen) / S <= TOL).sum())
        if old != r["old4"] or ra != (r["back4"] if BOTH else r["ra4"]):
            raise SystemExit(f"stored counts differ for {key} line {k} at {c}")
        flat = int((np.abs(pen - it["base"][k]) / S <= TOL).sum())
        for cat in ("all", "quake") if r["quake"] else ("all",):
            t = tot[cat]
            t["n"] += r["n"]; t["new"] += r["new4"]; t["old"] += old; t["ra"] += ra; t["flat"] += flat
    res = {"what": "research/54: E8 scored columns (holdout4, one true click per review window), share within S/4", "tol_S": TOL}
    for cat, t in tot.items():
        same(f"E8 {cat} re-track", round(t["new"] / t["n"], 4), stored[cat]["precision_S4_retrack"], 1e-4)
        same(f"E8 {cat} before", round(t["old"] / t["n"], 4), stored[cat]["precision_S4_sam3_before"], 1e-4)
        res[cat] = {"cols": t["n"], "retrack": pct(t["new"], t["n"]), "sam3_before": pct(t["old"], t["n"]),
                    ("backward" if BOTH else "ra2"): pct(t["ra"], t["n"]), "flat": pct(t["flat"], t["n"])}
    print(json.dumps(res))
    save("click", res)


# ---------------------------------------------------------------- real film (research/43, research/49)
def real_frame(name, ra2):
    import real_paired_eval as rpe
    from obspy import UTCDateTime
    cfg, rp = rpe.SCANS[name], ROOT / "out" / "real_paired"
    res = json.loads((rp / f"{name}_results.json").read_text())
    z = np.load(rp / name / f"{name}_paths.npz")
    bases, lay = z["bases"], res["layout"]
    x0, x1, xa, S = lay["x0"], lay["x1"], lay["xa"], rpe.TARGET_S
    P1 = np.load(ROOT / "out/ml/sam3_film2" / f"real_{name}_paths.npz")["SAM"].astype(float)
    P2 = np.load(fb.OUT / f"real_{name}_paths.npz")["B"].astype(float) if BOTH else np.load(ROOT / "out/ml/sim2real/ra2" / ra2.format(name))["RA2"].astype(float)
    with np.errstate(invalid="ignore"):
        d = np.where(np.isfinite(P1[:, x0:x1]) & np.isfinite(P2[:, x0:x1]), np.abs(P1[:, x0:x1] - P2[:, x0:x1]) / S, np.inf)
    acc = np.stack([mc.runs_ge(a, rule()["min_run_cols"]) for a in d < rule()["tau_S"]])
    t_on, t_off = UTCDateTime(cfg["on"]), UTCDateTime(cfg["off"])
    t_ref, ref, _ = rpe.reference(cfg, t_on)
    al, a_glob = res["alignment"]["chosen_dp_fit"], res["amplitude"]["global_scale_px_per_m"]
    cats = {ln["line"]: ln["category"] for ln in res["lines"]}
    return dict(rpe=rpe, bases=bases, x0=x0, x1=x1, xa=xa, S=S, P1=P1, P2=P2, acc=acc, t_ref=t_ref, ref=ref, al=al, a_glob=a_glob,
                cats=cats, span=float(t_off - t_on))


def line_samples(f, k, P=None):
    rpe = f["rpe"]
    idx, y, xcol = rpe.sample_path(f["P1"] if P is None else P, k, f["x0"], f["x1"], f["xa"], f["al"], f["t_ref"], 0.0, f["span"])
    if idx.size < 1200:                                           # same skip rule as ml_consensus.cmd_real
        return None
    m = f["acc"][k, np.clip(np.round(xcol[idx]).astype(int), f["x0"], f["x1"] - 1) - f["x0"]]
    return idx, y, m


def errors(f, dev, rf, tk):
    rpe, S = f["rpe"], f["S"]
    dl, rl = rpe.lowpass(dev), rpe.lowpass(rf)
    tt = tk - tk.mean()
    Xg = np.c_[np.ones_like(tt), tt]
    return (dl - f["a_glob"] * rl - Xg @ np.linalg.lstsq(Xg, dl - f["a_glob"] * rl, rcond=None)[0]) / S, rl


def cmd_real():
    res = {"what": "research/54: flat line (dev = 0) through the same filter and per-line offset and trend fit as research/43; "
                   "a flat line has no r. true_motion_S = |global amplitude * low-passed digital record| / S"}
    for name, ra2, fname in (("MAT", "real_{}_paths.npz", "real_results.json"), ("KIP", "real_{}_paths.npz", "real_results.json"),
                             ("ALQ", "realalq_{}_paths.npz", "real_alq_results.json")):
        f = real_frame(name, ra2)
        rpe, stored = f["rpe"], json.loads(((fb.OUT / "real_results.json") if BOTH else (mc.OUT / fname)).read_text())["scans"][name]
        per = {}
        for k in sorted(f["cats"]):
            got = line_samples(f, k)
            if got is None:
                continue
            idx, y, m = got
            rf, tk = f["ref"][idx], f["t_ref"][idx]
            dev = -(y - f["bases"][k]) * f["al"]["sign"]
            e_sam, rl = errors(f, dev, rf, tk)
            e_flat, _ = errors(f, np.zeros_like(dev), rf, tk)
            db, rb = rpe.bandpass(dev), rpe.bandpass(rf)
            motion = np.abs(f["a_glob"] * rl) / f["S"]

            def part(mm):
                if mm.sum() < 300:
                    return None
                return {"n": int(mm.sum()), "sam3_in_0.1S": float(np.mean(np.abs(e_sam[mm]) < 0.1)), "sam3_off_0.5S": float(np.mean(np.abs(e_sam[mm]) > 0.5)),
                        "flat_in_0.1S": float(np.mean(np.abs(e_flat[mm]) < 0.1)), "flat_off_0.5S": float(np.mean(np.abs(e_flat[mm]) > 0.5)),
                        "sam3_r": float(rpe.pearson(db[mm], rb[mm])), "true_motion_S_median": float(np.median(motion[mm]))}
            per[k] = {"cat": f["cats"][k], "acc": float(m.mean()), "accepted": part(m), "disputed": part(~m)}
            for prt, key in (("accepted", "accepted"), ("disputed", "disputed")):
                s = stored["per_line"][str(k)][key]
                if s:
                    same(f"{name} line {k} {prt} off", per[k][prt]["sam3_off_0.5S"], s["lost_0.5S_global"], 1e-9)
                    same(f"{name} line {k} {prt} in", per[k][prt]["sam3_in_0.1S"], s["in_0.1S_global"], 1e-9)
                    same(f"{name} line {k} {prt} r", per[k][prt]["sam3_r"], s["r_band"], 1e-9)
        out = {}
        for cat in ("quake", "neighbour", "other_active", "quiet"):
            ks = [k for k, v in per.items() if v["cat"] == cat]
            if not ks:
                continue
            out[cat] = {"n_lines": len(ks), "accept_frac": round(float(np.mean([per[k]["acc"] for k in ks])), 3)}
            for prt in ("accepted", "disputed"):
                vals = [per[k][prt] for k in ks if per[k][prt]]
                out[cat][prt] = {key: round(float(np.mean([v[key] for v in vals])), 3) for key in vals[0]} | {"n_lines": len(vals)} if vals else None
                if vals:
                    out[cat][prt].pop("n")
        res[name] = out
        print(name, json.dumps(out))
    save("real", res)


# ---------------------------------------------------------------- MAT line 6: try to break r +0.83
def cmd_mat6():
    f = real_frame("MAT", "real_{}_paths.npz")
    rpe, k = f["rpe"], 6
    idx, y, m = line_samples(f, k)
    ref, N = f["ref"], len(f["ref"])
    dev = -(y - f["bases"][k]) * f["al"]["sign"]
    db = rpe.bandpass(dev)
    rb = rpe.bandpass(ref[idx])
    r0, r_all = rpe.pearson(db[m], rb[m]), rpe.pearson(db, rb)
    results = (fb.OUT if BOTH else mc.OUT) / "real_results.json"
    stored = json.loads(results.read_text())["scans"]["MAT"]["per_line"]["6"]
    same("line 6 accepted r", r0, stored["accepted"]["r_band"])
    same("line 6 whole r", r_all, stored["all"]["r_band"])
    rng = np.random.default_rng(20261101)

    def r_of(rf):
        b = rpe.bandpass(rf)
        return rpe.pearson(db[m], b[m]), rpe.pearson(db, b)
    lags = [L for L in range(-600, 601) if idx[0] + L >= 0 and idx[-1] + L < N]
    by_lag = np.array([r_of(ref[idx + L]) for L in lags])
    best = int(np.argmax(by_lag[:, 0]))
    beyond = [i for i, L in enumerate(lags) if abs(L) > 30]
    shifts = rng.integers(600, N - 600, 1500)
    circ = np.array([r_of(np.roll(ref, int(s))[idx]) for s in shifts])

    spec = np.fft.rfft(rb)
    scr = []
    for _ in range(2000):
        ph = np.exp(2j * np.pi * rng.random(spec.size))
        ph[0] = 1
        b = np.fft.irfft(np.abs(spec) * ph, rb.size)
        scr.append((rpe.pearson(db[m], b[m]), rpe.pearson(db, b)))
    scr = np.array(scr)

    other = {}
    for j in sorted(f["cats"]):
        got = line_samples(f, j)
        if j == k or got is None:
            continue
        n = min(got[0].size, idx.size)
        b = rpe.bandpass(ref[got[0][:n]])
        mm = m[:n]
        other[j] = (rpe.pearson(db[:n][mm], b[mm]), rpe.pearson(db[:n], b))
    other_max = max(abs(v[0]) for v in other.values())

    win = []
    for a in range(0, idx.size - 600 + 1, 600):
        w = np.arange(a, a + 600)
        w = w[m[w]]
        if w.size >= 300:
            win.append({"start_s": int(f["t_ref"][idx[a]]), "n": int(w.size), "r": round(rpe.pearson(db[w], rb[w]), 3)})
    a_ls = float(np.dot(db[m], rb[m]) / np.dot(rb[m], rb[m]) / f["a_glob"])
    per_line = json.loads(results.read_text())["scans"]["MAT"]["per_line"]
    ranked = sorted(((v["accepted"]["r_band"], int(j)) for j, v in per_line.items() if v["accepted"]), reverse=True)

    res = {"what": "research/54: MAT line 6, the r quoted for the M6.4 event; line chosen after the fact on a development chip",
           "n_samples": int(idx.size), "n_accepted": int(m.sum()), "accept_frac": float(m.mean()),
           "r_accepted": round(r0, 3), "r_whole_line": round(r_all, 3),
           "lag": {"range": [lags[0], lags[-1]], "best_lag_s": lags[best], "best_r": round(float(by_lag[best, 0]), 3),
                   "max_r_beyond_30s": round(float(by_lag[beyond, 0].max()), 3), "max_abs_r_beyond_30s": round(float(np.abs(by_lag[beyond, 0]).max()), 3)},
           "circular_shift": {"n": len(shifts), "max_abs_r_accepted": round(float(np.abs(circ[:, 0]).max()), 3),
                              "max_abs_r_whole": round(float(np.abs(circ[:, 1]).max()), 3)},
           "phase_scramble": {"n": len(scr), "max_abs_r_accepted": round(float(np.abs(scr[:, 0]).max()), 3),
                              "max_abs_r_whole": round(float(np.abs(scr[:, 1]).max()), 3)},
           "other_lines_reference": {"n": len(other), "max_abs_r_accepted": round(other_max, 3)},
           "windows_600s": win, "amplitude_slope_vs_global": round(a_ls, 3),
           "accepted_r_all_lines_ranked": [[j, round(v, 3)] for v, j in ranked[:8]], "n_lines_with_accepted_part": len(ranked)}
    res["pass"] = {"lag_peak_at_0": bool(abs(lags[best]) <= 5), "beats_circular": bool(r0 > res["circular_shift"]["max_abs_r_accepted"]),
                   "beats_scramble": bool(r0 > res["phase_scramble"]["max_abs_r_accepted"]), "beats_other_lines": bool(r0 > other_max)}
    res["pass"]["all"] = all(res["pass"].values())
    print(json.dumps(res, indent=1))
    save("mat6", res)


if __name__ == "__main__":
    cmds = {"synth": cmd_synth, "val": cmd_val, "click": cmd_click, "real": cmd_real, "mat6": cmd_mat6}
    if len(sys.argv) != 2 or sys.argv[1] not in (*cmds, "all"):
        raise SystemExit(__doc__)
    for name, fn in cmds.items():
        if sys.argv[1] in (name, "all"):
            fn()
