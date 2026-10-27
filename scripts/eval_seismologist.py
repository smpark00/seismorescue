"""Score the seismologist agent against tool-only baselines with grouped CV and cluster bootstrap.

usage: python -I eval_seismologist.py <series_dir> <catalog_dir> <agent_dir> <run_tag> [<run_tag> ...]

Revised after the 10-27 Fable review (docs/RESULTS.md):
- target = the task's own definition: the largest catalogue event whose P arrives inside the record (for the two
  2011-03-11 M7.7 records that is the M7.9 aftershock, not the fetch target);
- P is scored on all-lines records only (single-active records copy pre-P noise onto most lines), and the baseline's
  AIC window is chosen on the TRAINING groups' scored items only (GroupKFold by event date, 4 folds x 3 shuffles);
- extra no-LLM baselines: catalogue P as the answer (no measurement), and a surface peak searched in a window
  narrower than one helicorder line, [R1 - 200, R1 + X] with X from a fixed grid chosen by the same grouped CV;
- every comparison is also reported without the 2011-03-11 group (references unstable under the M9.1 coda).
Ground truth (true signal, failure mask) is read from <series_dir>_gt; the agent only ever saw <series_dir>.
References: P = AIC on the true series around the target's predicted P (60 s); surface = max |true| in
[R1 - 200, R1 + 1800] (the true series has no line-jump duplicates). QC truth = digitized path off by > S/2.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from seismo_tools import aic_pick

series_dir, cat_dir, agent_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
gt_dir = series_dir.parent / (series_dir.name + "_gt")
RUNS = sys.argv[4:]
P_OK, SURF_OK = 10, 60
WINDOWS = [20, 40, 60]
NARROW = [500, 650, 800]

items = []
for f in sorted(series_dir.glob("*.npz")):
    rec = f.stem.rsplit("_", 2)[0]
    z, zg = np.load(f), np.load(gt_dir / f.name)
    cat = json.loads((cat_dir / f"{rec}.json").read_text())
    inside = [e for e in cat["events"] if e["P_s"] is not None and 0 <= e["P_s"] < 7200]
    tgt = max(inside, key=lambda e: e["mag"])            # task definition (== baseline's event choice)
    r1 = tgt["R1_s"]
    w0, w1 = int(max(0, r1 - 200)), int(min(7200, r1 + 1800))
    it = {"item": f.stem, "group": rec[:8], "cond": "_".join(f.stem.rsplit("_", 2)[1:]),
          "p_ref": aic_pick(zg["true"], tgt["P_s"], 60)["pick_s"], "p_theory": tgt["P_s"],
          "surf_ref": w0 + int(np.nanargmax(np.abs(zg["true"][w0:w1]))),
          "fail": zg["fail"], "lowconf": z["lowconf"], "target_id": tgt["id"],
          "fetch_target_differs": not tgt["is_target"],
          "base_p": {W: aic_pick(z["digi"], tgt["P_s"], W)["pick_s"] for W in WINDOWS},
          "surf_naive": int(np.nanargmax(np.abs(z["digi"]))),
          "surf_win2000": w0 + int(np.nanargmax(np.abs(z["digi"][w0:w1]))),
          "surf_narrow": {X: int(max(0, r1 - 200)) + int(np.nanargmax(np.abs(z["digi"][int(max(0, r1 - 200)):int(min(7200, r1 + X))])))
                          for X in NARROW}, "agent": {}}
    for run in RUNS:
        p = agent_dir / run / f"{f.stem}.json"
        if p.exists():
            it["agent"][run] = (json.loads(p.read_text()) or {}).get("answer")
    items.append(it)

scored_p = lambda it: it["cond"].startswith("all")
groups = sorted({it["group"] for it in items})
cv_p, cv_s = {}, {}
for rep in range(3):
    gs = groups[:]
    random.Random(rep).shuffle(gs)
    for fold in [gs[i::4] for i in range(4)]:
        train = [it for it in items if it["group"] not in fold]
        W = max(WINDOWS, key=lambda w: np.mean([abs(it["base_p"][w] - it["p_ref"]) <= P_OK for it in train if scored_p(it)]))
        X = max(NARROW, key=lambda x: np.mean([abs(it["surf_narrow"][x] - it["surf_ref"]) <= SURF_OK for it in train]))
        for it in items:
            if it["group"] in fold:
                cv_p.setdefault(it["item"], []).append(it["base_p"][W])
                cv_s.setdefault(it["item"], []).append(it["surf_narrow"][X])


def mask_from_spans(spans, n=7200):
    m = np.zeros(n, bool)
    for s in spans or []:
        try:
            a, b = int(max(0, s[0])), int(min(n, s[1]))
            m[a:b] = True
        except Exception:
            pass
    return m


def pr(flag, truth):
    tp = (flag & truth).sum()
    return (tp / flag.sum() if flag.sum() else np.nan), (tp / truth.sum() if truth.sum() else np.nan)


def boot(vals, n=4000, seed=0):
    """vals: list of (group, value); percentile cluster bootstrap of the mean (11 clusters: CIs are optimistic)"""
    rng = np.random.default_rng(seed)
    gs = sorted({g for g, _ in vals})
    by = {g: np.array([v for gg, v in vals if gg == g], float) for g in gs}
    ms = [np.concatenate([by[gs[i]] for i in rng.choice(len(gs), len(gs))]).mean() for _ in range(n)]
    return float(np.mean([v for _, v in vals])), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


rows = {}
for it in items:
    r = {"group": it["group"], "cond": it["cond"], "fetch_target_differs": it["fetch_target_differs"]}
    r["base_id"] = 1.0                                    # the baseline picks the target by definition
    r["p_aic_cv"] = float(np.mean([abs(p - it["p_ref"]) <= P_OK for p in cv_p[it["item"]]]))
    r["p_theory"] = float(abs(it["p_theory"] - it["p_ref"]) <= P_OK)
    r["s_naive"] = float(abs(it["surf_naive"] - it["surf_ref"]) <= SURF_OK)
    r["s_win2000"] = float(abs(it["surf_win2000"] - it["surf_ref"]) <= SURF_OK)
    r["s_narrow_cv"] = float(np.mean([abs(s - it["surf_ref"]) <= SURF_OK for s in cv_s[it["item"]]]))
    r["base_qc"] = pr(it["lowconf"], it["fail"])
    for run, a in it["agent"].items():
        if not isinstance(a, dict):
            r[run] = None
            continue

        def num(k):
            try:
                return float(a.get(k))
            except Exception:
                return np.nan
        r[run] = {"id": float(a.get("event_id") == it["target_id"]),
                  "p_ok": float(abs(num("p_arrival_s") - it["p_ref"]) <= P_OK),
                  "surf_ok": float(abs(num("surface_peak_s") - it["surf_ref"]) <= SURF_OK),
                  "qc": pr(mask_from_spans(a.get("suspect_spans")), it["fail"])}
    rows[it["item"]] = r


def table(keys, run, title):
    print(f"\n--- {title} (n={len(keys)} items)")
    kp = [k for k in keys if rows[k]["cond"].startswith("all")]
    print(f"  event id: agent {np.mean([rows[k][run]['id'] for k in keys]):.2f} (baseline 1.00 by definition)")
    for label, kk, akey, bases in [("P within 10 s (all-lines)", kp, "p_ok", ["p_aic_cv", "p_theory"]),
                                   ("surface within 60 s", keys, "surf_ok", ["s_naive", "s_win2000", "s_narrow_cv"])]:
        a = boot([(rows[k]["group"], rows[k][run][akey]) for k in kk])
        print(f"  {label} [n={len(kk)}]: agent {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}]")
        for b in bases:
            bb = boot([(rows[k]["group"], rows[k][b]) for k in kk])
            d = boot([(rows[k]["group"], rows[k][run][akey] - rows[k][b]) for k in kk])
            print(f"      vs {b:12s} {bb[0]:.2f}   agent - baseline {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]")


print(f"items {len(rows)}; target differs from fetch target in {sum(r['fetch_target_differs'] for r in rows.values())} items; "
      + "agent answers: " + ", ".join(f"{run} {sum(isinstance(r.get(run), dict) for r in rows.values())}" for run in RUNS))
for run in RUNS:
    ok = [k for k, r in rows.items() if isinstance(r.get(run), dict)]
    table(ok, run, f"run {run}, all records")
    table([k for k in ok if rows[k]["group"] != "20110311"], run, f"run {run}, without the 2011-03-11 group")
    bp = np.array([rows[k]["base_qc"] for k in ok], float)
    ap = np.array([rows[k][run]["qc"] for k in ok], float)
    print(f"  QC precision/recall: digitizer low-conf {np.nanmean(bp[:, 0]):.2f}/{np.nanmean(bp[:, 1]):.2f}  "
          f"agent flags {np.nanmean(ap[:, 0]):.2f}/{np.nanmean(ap[:, 1]):.2f}")
    for cond in ["all_g2", "all_g5", "active_g1.5", "active_g2"]:
        kk = [k for k in ok if rows[k]["cond"] == cond]
        ptxt = (f"P: agent {np.mean([rows[k][run]['p_ok'] for k in kk]):.2f} aic {np.mean([rows[k]['p_aic_cv'] for k in kk]):.2f} "
                f"theory {np.mean([rows[k]['p_theory'] for k in kk]):.2f} | " if cond.startswith("all") else "")
        print(f"    {cond:11s} {ptxt}surf: agent {np.mean([rows[k][run]['surf_ok'] for k in kk]):.2f} "
              f"narrow {np.mean([rows[k]['s_narrow_cv'] for k in kk]):.2f} win2000 {np.mean([rows[k]['s_win2000'] for k in kk]):.2f}")
if len(RUNS) >= 2:
    both = [k for k, r in rows.items() if all(isinstance(r.get(x), dict) for x in RUNS[:2])]
    bp_ = [k for k in both if rows[k]["cond"].startswith("all")]
    print(f"\nrun-to-run agreement: P success {np.mean([rows[k][RUNS[0]]['p_ok'] == rows[k][RUNS[1]]['p_ok'] for k in bp_]):.2f} (n={len(bp_)}), "
          f"surface success {np.mean([rows[k][RUNS[0]]['surf_ok'] == rows[k][RUNS[1]]['surf_ok'] for k in both]):.2f} (n={len(both)})")
(agent_dir / "eval_summary.json").write_text(json.dumps(rows, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
