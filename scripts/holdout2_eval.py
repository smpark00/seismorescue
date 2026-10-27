"""Second held-out test of the frozen system: judge v5 + active-line detector + proposer C (docs/RESULTS.md, pre-registered).

usage: python -I holdout2_eval.py <dev_bench1> <dev_cache1> <dev_bench2> <dev_cache2> <test_bench> <test_cache> <out.json>

The judge is trained once on all non-tie events of both development sets; the detector and proposer C are fixed
(active_guard.py, Q=5). Pre-registered tests H4-H8 (see docs/RESULTS.md) use a cluster bootstrap over test event dates.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from active_guard import detect_active, track_active_clean
from arbiter import assemble, cache_proposals
from cv_judges import CONDS, fit_lr, load_items
from track_joint import score

Q = 5
db1, dc1, db2, dc2, tb, tc, out = map(Path, sys.argv[1:8])
cache_proposals(tb, tc, CONDS)
dev = load_items(db1, dc1) + load_items(db2, dc2)
rows = [r for it in dev for r in it["rows"] if r["label"] >= 0]
model = fit_lr(np.array([r["x"] for r in rows]), np.array([r["label"] for r in rows], float), np.array([r["stake"] for r in rows], float))
test = load_items(tb, tc)
print(f"judge trained once on {len(rows)} non-tie events ({len(dev)} dev items); testing {len(test)} items")

res = []
for it in test:
    meta = json.loads((tb / it["record"] / it["mode"] / "meta.json").read_text())
    ink = np.load(tc / f"{it['record']}_{it['mode']}_g{it['gain']:g}.npz")["ink"]
    p = model(np.array([r["x"] for r in it["rows"]])) if it["rows"] else np.array([])
    judge = assemble(it["A"], it["B"], it["rows"], ["A" if v > 0.5 else "B" for v in p])
    k, nq = detect_active(it["B"], ink, meta["margin_y"], it["S"], it["x0"], it["x1"])
    system = judge.copy()
    fired = k is not None and nq >= Q
    if fired:
        c = track_active_clean(it["B"], ink, it["pen"][k, it["x0"]], meta["margin_y"], it["S"], it["x0"], it["x1"])
        system[k, it["x0"]:it["x1"]] = c[it["x0"]:it["x1"]]
    row = {"record": it["record"], "group": it["group"], "mode": it["mode"], "gain": it["gain"], "fired": fired,
           "detected": k, "active": it["active"], "n_quiet": nq}
    N = it["pen"].shape[0]
    for name, P in (("A", it["A"]), ("B", it["B"]), ("judge", judge), ("system", system)):
        row[name] = score(P, it["pen"], range(N), it["x0"], it["x1"], it["S"])[0]
        row[name + "_active"] = score(P, it["pen"], [it["active"]], it["x0"], it["x1"], it["S"])[0] if it["mode"] == "active" else None
    res.append(row)


def boot(vals, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    gs = sorted({g for g, _ in vals})
    by = {g: np.array([v for gg, v in vals if gg == g], float) for g in gs}
    ms = [np.concatenate([by[gs[i]] for i in rng.choice(len(gs), len(gs))]).mean() for _ in range(n)]
    return float(np.mean([v for _, v in vals])), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def test_(tag, mode, g, lhs, rhs, bound=0.0):
    rr = [r for r in res if r["mode"] == mode and r["gain"] == g]
    vals = [(r["group"], r[lhs] - r[rhs]) for r in rr]
    m, lo, hi = boot(vals)
    ok = hi < bound
    print(f"  {tag} {mode} g{g}: {lhs} - {rhs} = {m:+.1f} [{lo:+.1f}, {hi:+.1f}] (system better in {sum(v < 0 for _, v in vals)}/{len(vals)}) "
          f"-> {'SUPPORTED' if ok else 'NOT supported'} (need upper < {bound:+.1f})")


print("\n== pre-registered (docs/RESULTS.md)")
test_("H4", "active", 2.0, "system_active", "A_active")
test_("H5", "active", 1.5, "system_active", "A_active")
test_("H6", "active", 2.0, "system_active", "judge_active")
test_("H7", "all", 2.0, "system", "judge", bound=1.0)
test_("H8", "all", 2.0, "system", "A")
print("\n== exploratory (mean fail %)")
for mode, g in CONDS:
    rr = [r for r in res if r["mode"] == mode and r["gain"] == g]
    line = f"  {mode:6s} g{g:<3} whole: " + "  ".join(f"{k} {np.mean([r[k] for r in rr]):5.1f}" for k in ("A", "B", "judge", "system"))
    line += f"  | regime fired {np.mean([r['fired'] for r in rr]):.0%}"
    if mode == "active":
        line += f"\n  {'':11s} active line: " + "  ".join(f"{k} {np.mean([r[k + '_active'] for r in rr]):5.1f}" for k in ("A", "B", "judge", "system"))
        line += f"  | detector right {np.mean([r['detected'] == r['active'] for r in rr]):.0%}"
    print(line)
out.write_text(json.dumps(res, indent=1))
