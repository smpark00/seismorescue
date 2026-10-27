"""Held-out validation of the frozen digitizer arbitration on new events (docs/RESULTS.md, pre-registered).

usage: python -I holdout_eval.py <train_bench> <train_cache> <holdout_bench> <holdout_cache> <out.json>

The judge (cv_judges.ctx_features + stake-weighted logistic regression, TAU 6, threshold 0.5) is trained ONCE on all
non-tie events of the training records and applied unchanged to the held-out records, whose event dates never occur
in training. Proposals for held-out records are computed here (same proposers and the same GT start position as in
training). Pre-registered tests: H1 all-lines g2 learned < A; H2 all-lines g2 learned < per-record min(A, B);
H3 active g2 (whole image) learned < min(A, B). Everything else is exploratory.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from arbiter import cache_proposals, oracle_choice
from cv_judges import CONDS, evaluate, fit_lr, load_items

train_bench, train_cache, ho_bench, ho_cache, out = map(Path, sys.argv[1:6])
cache_proposals(ho_bench, ho_cache, CONDS)              # proposals for the held-out records (cached)

train, test = load_items(train_bench, train_cache), load_items(ho_bench, ho_cache)
rows = [r for it in train for r in it["rows"] if r["label"] >= 0]
model = fit_lr(np.array([r["x"] for r in rows]), np.array([r["label"] for r in rows], float),
               np.array([r["stake"] for r in rows], float))
print(f"trained once on {len(rows)} non-tie events from {len(train)} training items; testing on {len(test)} held-out items")

res = []
for it in test:
    p = model(np.array([r["x"] for r in it["rows"]])) if it["rows"] else np.array([])
    choices = {"learned": ["A" if q > 0.5 else "B" for q in p],
               "rule": ["A" if r["rule_A"] else "B" for r in it["rows"]],
               "oracle": oracle_choice(it["A"], it["B"], it["rows"], it["pen"], it["S"]),
               "A": ["A"] * len(it["rows"]), "B": ["B"] * len(it["rows"])}
    row = {"record": it["record"], "group": it["group"], "mode": it["mode"], "gain": it["gain"], "events": len(it["rows"])}
    for name, ch in choices.items():
        f, a = evaluate(it, ch)
        row[name] = f
        row[name + "_active"] = a
    # A alone / B alone are the proposers' own paths (assembling all-A over events equals A only inside events)
    from track_joint import score
    for name, P in (("A_pure", it["A"]), ("B_pure", it["B"])):
        row[name] = score(P, it["pen"], range(it["pen"].shape[0]), it["x0"], it["x1"], it["S"])[0]
        row[name + "_active"] = score(P, it["pen"], [it["active"]], it["x0"], it["x1"], it["S"])[0] if it["mode"] == "active" else None
    res.append(row)


def boot(vals, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    gs = sorted({g for g, _ in vals})
    by = {g: np.array([v for gg, v in vals if gg == g], float) for g in gs}
    ms = [np.concatenate([by[gs[i]] for i in rng.choice(len(gs), len(gs))]).mean() for _ in range(n)]
    return float(np.mean([v for _, v in vals])), float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))


def paired(mode, g, lhs, rhs_fn, suffix=""):
    rr = [r for r in res if r["mode"] == mode and r["gain"] == g]
    vals = [(r["group"], r[lhs + suffix] - rhs_fn(r)) for r in rr]
    m, lo, hi = boot(vals)
    wins = sum(v < 0 for _, v in vals)
    return m, lo, hi, wins, len(vals)


print("\n== pre-registered")
for tag, mode, g, rhs_name, rhs in [("H1", "all", 2.0, "A alone", lambda r: r["A_pure"]),
                                     ("H2", "all", 2.0, "min(A,B) per record", lambda r: min(r["A_pure"], r["B_pure"])),
                                     ("H3", "active", 2.0, "min(A,B) per record", lambda r: min(r["A_pure"], r["B_pure"]))]:
    m, lo, hi, w, n = paired(mode, g, "learned", rhs)
    verdict = "SUPPORTED" if hi < 0 else "NOT supported"
    print(f"  {tag} {mode} g{g} whole: learned - {rhs_name} = {m:+.1f} [{lo:+.1f}, {hi:+.1f}], learned better in {w}/{n} -> {verdict}")

print("\n== exploratory (mean fail %, held-out)")
for mode, g in CONDS:
    rr = [r for r in res if r["mode"] == mode and r["gain"] == g]
    line = f"  {mode:6s} g{g:<3} whole: " + "  ".join(f"{k} {np.mean([r[k] for r in rr]):5.1f}" for k in ("A_pure", "B_pure", "rule", "learned", "oracle"))
    print(line)
    if mode == "active":
        print(f"  {'':6s} {'':4s} active line: " + "  ".join(f"{k} {np.mean([r[k + '_active'] for r in rr]):5.1f}" for k in ("A_pure", "B_pure", "rule", "learned", "oracle")))
        m, lo, hi, w, n = paired(mode, g, "learned", lambda r: r["A_pure_active"], "_active")
        print(f"  {'':6s} {'':4s} active line learned - A = {m:+.1f} [{lo:+.1f}, {hi:+.1f}], learned better in {w}/{n}")
out.write_text(json.dumps(res, indent=1))
