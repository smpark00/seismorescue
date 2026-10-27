"""Grouped cross-validation of the offline arbiter (stage 1 of docs/RESULTS.md-multiagent-design.md).

usage: python -I cv_arbiter.py <bench_out_dir> <cache_dir> <results.json>

1) caches both proposers' paths per record/condition (slow part, once);
2) builds the event table per TAU;
3) GroupKFold by event date (groups never split), repeated with 3 group shuffles: the rule weights and TAU are chosen
   on the training groups, then scored on the test groups. Also scores A alone, B alone and the GT oracle (upper bound).
Objective = mean whole-image fail over all conditions + mean active-line fail over active conditions.
"""
import itertools
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from arbiter import assemble, decide, event_table, oracle_choice, proposals
from track_joint import score

root, cache, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
cache.mkdir(parents=True, exist_ok=True)
CONDS = [("all", 2.0), ("all", 5.0), ("active", 1.5), ("active", 2.0)]
TAUS = [6.0, 12.0]
GRID = [np.array(w) for w in itertools.product([0.5, 1, 2], [5, 10, 20], [5, 10, 20])]
K, REPEATS = 4, 3

items = []                                         # one per record x condition
for rec in sorted(p for p in root.iterdir() if p.is_dir()):
    for mode, g in CONDS:
        d = rec / mode
        meta = json.loads((d / "meta.json").read_text())
        S, MX, N = meta["spacing"], meta["margin_x"], meta["n_lines"]
        img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        x0, x1 = MX, img.shape[1] - MX
        cf = cache / f"{rec.name}_{mode}_g{g:g}.npz"
        if cf.exists():
            z = np.load(cf); A, B, ink = z["A"], z["B"], z["ink"]
        else:
            A, B, ink = proposals(img, pen[:, x0], x0, x1, S)
            np.savez_compressed(cf, A=A, B=B, ink=ink)
        tables = {t: event_table(A, B, ink, x0, x1, t) for t in TAUS}
        items.append({"record": rec.name, "group": rec.name[:8], "mode": mode, "gain": g, "S": S, "x0": x0, "x1": x1,
                      "active": meta["active_line"], "A": A, "B": B, "pen": pen, "tables": tables})
    print(rec.name, flush=True)


def evaluate(it, est):
    N = it["pen"].shape[0]
    fail = score(est, it["pen"], range(N), it["x0"], it["x1"], it["S"])[0]
    act = score(est, it["pen"], [it["active"]], it["x0"], it["x1"], it["S"])[0] if it["mode"] == "active" else None
    return fail, act


def objective(scores):
    f = np.mean([s[0] for s in scores])
    a = [s[1] for s in scores if s[1] is not None]
    return f + (np.mean(a) if a else 0)


def run_cfg(it, tau, w):
    rows = it["tables"][tau]
    return evaluate(it, assemble(it["A"], it["B"], rows, [c for c, _ in decide(rows, w)]))


# fixed reference methods
ref = {}
for it in items:
    key = (it["record"], it["mode"], it["gain"])
    rows = it["tables"][6.0]
    ref[key] = {"A": evaluate(it, it["A"]), "B": evaluate(it, it["B"]),
                "oracle": evaluate(it, assemble(it["A"], it["B"], rows, oracle_choice(it["A"], it["B"], rows, it["pen"], it["S"]))),
                "n_events": len(rows)}

# precompute every (tau, w) score per item: tuning is then a lookup
cfgs = [(t, i) for t in TAUS for i in range(len(GRID))]
table = {(id(it), c): run_cfg(it, c[0], GRID[c[1]]) for it in items for c in cfgs}

groups = sorted({it["group"] for it in items})
test_scores = []                                   # (repeat, item key, score, chosen cfg)
for rep in range(REPEATS):
    gs = groups[:]
    random.Random(rep).shuffle(gs)
    folds = [gs[i::K] for i in range(K)]
    for f in folds:
        train = [it for it in items if it["group"] not in f]
        test = [it for it in items if it["group"] in f]
        best = min(cfgs, key=lambda c: objective([table[(id(it), c)] for it in train]))
        for it in test:
            test_scores.append({"repeat": rep, "record": it["record"], "group": it["group"], "mode": it["mode"],
                                "gain": it["gain"], "fail": table[(id(it), best)][0],
                                "active_line_fail": table[(id(it), best)][1],
                                "tau": best[0], "w": GRID[best[1]].tolist()})


def boot_ci(vals_by_group, n=2000, seed=0):
    """cluster bootstrap over groups -> 95% CI of the mean"""
    rng = np.random.default_rng(seed)
    keys = list(vals_by_group)
    means = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys))
        v = np.concatenate([vals_by_group[keys[i]] for i in pick])
        means.append(v.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


summary = []
for mode, g in CONDS:
    for metric in (["fail"] + (["active_line_fail"] if mode == "active" else [])):
        row = {"mode": mode, "gain": g, "metric": metric}
        idx = 0 if metric == "fail" else 1
        for m in ("A", "B", "oracle"):
            byg = {}
            for it in items:
                if it["mode"] == mode and it["gain"] == g:
                    byg.setdefault(it["group"], []).append(ref[(it["record"], mode, g)][m][idx])
            vals = np.concatenate([np.array(v) for v in byg.values()])
            row[m] = [round(float(vals.mean()), 1), *[round(x, 1) for x in boot_ci({k: np.array(v) for k, v in byg.items()})]]
        byg = {}
        for s in test_scores:
            if s["mode"] == mode and s["gain"] == g:
                byg.setdefault(s["group"], []).append(s[metric])
        vals = np.concatenate([np.array(v) for v in byg.values()])
        row["rule_arbiter_cv"] = [round(float(vals.mean()), 1), *[round(x, 1) for x in boot_ci({k: np.array(v) for k, v in byg.items()})]]
        summary.append(row)
        print(f"{mode:6s} g{g:<3} {metric:16s} A {row['A']}  B {row['B']}  rule-arbiter(CV) {row['rule_arbiter_cv']}  oracle {row['oracle']}")
chosen = {}
for s in test_scores:
    k = f"tau={s['tau']} w={s['w']}"
    chosen[k] = chosen.get(k, 0) + 1
print("configs chosen across folds:", chosen)
print("events per item: mean", round(float(np.mean([v["n_events"] for v in ref.values()])), 1))
out.write_text(json.dumps({"summary": summary, "test_scores": test_scores, "chosen": chosen,
                           "ref": {"|".join(map(str, k)): v for k, v in ref.items()}}, indent=1))
