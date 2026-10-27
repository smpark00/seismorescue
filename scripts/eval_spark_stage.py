"""Score stage 2: learned judge alone vs learned judge + Spark on its least confident events.

usage: python -I eval_spark_stage.py <bench_out_dir> <cache_dir> <judges_dir>

Event level (non-tie events that went to Spark): accuracy of the learned judge, of Spark where its two swapped
answers agree, swap-consistency rate, and an exact McNemar test (learned vs learned+Spark) on the same events.
Record level: assemble paths with learned choices everywhere vs with consistent Spark answers replacing them, then
the usual fail metrics, with a cluster bootstrap (groups = event dates) on the paired difference.
"""
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from arbiter import assemble, event_table
from track_joint import score

root, cache, jdir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
TAU = 6.0
preds = {(p["record"], p["mode"], p["gain"], p["line"], p["s"]): p for p in json.loads((jdir / "event_predictions.json").read_text())}
ans = {}
for line in (jdir / "spark_event_answers.jsonl").read_text().splitlines():
    a = json.loads(line)
    ans.setdefault((a["record"], a["mode"], a["gain"], a["line"], a["s"]), {})[a["swap"]] = a

spark = {}                                        # event key -> "A"/"B" if the two swapped answers agree, else None
for k, v in ans.items():
    if False in v and True in v:
        p0, p1 = v[False]["pick"], v[True]["pick"]
        spark[k] = p0 if (p0 is not None and p0 == p1) else None
n_cons = sum(v is not None for v in spark.values())
print(f"judged events {len(spark)}; swap-consistent {n_cons} ({n_cons / max(1, len(spark)):.1%})")

# event level
ev = [(k, preds[k]) for k in spark if preds[k]["label"] >= 0]
truth = {k: "A" if p["label"] == 1 else "B" for k, p in ev}
learned = {k: "A" if p["p_A"] > 0.5 else "B" for k, p in ev}
combined = {k: (spark[k] or learned[k]) for k, _ in ev}
cons = [k for k, _ in ev if spark[k] is not None]
acc = lambda d, keys: float(np.mean([d[k] == truth[k] for k in keys])) if keys else float("nan")
print(f"non-tie judged events {len(ev)}: learned {acc(learned, [k for k, _ in ev]):.3f} | learned+Spark {acc(combined, [k for k, _ in ev]):.3f}")
print(f"  on swap-consistent events ({len(cons)}): Spark {acc({k: spark[k] for k in cons}, cons):.3f} vs learned {acc(learned, cons):.3f}")
b = sum(learned[k] == truth[k] and combined[k] != truth[k] for k, _ in ev)
c = sum(learned[k] != truth[k] and combined[k] == truth[k] for k, _ in ev)
pval = min(1.0, 2 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
print(f"  McNemar exact: Spark fixed {c}, broke {b}, p = {pval:.4f}")

# record level
rows_out = []
for rec in sorted(p for p in root.iterdir() if p.is_dir()):
    for mode, g in [("all", 2.0), ("active", 2.0)]:
        d = rec / mode
        meta = json.loads((d / "meta.json").read_text())
        z = np.load(cache / f"{rec.name}_{mode}_g{g:g}.npz")
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        S, x0 = meta["spacing"], meta["margin_x"]
        x1 = pen.shape[1] - x0
        rows = event_table(z["A"], z["B"], z["ink"], x0, x1, TAU)
        keys = [(rec.name, mode, g, r["line"], r["s"]) for r in rows]
        ch_l = ["A" if preds[k]["p_A"] > 0.5 else "B" for k in keys]
        ch_s = [spark.get(k) or c for k, c in zip(keys, ch_l)]
        res = {"record": rec.name, "group": rec.name[:8], "mode": mode, "gain": g,
               "spark_events": sum(k in spark for k in keys), "events": len(keys)}
        for name, ch in [("learned", ch_l), ("learned+spark", ch_s)]:
            est = assemble(z["A"], z["B"], rows, ch)
            res[name] = score(est, pen, range(pen.shape[0]), x0, x1, S)[0]
            if mode == "active":
                res[name + "_active"] = score(est, pen, [meta["active_line"]], x0, x1, S)[0]
        rows_out.append(res)


def paired_ci(vals, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    gs = sorted({v[0] for v in vals})
    by = {g: np.array([v[1] for v in vals if v[0] == g]) for g in gs}
    means = [np.concatenate([by[gs[i]] for i in rng.choice(len(gs), len(gs))]).mean() for _ in range(n)]
    return np.mean([v[1] for v in vals]), np.percentile(means, 2.5), np.percentile(means, 97.5)


for mode, metric in [("all", ""), ("active", ""), ("active", "_active")]:
    rr = [r for r in rows_out if r["mode"] == mode]
    l = np.mean([r["learned" + metric] for r in rr]); s_ = np.mean([r["learned+spark" + metric] for r in rr])
    m, lo, hi = paired_ci([(r["group"], r["learned+spark" + metric] - r["learned" + metric]) for r in rr])
    print(f"{mode:6s} g2 {('active line' if metric else 'whole image'):11s}: learned {l:5.1f} -> +Spark {s_:5.1f}  "
          f"(paired diff {m:+.2f}, 95% CI {lo:+.2f}..{hi:+.2f})")
toks = [a.get("out_tokens") or 0 for v in ans.values() for a in v.values()]
secs = [a.get("secs") or 0 for v in ans.values() for a in v.values()]
print(f"Spark calls {len(toks)}, output tokens {sum(toks):,} (~${sum(toks) * 0.2 / 1e6:.2f} at contributor output price), median {np.median(secs):.0f} s/call")
(jdir / "spark_stage_results.json").write_text(json.dumps({"records": rows_out, "mcnemar": {"fixed": c, "broke": b, "p": pval},
                                                           "swap_consistency": n_cons / max(1, len(spark))}, indent=1))
