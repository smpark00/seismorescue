"""Grouped CV of event judges on cached proposals: tuned rule vs learned judge (logistic regression on event context).

usage: python -I cv_judges.py <bench_out_dir> <cache_dir> <out_dir>
(importable: load_items / ctx_features / fit_lr / evaluate are reused by holdout_eval.py with the method frozen)

Stage-1 CV (cv_arbiter.py) showed the arbiter architecture has headroom (GT oracle ~halves failures) but the
hand-weighted rule does not know which proposer to trust, e.g. on the large-amplitude line. The learned judge sees the
rule features of both segments plus context (line amplitude, event length, A-B separation) and is trained on the
training groups' events with GT-oracle labels; ties (same failed columns) are left out of training.
Writes per-event test predictions (with probability) so the Spark stage can take the least confident ones.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from arbiter import assemble, event_table
from track_joint import score

CONDS = [("all", 2.0), ("all", 5.0), ("active", 1.5), ("active", 2.0)]
TAU, K, REPEATS = 6.0, 4, 3
W_RULE = np.array([0.5, 20.0, 5.0])               # the weights stage-1 CV picked most often


def line_amp(P, k, it):
    seg = P[k, it["x0"]:it["x1"]]
    return float(np.nanpercentile(np.abs(seg - np.nanmedian(seg)), 95) / it["S"])


def ctx_features(it, r):
    A, B, S = it["A"], it["B"], it["S"]
    k, s, e = r["line"], r["s"], r["e"]
    # amplitude from both proposers: B tends to lose the big trace, so B alone under-reports it;
    # is_top = this line swings most among A's lines (the "active line" regime where A is usually right)
    amp_a = [line_amp(A, j, it) for j in range(A.shape[0])]
    amp = max(amp_a[k], line_amp(B, k, it))
    is_top = float(k == int(np.argmax(amp_a)))
    # baseline centring: a band-passed trace oscillates about its own zero line, so over a span the true path's median
    # stays near the line's baseline while a path that jumped onto a neighbour's trace sits ~S away.
    # baseline (no GT): median of the columns where both proposers agree on this line
    agree = np.abs(A[k, it["x0"]:it["x1"]] - B[k, it["x0"]:it["x1"]]) <= TAU
    base = float(np.nanmedian(B[k, it["x0"]:it["x1"]][agree])) if agree.any() else float(np.nanmedian(B[k, it["x0"]:it["x1"]]))
    cA, cB = abs(np.nanmedian(A[k, s:e]) - base) / S, abs(np.nanmedian(B[k, s:e]) - base) / S
    sep = float(np.nanmean(np.abs(A[k, s:e] - B[k, s:e])) / S)
    d = r["fA"] - r["fB"]
    return np.nan_to_num(np.r_[d, r["dupAA"], r["dupBB"], amp, amp * d, amp * r["dupAA"], cA - cB, cA, cB, (cA - cB) * (e - s) / 100, (e - s) / 100, sep, 1.0], nan=0.0)


def load_items(root, cache, conds=None):
    """one item per record x condition: cached proposals, events, features, GT labels (labels only for training)"""
    items = []
    for rec in sorted(p for p in root.iterdir() if p.is_dir()):
        for mode, g in (conds or CONDS):
            d = rec / mode
            meta = json.loads((d / "meta.json").read_text())
            z = np.load(cache / f"{rec.name}_{mode}_g{g:g}.npz")
            pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
            S, x0 = meta["spacing"], meta["margin_x"]
            x1 = pen.shape[1] - x0
            it = {"record": rec.name, "group": rec.name[:8], "mode": mode, "gain": g, "S": S, "x0": x0, "x1": x1,
                  "active": meta["active_line"], "A": z["A"], "B": z["B"], "pen": pen}
            it["rows"] = event_table(z["A"], z["B"], z["ink"], x0, x1, TAU)
            for r in it["rows"]:
                k, s, e = r["line"], r["s"], r["e"]
                fa = int(np.nansum(np.abs(z["A"][k, s:e] - pen[k, s:e]) > S / 2))
                fb = int(np.nansum(np.abs(z["B"][k, s:e] - pen[k, s:e]) > S / 2))
                r["label"] = 1 if fa < fb else (0 if fb < fa else -1)     # 1: A better, 0: B better, -1: tie
                r["stake"] = abs(fa - fb)                                  # columns won or lost by this decision
                r["x"] = ctx_features(it, r)
                r["rule_A"] = float(W_RULE @ r["fA"]) < float(W_RULE @ r["fB"])
            items.append(it)
    return items


def fit_lr(X, y, wt, l2=1.0):
    mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-9
    Z = np.c_[(X[:, :-1] - mu) / sd, X[:, -1]]

    def loss(w):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        return -np.sum(wt * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))) / wt.sum() + l2 * np.sum(w[:-1] ** 2) / len(y)
    w = minimize(loss, np.zeros(Z.shape[1]), method="L-BFGS-B").x
    return lambda Xn: 1 / (1 + np.exp(-np.clip(np.c_[(Xn[:, :-1] - mu) / sd, Xn[:, -1]] @ w, -30, 30)))


def evaluate(it, choice):
    est = assemble(it["A"], it["B"], it["rows"], choice)
    N = it["pen"].shape[0]
    f = score(est, it["pen"], range(N), it["x0"], it["x1"], it["S"])[0]
    a = score(est, it["pen"], [it["active"]], it["x0"], it["x1"], it["S"])[0] if it["mode"] in ("active", "film") else None
    return f, a


if __name__ == "__main__":
    root, cache, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    items = load_items(root, cache)
    groups = sorted({it["group"] for it in items})
    res, preds = [], []
    for rep in range(REPEATS):
        gs = groups[:]
        random.Random(rep).shuffle(gs)
        for f in [gs[i::K] for i in range(K)]:
            train = [r for it in items if it["group"] not in f for r in it["rows"] if r["label"] >= 0]
            # weight = columns at stake, so the judge optimises failed columns (the metric), not event count
            model = fit_lr(np.array([r["x"] for r in train]), np.array([r["label"] for r in train], float),
                           np.array([r["stake"] for r in train], float))
            for it in items:
                if it["group"] not in f or not it["rows"]:
                    continue
                p = model(np.array([r["x"] for r in it["rows"]]))
                for meth, ch in [("rule", ["A" if r["rule_A"] else "B" for r in it["rows"]]),
                                 ("learned", ["A" if q > 0.5 else "B" for q in p])]:
                    fl, al = evaluate(it, ch)
                    res.append({"repeat": rep, "group": it["group"], "record": it["record"], "mode": it["mode"],
                                "gain": it["gain"], "method": meth, "fail": fl, "active_line_fail": al})
                if rep == 0:
                    for r, q in zip(it["rows"], p):
                        preds.append({"record": it["record"], "mode": it["mode"], "gain": it["gain"], "line": r["line"],
                                      "s": r["s"], "e": r["e"], "p_A": round(float(q), 4), "label": r["label"], "stake": r["stake"],
                                      "rule_A": bool(r["rule_A"])})

    # event-level accuracy on non-tie events (repeat 0 predictions)
    ev = [p for p in preds if p["label"] >= 0]
    acc_l = np.mean([(p["p_A"] > 0.5) == (p["label"] == 1) for p in ev])
    acc_r = np.mean([p["rule_A"] == (p["label"] == 1) for p in ev])
    print(f"event accuracy (non-tie, n={len(ev)}): rule {acc_r:.3f}  learned {acc_l:.3f}")
    conf = sorted(ev, key=lambda p: abs(p["p_A"] - 0.5))
    for q in (0.1, 0.2, 0.3):
        low = conf[:int(q * len(conf))]
        print(f"  least-confident {int(q*100)}% of events: learned acc {np.mean([(p['p_A'] > 0.5) == (p['label'] == 1) for p in low]):.3f}")
    summary = []
    for mode, g in CONDS:
        for metric in ["fail"] + (["active_line_fail"] if mode == "active" else []):
            row = {"mode": mode, "gain": g, "metric": metric}
            for meth in ("rule", "learned"):
                v = [r[metric] for r in res if r["mode"] == mode and r["gain"] == g and r["method"] == meth]
                row[meth] = round(float(np.mean(v)), 1)
            summary.append(row)
            print(f"{mode:6s} g{g:<3} {metric:16s} rule {row['rule']:5.1f}  learned {row['learned']:5.1f}")
    (out / "cv_judges_results.json").write_text(json.dumps({"summary": summary, "cv": res}, indent=1))
    (out / "event_predictions.json").write_text(json.dumps(preds))
