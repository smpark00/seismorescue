"""Global assembly agent (docs/RESULTS.md P4): decide time-overlapping events together, with an exclusivity penalty.

usage: python -I cv_joint_assembly.py <bench_out_dir> <cache_dir> <out_dir>

Per-event judging cannot tell which of two lines owns a trace they share: greedy A lets a quiet neighbour jump onto
the big trace, so A's (correct) active-line segment looks "duplicated" and gets rejected. Here the learned judge
scores only intrinsic segment quality (roughness, off-ink, amplitude context, A-B separation, length; no duplication
features), and events whose spans overlap in time form a cluster. Each cluster's 2^m choices are enumerated
(coordinate descent above MAX_ENUM) with cost = sum(-log p(choice)) + LAMBDA * columns where two final paths sit
within 2 px. LAMBDA is chosen on the training groups (GroupKFold by event date, 3 shuffles).
"""
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from arbiter import event_table
from track_joint import score

root, cache, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
CONDS = [("all", 2.0), ("all", 5.0), ("active", 1.5), ("active", 2.0)]
TAU, K, REPEATS, MAX_ENUM = 6.0, 4, 3, 10
LAMBDAS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5]


def line_amp(P, k, x0, x1, S):
    seg = P[k, x0:x1]
    return float(np.nanpercentile(np.abs(seg - np.nanmedian(seg)), 95) / S)


items = []
for rec in sorted(p for p in root.iterdir() if p.is_dir()):
    for mode, g in CONDS:
        d = rec / mode
        meta = json.loads((d / "meta.json").read_text())
        z = np.load(cache / f"{rec.name}_{mode}_g{g:g}.npz")
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        S, x0 = meta["spacing"], meta["margin_x"]
        x1 = pen.shape[1] - x0
        A, B = z["A"], z["B"]
        rows = event_table(A, B, z["ink"], x0, x1, TAU)
        for r in rows:
            k, s, e = r["line"], r["s"], r["e"]
            fa = int(np.nansum(np.abs(A[k, s:e] - pen[k, s:e]) > S / 2))
            fb = int(np.nansum(np.abs(B[k, s:e] - pen[k, s:e]) > S / 2))
            r["label"], r["stake"] = (1 if fa < fb else (0 if fb < fa else -1)), abs(fa - fb)
            amp = max(line_amp(A, k, x0, x1, S), line_amp(B, k, x0, x1, S))
            dq = (r["fA"] - r["fB"])[:2]                           # roughness, off-ink (no duplication)
            sep = float(np.nanmean(np.abs(A[k, s:e] - B[k, s:e])) / S)
            r["x"] = np.nan_to_num(np.r_[dq, amp, amp * dq, (e - s) / 100, sep, 1.0], nan=0.0)
        # clusters: events whose spans overlap in time (any line)
        order = sorted(range(len(rows)), key=lambda i: rows[i]["s"])
        clusters, cur, end = [], [], -1
        for i in order:
            if cur and rows[i]["s"] >= end:
                clusters.append(cur); cur = []
            cur.append(i); end = max(end, rows[i]["e"]) if len(cur) > 1 else rows[i]["e"]
        if cur:
            clusters.append(cur)
        items.append({"record": rec.name, "group": rec.name[:8], "mode": mode, "gain": g, "S": S, "x0": x0, "x1": x1,
                      "active": meta["active_line"], "A": A, "B": B, "pen": pen, "rows": rows, "clusters": clusters})
    print(rec.name, flush=True)


def fit_lr(X, y, wt, l2=1.0):
    mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-9
    Z = np.c_[(X[:, :-1] - mu) / sd, X[:, -1]]

    def loss(w):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        return -np.sum(wt * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))) / wt.sum() + l2 * np.sum(w[:-1] ** 2) / len(y)
    w = minimize(loss, np.zeros(Z.shape[1]), method="L-BFGS-B").x
    return lambda Xn: 1 / (1 + np.exp(-np.clip(np.c_[(Xn[:, :-1] - mu) / sd, Xn[:, -1]] @ w, -30, 30)))


def assemble_joint(it, p, lam):
    """choose A/B per event, cluster by cluster, minimising judge NLL + lam * duplicated columns"""
    A, B, rows = it["A"], it["B"], it["rows"]
    est = B.copy()
    choice = np.array(["B"] * len(rows), object)
    for cl in it["clusters"]:
        s0, e0 = min(rows[i]["s"] for i in cl), max(rows[i]["e"] for i in cl)
        nll = {i: (-np.log(p[i] + 1e-9), -np.log(1 - p[i] + 1e-9)) for i in cl}   # (cost if A, cost if B)

        def cost(ch):
            P = B[:, s0:e0].copy()
            for i, c in zip(cl, ch):
                if c:
                    r = rows[i]
                    P[r["line"], r["s"] - s0:r["e"] - s0] = A[r["line"], r["s"]:r["e"]]
            dup = 0
            if lam > 0:
                for a, b in itertools.combinations(range(P.shape[0]), 2):
                    dup += int(np.sum(np.nan_to_num(np.abs(P[a] - P[b]), nan=99) < 2.0))
            return sum(nll[i][0] if c else nll[i][1] for i, c in zip(cl, ch)) + lam * dup
        if lam == 0:
            best = tuple(p[i] > 0.5 for i in cl)
        elif len(cl) <= MAX_ENUM:
            best = min(itertools.product([False, True], repeat=len(cl)), key=cost)
        else:                                         # coordinate descent from the judge's own choices
            best = [p[i] > 0.5 for i in cl]
            improved = True
            while improved:
                improved = False
                for j in range(len(cl)):
                    alt = best[:]; alt[j] = not alt[j]
                    if cost(alt) < cost(best):
                        best, improved = alt, True
            best = tuple(best)
        for i, c in zip(cl, best):
            choice[i] = "A" if c else "B"
            if c:
                r = rows[i]
                est[r["line"], r["s"]:r["e"]] = A[r["line"], r["s"]:r["e"]]
    return est, choice


def evaluate(it, est):
    f = score(est, it["pen"], range(it["pen"].shape[0]), it["x0"], it["x1"], it["S"])[0]
    a = score(est, it["pen"], [it["active"]], it["x0"], it["x1"], it["S"])[0] if it["mode"] == "active" else None
    return f, a


def objective(sc):
    f = np.mean([s[0] for s in sc]); a = [s[1] for s in sc if s[1] is not None]
    return f + (np.mean(a) if a else 0)


groups = sorted({it["group"] for it in items})
res = []
for rep in range(REPEATS):
    gs = groups[:]
    random.Random(rep).shuffle(gs)
    for fold in [gs[i::K] for i in range(K)]:
        tr_rows = [r for it in items if it["group"] not in fold for r in it["rows"] if r["label"] >= 0]
        model = fit_lr(np.array([r["x"] for r in tr_rows]), np.array([r["label"] for r in tr_rows], float),
                       np.array([r["stake"] for r in tr_rows], float))
        probs = {id(it): (model(np.array([r["x"] for r in it["rows"]])) if it["rows"] else np.array([])) for it in items}
        # lambda chosen on training groups only
        train = [it for it in items if it["group"] not in fold]
        lam = min(LAMBDAS, key=lambda l: objective([evaluate(it, assemble_joint(it, probs[id(it)], l)[0]) for it in train]))
        for it in items:
            if it["group"] in fold:
                f, a = evaluate(it, assemble_joint(it, probs[id(it)], lam)[0])
                res.append({"repeat": rep, "group": it["group"], "record": it["record"], "mode": it["mode"],
                            "gain": it["gain"], "lambda": lam, "fail": f, "active_line_fail": a})
        print(f"repeat {rep} fold {fold}: lambda={lam}", flush=True)

for mode, g in CONDS:
    rr = [r for r in res if r["mode"] == mode and r["gain"] == g]
    line = f"{mode:6s} g{g:<3} joint-assembly CV: whole {np.mean([r['fail'] for r in rr]):5.1f}"
    if mode == "active":
        line += f"  active line {np.mean([r['active_line_fail'] for r in rr]):5.1f}"
    print(line)
(out / "cv_joint_assembly_results.json").write_text(json.dumps(res, indent=1))
