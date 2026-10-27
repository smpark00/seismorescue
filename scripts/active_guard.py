"""Active-line guard: find the large-amplitude line without GT and trust proposer A on it (docs/RESULTS.md).

usage: python -I active_guard.py <train_bench> <train_cache> <train_event_predictions.json> <dev2_bench> <dev2_cache> <out.json>

Why: greedy A follows the big trace well, but its quiet neighbours jump onto that trace too. So A alone cannot say
which line owns it, and the learned judge (trained on all lines) under-trusts A there (holdout: +9.4 pts on the
active line at gain 1.5). Proposer B keeps quiet lines almost perfectly. Erase the ink under B's quiet lines: the
leftover ink belongs to the big trace, and since a band-passed trace oscillates about its own zero line, its median
row sits on that line's baseline. If at least Q lines are quiet (single-active regime), events on that line take A
unless the judge's p(A) < TAU_LOW. Q and TAU_LOW are chosen by grouped CV on the development records only.
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cv_judges import CONDS, evaluate, fit_lr, load_items

Q_GRID, TAU_GRID = [5, 6, 7], [0.0]   # C replaces the line's path; tau is unused


def layout_bases(n, MY, S, bases=None):
    """line baselines from the layout agent: uniform grid by default, or measured per line (real scans)"""
    return np.asarray(bases, float) if bases is not None else MY + S * np.arange(n)


def detect_active(B, ink, MY, S, x0, x1, quiet_amp=0.3, erase=4, bases=None):
    """-> (active line index or None, number of quiet lines). GT-free: uses B's paths, the ink mask and the layout"""
    n = B.shape[0]
    bl = layout_bases(n, MY, S, bases)
    quiet = [np.nanpercentile(np.abs(B[k, x0:x1] - bl[k]), 95) < quiet_amp * S for k in range(n)]
    res = clean_ink(B, ink, MY, S, x0, x1, quiet_amp, erase, bases)
    ys = np.nonzero(res[:, x0:x1])[0]
    if ys.size < 200:
        return None, int(sum(quiet))
    c = float(np.median(ys))
    return int(np.argmin(np.abs(c - bl))), int(sum(quiet))


def clean_ink(B, ink, MY, S, x0, x1, quiet_amp=0.3, erase=4, bases=None):
    """ink with B's quiet lines erased: what remains belongs to the big trace (and stray ink)"""
    bl = layout_bases(B.shape[0], MY, S, bases)
    res, H, cols = ink.copy(), ink.shape[0], np.arange(x0, x1)
    for k in range(B.shape[0]):
        if np.nanpercentile(np.abs(B[k, x0:x1] - bl[k]), 95) < quiet_amp * S:
            y = np.round(np.nan_to_num(B[k, x0:x1], nan=bl[k])).astype(int)
            for dy in range(-erase, erase + 1):
                res[np.clip(y + dy, 0, H - 1), cols] = False
    return res


def track_active_clean(B, ink, y_start, MY, S, x0, x1, bases=None):
    """proposer C: greedy tracking of the big trace on clean ink, where the quiet neighbours it used to jump onto
    are gone; gaps (erased crossings) are bridged linearly"""
    from track_joint import tracker_greedy
    c = tracker_greedy(clean_ink(B, ink, MY, S, x0, x1, bases=bases), y_start, x0, x1, S)
    idx = np.arange(x0, x1)
    ok = ~np.isnan(c[x0:x1])
    if ok.sum() > 2:
        c[x0:x1] = np.interp(idx, idx[ok], c[x0:x1][ok])
    return c


def guard_choice(rows, p, active, n_quiet, q, tau_low):
    ch = ["A" if v > 0.5 else "B" for v in p]
    if active is None or n_quiet < q:
        return ch
    return ["A" if (r["line"] == active and v >= tau_low) else c for r, v, c in zip(rows, p, ch)]


def attach(items, bench):
    """add the detector output to each item (needs the ink mask from the proposer cache)"""
    for it in items:
        meta = json.loads((bench / it["record"] / it["mode"] / "meta.json").read_text())
        it["detected"], it["n_quiet"] = detect_active(it["B"], it["ink"], meta["margin_y"], it["S"], it["x0"], it["x1"])
    return items


if __name__ == "__main__":
    tb, tc, tpred, db, dc, out = sys.argv[1:7]
    train = load_items(Path(tb), Path(tc))
    dev2 = load_items(Path(db), Path(dc))
    for it, cache in [(x, tc) for x in train] + [(x, dc) for x in dev2]:
        it["ink"] = np.load(Path(cache) / f"{it['record']}_{it['mode']}_g{it['gain']:g}.npz")["ink"]
    attach(train, Path(tb)); attach(dev2, Path(db))
    for it in train + dev2:                                  # proposer C for the detected line (start = layout start)
        it["C"] = None
        if it["detected"] is not None:
            bench = Path(tb) if it in train else Path(db)
            meta = json.loads((bench / it["record"] / it["mode"] / "meta.json").read_text())
            it["C"] = track_active_clean(it["B"], it["ink"], it["pen"][it["detected"], it["x0"]], meta["margin_y"], it["S"], it["x0"], it["x1"])
    # out-of-sample judge probabilities: CV-held-out for the original 24, train-on-24 model for the first holdout
    pv = {(p["record"], p["mode"], p["gain"], p["line"], p["s"]): p["p_A"] for p in json.loads(Path(tpred).read_text())}
    for it in train:
        it["p"] = np.array([pv[(it["record"], it["mode"], it["gain"], r["line"], r["s"])] for r in it["rows"]])
    rows = [r for it in train for r in it["rows"] if r["label"] >= 0]
    model = fit_lr(np.array([r["x"] for r in rows]), np.array([r["label"] for r in rows], float), np.array([r["stake"] for r in rows], float))
    for it in dev2:
        it["p"] = model(np.array([r["x"] for r in it["rows"]])) if it["rows"] else np.array([])
    dev = train + dev2
    act = [it for it in dev if it["mode"] == "active"]
    print(f"detector on active records: correct {np.mean([it['detected'] == it['active'] for it in act]):.0%} of {len(act)}")

    from arbiter import assemble
    from track_joint import score

    def eval_c(it, q):
        """judge everywhere; in the single-active regime the detected line takes proposer C's path"""
        est = assemble(it["A"], it["B"], it["rows"], ["A" if v > 0.5 else "B" for v in it["p"]])
        if it["C"] is not None and it["n_quiet"] >= q:
            est[it["detected"], it["x0"]:it["x1"]] = it["C"][it["x0"]:it["x1"]]
        N = it["pen"].shape[0]
        f = score(est, it["pen"], range(N), it["x0"], it["x1"], it["S"])[0]
        a = score(est, it["pen"], [it["active"]], it["x0"], it["x1"], it["S"])[0] if it["mode"] == "active" else None
        return f, a

    def score_cfg(its, q, tau):
        f, a = [], []
        for it in its:
            fl, al = eval_c(it, q)
            f.append(fl); a.append(al if al is not None else np.nan)
        return np.mean(f) + np.nanmean(a), f, a

    groups = sorted({it["group"] for it in dev})
    res = []
    for rep in range(3):
        gs = groups[:]
        random.Random(rep).shuffle(gs)
        for fold in [gs[i::4] for i in range(4)]:
            tr = [it for it in dev if it["group"] not in fold]
            q, tau = min(((q, t) for q in Q_GRID for t in TAU_GRID), key=lambda c: score_cfg(tr, *c)[0])
            for it in dev:
                if it["group"] in fold:
                    g_f, g_a = eval_c(it, q)
                    j_f, j_a = evaluate(it, ["A" if v > 0.5 else "B" for v in it["p"]])
                    res.append({"rep": rep, "record": it["record"], "group": it["group"], "mode": it["mode"], "gain": it["gain"],
                                "q": q, "tau": tau, "guard": g_f, "guard_active": g_a, "judge": j_f, "judge_active": j_a})
    for mode, g in CONDS:
        rr = [r for r in res if r["mode"] == mode and r["gain"] == g]
        line = f"{mode:6s} g{g:<3} whole: judge {np.mean([r['judge'] for r in rr]):5.1f} guard {np.mean([r['guard'] for r in rr]):5.1f}"
        if mode == "active":
            line += f" | active line: judge {np.mean([r['judge_active'] for r in rr]):5.1f} guard {np.mean([r['guard_active'] for r in rr]):5.1f}"
        print(line)
    picked = {}
    for r in res:
        picked[f"Q={r['q']} tau={r['tau']}"] = picked.get(f"Q={r['q']} tau={r['tau']}", 0) + 1
    print("configs chosen:", picked)
    Path(out).write_text(json.dumps(res, indent=1))
