"""Spike S2: can Muse Spark pick the right trace identity at a touch/crossing when shown two coloured hypotheses?

usage: python -I exp_spark_crossing_judge.py <scripts_dir> <synth_dir> <synth_active_dir> <out_dir> [n_cases]

Cases are GT touch events (two lines < 3 px apart) on the gain-2 all-lines record and the gain-1.5/2 single-active
records, balanced between true crossings (line order flips) and kisses (lines touch and separate again).
Each case image stacks RAW plus panels A/B: left of the touch both colour the two lines the same (red, blue);
right of it one panel swaps them. Which panel is correct is random. Hypothesis paths are GT (oracle candidates),
so this measures the judge only. Baseline: two-sided cubic smoothness rule on the same two hypotheses, no image.
"""
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
from metaapi import spark_ask

d_all, d_act, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
N_CASES = int(sys.argv[5]) if len(sys.argv) > 5 else 40
HALF, ZOOM, LOOK, PAD = 80, 3, 20, 15
rng = random.Random(0)
Q = ("This image has three panels cut from a scanned analog seismogram (dark pen traces on paper), enlarged 3x. "
     "RAW is the original. Two pen traces touch or cross near the horizontal centre. Panels A and B overlay two "
     "hypotheses for which trace is which: left of the centre both panels colour the traces identically (red, blue); "
     "right of the centre they differ. A real pen trace is one continuous smooth curve whose slope and curvature do "
     "not jump. Which panel colours the traces so that each colour follows one physical pen trace? Answer only JSON: "
     "{\"answer\": \"A\" or \"B\", \"confidence\": number 0-1, \"reason\": short string}.")


def events(pen):
    """touch events between any line pair -> (i, j, centre column, 'cross'|'kiss')"""
    ev, (n, W) = [], pen.shape
    for i in range(n):
        for j in range(i + 1, n):
            dd = pen[i] - pen[j]
            t = np.nan_to_num(np.abs(dd), nan=99) < 3
            for s in np.flatnonzero(np.diff(t.astype(int)) == 1) + 1:
                e = s
                while e < W and t[e]:
                    e += 1
                c = (s + e) // 2
                if c - HALF - LOOK < 0 or c + HALF + LOOK >= W or e - s > 20:
                    continue
                # both hypotheses must look different: lines clearly apart before and after the touch
                if not (abs(dd[s - LOOK]) > 4 and abs(dd[e + LOOK]) > 4):
                    continue
                ev.append((i, j, int(c), "cross" if np.sign(dd[s - LOOK]) != np.sign(dd[min(W - 1, e + LOOK)]) else "kiss"))
    return ev


def paths(pi, pj, c, xs, swap):
    """red follows line i left of c, blue follows line j; right of c they swap if `swap`"""
    red = np.where(xs < c, pi[xs], (pj if swap else pi)[xs])
    blue = np.where(xs < c, pj[xs], (pi if swap else pj)[xs])
    return red, blue


def continuation_cost(y, xs, c):
    """two-sided smoothness: cubic fit across the touch (+-LOOK+10 px), mean abs residual. A swapped path has a kink."""
    w = (xs >= c - LOOK - 10) & (xs <= c + LOOK + 10)
    p = np.polyfit(xs[w] - c, y[w], 3)
    return float(np.mean(np.abs(np.polyval(p, xs[w] - c) - y[w])))


def render(img, pen, case, c0, r0, r1, swap_a):
    i, j, c, _ = case
    xs = np.arange(c0, c0 + 2 * HALF)
    raw = cv2.resize(cv2.cvtColor(img[r0:r1, c0:c0 + 2 * HALF], cv2.COLOR_GRAY2RGB), None, fx=ZOOM, fy=ZOOM,
                     interpolation=cv2.INTER_CUBIC)
    panels = [("RAW", raw)]
    for label, swap in [("A", swap_a), ("B", not swap_a)]:
        over = raw.copy()
        for y, col in zip(paths(pen[i], pen[j], c, xs, swap), [(230, 30, 30), (30, 60, 230)]):
            ok = ~np.isnan(y)
            pts = np.stack([(xs[ok] - c0 + 0.5) * ZOOM - 0.5, (y[ok] - r0 + 0.5) * ZOOM - 0.5], 1)
            cv2.polylines(over, [np.round(pts).astype(np.int32)], False, col, 2, cv2.LINE_AA)
        panels.append((label, cv2.addWeighted(over, 0.6, raw, 0.4, 0)))
    rows = []
    for label, p in panels:
        head = np.full((28, p.shape[1], 3), 255, np.uint8)
        cv2.putText(head, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        rows += [head, p, np.full((6, p.shape[1], 3), 255, np.uint8)]
    return np.vstack(rows)


pool = []
for d, g in [(d_all, 2.0), (d_act, 1.5), (d_act, 2.0)]:
    pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
    pool += [(d, g, e) for e in events(pen)]
cross = [p for p in pool if p[2][3] == "cross"]
kiss = [p for p in pool if p[2][3] == "kiss"]
rng.shuffle(cross); rng.shuffle(kiss)
k = min(N_CASES // 2, len(kiss))
cases = cross[:N_CASES - k] + kiss[:k]
print(f"pool: {len(cross)} crossings, {len(kiss)} kisses -> using {len(cases) - k} + {k}")

swaps = [n % 2 == 0 for n in range(len(cases))]      # exactly half the cases have the swapped panel as A
rng.shuffle(swaps)
jobs = []
for n, (d, g, case) in enumerate(cases):
    img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
    pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
    i, j, c, kind = case
    c0, xs = c - HALF, np.arange(c - HALF, c + HALF)
    ys = np.concatenate([pen[i][xs], pen[j][xs]])
    r0, r1 = max(0, int(np.nanmin(ys)) - PAD), min(img.shape[0], int(np.nanmax(ys)) + PAD)
    swap_a = swaps[n]                                 # True -> A is the swapped (wrong) hypothesis
    comp = render(img, pen, case, c0, r0, r1, swap_a)
    cv2.imwrite(str(out / f"case_{n:02d}.png"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
    cost = {s: sum(continuation_cost(y, xs, c) for y in paths(pen[i], pen[j], c, xs, s)) for s in (False, True)}
    jobs.append({"case": n, "src": f"{d.name}/g{g:g}", "lines": [i, j], "col": c, "kind": kind,
                 "correct": "B" if swap_a else "A", "rule_correct": cost[False] < cost[True],
                 "rule_margin_px": round(abs(cost[True] - cost[False]), 2), "_img": comp})


def ask(job):
    try:
        r = spark_ask([job["_img"]], Q)
        m = re.search(r"\{.*\}", r["text"], re.S)
        a = json.loads(m.group(0)) if m else {}
        return {**job, "spark": str(a.get("answer", "")).strip().upper()[:1], "conf": a.get("confidence"),
                "reason": a.get("reason"), "secs": round(r["secs"], 1)}
    except Exception as e:
        return {**job, "spark": "", "error": str(e)[:200]}


with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(ask, jobs))
for r in res:
    r.pop("_img")
    r["spark_ok"] = r["spark"] == r["correct"]
summary = {}
for kind in ("cross", "kiss", "all"):
    rr = [r for r in res if kind == "all" or r["kind"] == kind]
    if rr:
        summary[kind] = {"n": len(rr), "spark_acc": round(np.mean([r["spark_ok"] for r in rr]), 3),
                         "rule_acc": round(np.mean([r["rule_correct"] for r in rr]), 3),
                         "spark_unanswered": sum(r["spark"] not in ("A", "B") for r in rr)}
print(json.dumps(summary, indent=1))
(out / "spark_crossing_judge_results.json").write_text(json.dumps({"summary": summary, "cases": res}, indent=1, ensure_ascii=False))
