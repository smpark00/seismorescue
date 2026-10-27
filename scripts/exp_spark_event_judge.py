"""Stage 2 of docs/RESULTS.md-multiagent-design.md: Spark judges the events the learned judge is least sure about.

usage: python -I exp_spark_event_judge.py <scripts_dir> <bench_out_dir> <cache_dir> <judges_dir> [q] [workers]

Events and CV-held-out learned-judge probabilities come from cv_judges.py (event_predictions.json, repeat 0).
For conditions all-lines g2 and single-active g2, the q (default 0.15) least confident events per condition go to
Spark. Each event = one line, span [s, e) where proposer A and B disagree. Spark sees RAW plus two panels, each with
one candidate path of that line drawn over [s-60, e+60] (outside the span both candidates agree). It is asked twice,
with panel order and colour swapped (docs/RESULTS.md P3); only consistent answers are used, otherwise the event keeps the
learned judge's choice and is flagged. Answers stream to spark_event_answers.jsonl so an interrupted run resumes.
"""
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
from metaapi import spark_ask

root, cache, jdir = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
Q = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
WORKERS = int(sys.argv[6]) if len(sys.argv) > 6 else 40
CONDS = [("all", 2.0), ("active", 2.0)]
PAD, MAXW = 60, 1200
log = jdir / "spark_event_answers.jsonl"
QUESTION = ("These panels are cut from a scanned analog seismogram: several dark pen traces on paper, each a smooth "
            "continuous curve. RAW is the original. In panel 1 and panel 2 one coloured path is drawn over the image; "
            "both paths start on the same pen trace at the left edge and end on it at the right edge, but differ in the "
            "middle. Which panel's coloured path stays on ONE physical pen trace the whole way, without jumping onto a "
            "neighbouring trace or cutting across empty paper? Answer only JSON: "
            "{\"answer\": 1 or 2, \"confidence\": number 0-1, \"reason\": short string}.")

preds = json.loads((jdir / "event_predictions.json").read_text())
picked = []
for mode, g in CONDS:
    ev = sorted([p for p in preds if p["mode"] == mode and p["gain"] == g], key=lambda p: abs(p["p_A"] - 0.5))
    picked += ev[:int(Q * len(ev))]
done = {}
if log.exists():
    for line in log.read_text().splitlines():
        a = json.loads(line)
        done[(a["record"], a["mode"], a["gain"], a["line"], a["s"], a["swap"])] = a
print(f"events to judge: {len(picked)} ({Q:.0%} least confident per condition); answers on file: {len(done)}", flush=True)

_imgs = {}


def load(rec, mode, g):
    key = (rec, mode, g)
    if key not in _imgs:
        z = np.load(cache / f"{rec}_{mode}_g{g:g}.npz")
        _imgs[key] = (cv2.imread(str(root / rec / mode / f"heli_g{g:g}.png"), 0), z["A"], z["B"])
    return _imgs[key]


def compose(p, swap):
    """RAW + panel 1 + panel 2; swap=False: panel 1 = A in red, panel 2 = B in red; swap=True: panel 1 = B, blue"""
    img, A, B = load(p["record"], p["mode"], p["gain"])
    k, s, e = p["line"], p["s"], p["e"]
    c0, c1 = max(0, s - PAD), min(img.shape[1], e + PAD)
    xs = np.arange(c0, c1)
    # outside [s, e) both panels show B's path (what assembly keeps there), so only this event's choice differs
    yb = B[k, c0:c1]
    ya = yb.copy()
    ya[s - c0:e - c0] = A[k, s:e]
    r0 = max(0, int(np.nanmin(np.r_[ya, yb])) - 25)
    r1 = min(img.shape[0], int(np.nanmax(np.r_[ya, yb])) + 25)
    zoom = 2 if (c1 - c0) <= MAXW // 2 else 1
    raw = cv2.resize(cv2.cvtColor(img[r0:r1, c0:c1], cv2.COLOR_GRAY2RGB), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    if raw.shape[1] > MAXW:
        raw = cv2.resize(raw, (MAXW, int(raw.shape[0] * MAXW / raw.shape[1])), interpolation=cv2.INTER_AREA)
    fx, fy = raw.shape[1] / (c1 - c0), raw.shape[0] / (r1 - r0)
    colour = (30, 60, 230) if swap else (230, 30, 30)
    order = [yb, ya] if swap else [ya, yb]
    panels = [("RAW", raw)]
    for n, y in enumerate(order, 1):
        over = raw.copy()
        ok = ~np.isnan(y)
        pts = np.stack([(xs[ok] - c0 + 0.5) * fx - 0.5, (y[ok] - r0 + 0.5) * fy - 0.5], 1)
        cv2.polylines(over, [np.round(pts).astype(np.int32)], False, colour, 2, cv2.LINE_AA)
        panels.append((f"panel {n}", cv2.addWeighted(over, 0.65, raw, 0.35, 0)))
    rows = []
    for label, im in panels:
        head = np.full((28, im.shape[1], 3), 255, np.uint8)
        cv2.putText(head, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        rows += [head, im, np.full((6, im.shape[1], 3), 255, np.uint8)]
    return np.vstack(rows)


def ask(job):
    p, swap = job
    key = (p["record"], p["mode"], p["gain"], p["line"], p["s"], swap)
    if key in done:
        return done[key]
    comp = compose(p, swap)
    (jdir / "spark_event_imgs").mkdir(exist_ok=True)
    cv2.imwrite(str(jdir / "spark_event_imgs" / f"{p['record']}_{p['mode']}_l{p['line']}_s{p['s']}_{int(swap)}.jpg"),
                cv2.cvtColor(comp, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    for attempt in range(5):
        try:
            r = spark_ask([comp], QUESTION)
            m = re.search(r"\{.*\}", r["text"], re.S)
            a = json.loads(m.group(0)) if m else {}
            panel = int(a.get("answer")) if str(a.get("answer", "")).strip() in ("1", "2") else None
            pick = None if panel is None else (("A" if panel == 1 else "B") if not swap else ("B" if panel == 1 else "A"))
            out = {"record": p["record"], "mode": p["mode"], "gain": p["gain"], "line": p["line"], "s": p["s"],
                   "e": p["e"], "swap": swap, "pick": pick, "conf": a.get("confidence"), "secs": round(r["secs"], 1),
                   "out_tokens": (r["usage"] or {}).get("output_tokens")}
            with open(log, "a") as fh:
                fh.write(json.dumps(out) + "\n")
            return out
        except Exception as ex:
            time.sleep(5 * (attempt + 1))
            err = str(ex)[:200]
    return {"record": p["record"], "mode": p["mode"], "gain": p["gain"], "line": p["line"], "s": p["s"], "e": p["e"],
            "swap": swap, "pick": None, "error": err}


t0 = time.time()
jobs = [(p, sw) for p in picked for sw in (False, True)]
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    answers = list(ex.map(ask, jobs))
print(f"done {len(answers)} calls in {time.time() - t0:.0f} s; errors {sum('error' in a for a in answers)}", flush=True)
