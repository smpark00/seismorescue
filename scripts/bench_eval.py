"""Benchmark harness: run offline digitizers on every rendered record and score them the same way.

usage: python -I bench_eval.py <bench_out_dir> <results.json>

<bench_out_dir>/<tag>/{all,active}/ come from render_helicorder.py on data/bench/*.mseed (fetch_benchmark.py).
Conditions: all-lines gains 2 and 5, single-active-line gains 1.5 and 2. Metrics (same definitions as
exp_sam_crossings.py / track_joint.py): fail = missing or |err| > S/2, over the whole image; touch-zone fail; and the
active line alone in "active" mode. Records carry a CV group = event date, so same-day events (e.g. 2011-03-11
aftershocks) never land in different folds.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from track_joint import ink_mask, score, track_joint, tracker_greedy

root, out = Path(sys.argv[1]), Path(sys.argv[2])
CONDS = [("all", 2.0), ("all", 5.0), ("active", 1.5), ("active", 2.0)]


def methods(img, y0, x0, x1, S):
    _, binv = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 0.8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return {"greedy": np.array([tracker_greedy(binv > 0, y, x0, x1, S) for y in y0]),
            "joint": track_joint(ink_mask(img), y0, x0, x1, S)}


rows = []
for rec in sorted(p for p in root.iterdir() if p.is_dir()):
    for mode, g in CONDS:
        d = rec / mode
        meta = json.loads((d / "meta.json").read_text())
        S, MX, N = meta["spacing"], meta["margin_x"], meta["n_lines"]
        img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        x0, x1 = MX, img.shape[1] - MX
        for name, est in methods(img, pen[:, x0], x0, x1, S).items():
            fail, zone = score(est, pen, range(N), x0, x1, S)
            row = {"record": rec.name, "group": rec.name[:8], "mode": mode, "gain": g, "method": name,
                   "fail": fail, "touch_zone_fail": zone}
            if mode == "active":
                row["active_line_fail"] = score(est, pen, [meta["active_line"]], x0, x1, S)[0]
            rows.append(row)
    print(rec.name, flush=True)
out.write_text(json.dumps(rows, indent=1))

# summary: mean and spread over records, per condition and method
for mode, g in CONDS:
    for name in ("greedy", "joint"):
        rr = [r for r in rows if r["mode"] == mode and r["gain"] == g and r["method"] == name]
        f = np.array([r["fail"] for r in rr])
        line = f"{mode:6s} g{g:<3} {name:7s} fail mean {f.mean():5.1f} (min {f.min():5.1f}, max {f.max():5.1f})"
        if mode == "active":
            a = np.array([r["active_line_fail"] for r in rr])
            line += f" | active line mean {a.mean():5.1f} (min {a.min():5.1f}, max {a.max():5.1f})"
        print(line)
