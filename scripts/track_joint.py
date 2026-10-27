"""Spike S1 (E1 prototype): track all helicorder lines jointly with a smooth-motion prior.

usage: python -I track_joint.py <synth_dir> <synth_active_dir> <crossings_results.json>

Each column: predict every line from a quadratic fit to its last measured points; an ink run that contains two or more
predictions is a merge (crossing/kiss) and those lines coast on their prediction instead of taking the run centre;
other lines are matched to runs one-to-one (Hungarian) within a slope-dependent gate. Coasted points are not used
for fitting, so identity is carried through a merge by the curve's own momentum and curvature.
Evaluated on the same windows as exp_sam_crossings.py (and the whole image) against the greedy tracker, with the
same oracle start position (GT at the first column) for both.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

K_FIT, COAST_QUAD, MARGIN, DAMP = 8, 8, 2.0, 0.8
RUN_LEN = lambda s: abs(s) + 6            # longest ink run one stroke makes in a column at slope s (px)


def ink_mask(img, thr=22, min_area=20):
    """dark ink vs local paper level (closing removes strokes thinner than 15 px); drops speckle components,
    which a joint tracker would otherwise take as measurements while coasting"""
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    dark = cv2.GaussianBlur(bg.astype(np.float32) - img.astype(np.float32), (0, 0), 0.8)
    _, lab, stats, _ = cv2.connectedComponentsWithStats((dark > thr).astype(np.uint8), connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[lab]


def runs(col):
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    sp = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[sp + 1]], np.r_[idx[sp], idx[-1]]))


def track_joint(ink, y0, x0, x1, spacing):
    """y0: start row per line at column x0 -> (n_lines, W) estimates (NaN outside [x0, x1))"""
    H, W = ink.shape
    n = len(y0)
    est = np.full((n, W), np.nan)
    hx = [[x0 - 1] for _ in range(n)]
    hy = [[float(y)] for y in y0]
    coast = np.zeros(n, int)
    gap = np.zeros(n, bool)      # coasting because ink is missing (gap, time-mark step) rather than merged
    for x in range(x0, x1):
        pred, slope = np.empty(n), np.empty(n)
        for k in range(n):
            xs, ys = np.array(hx[k][-K_FIT:]), np.array(hy[k][-K_FIT:])
            deg = 2 if len(xs) >= 5 and coast[k] <= COAST_QUAD else (1 if len(xs) >= 2 else 0)
            xe = x
            if gap[k]:               # damp the motion in gaps; inside a merge keep full momentum to pass the crossing
                deg = min(deg, 1)
                xe = xs[-1] + DAMP * (1 - DAMP ** (x - xs[-1])) / (1 - DAMP)
            p = np.polyfit(xs - xe, ys, deg)
            pred[k] = np.clip(p[-1], 0, H - 1)
            slope[k] = p[-2] if deg >= 1 else 0.0
        rr = runs(ink[:, x])
        meas = pred.copy()
        measured = np.zeros(n, bool)
        merged = np.zeros(n, bool)
        taken = set()
        # merges: a run holding two or more predictions -> those lines coast inside the run
        for r, (a, b) in enumerate(rr):
            inside = [k for k in range(n) if a - MARGIN <= pred[k] <= b + MARGIN]
            if len(inside) >= 2:
                for k in inside:
                    meas[k] = np.clip(pred[k], a, b)
                    merged[k] = True
                taken.add(r)
        free_l = [k for k in range(n) if not any(a - MARGIN <= pred[k] <= b + MARGIN for r, (a, b) in enumerate(rr) if r in taken)]
        free_r = [r for r in range(len(rr)) if r not in taken]
        if free_l and free_r:
            cost = np.full((len(free_l), len(free_r)), 1e6)
            for i, k in enumerate(free_l):
                # tight gate on the run centre: a smooth trace moves ~1 px off its 1-column prediction, while time-mark
                # steps (~9 px) and speckle fall outside, so the line coasts through them; widens slowly to reacquire
                gate = 2.5 + 0.35 * abs(slope[k]) + 1.0 * min(coast[k], 20)
                for j, r in enumerate(free_r):
                    a, b = rr[r]
                    dist = abs((a + b) / 2 - pred[k])
                    if dist <= gate:
                        cost[i, j] = dist
            for i, j in zip(*linear_sum_assignment(cost)):
                if cost[i, j] >= 1e6:
                    continue
                k, (a, b) = free_l[i], rr[free_r[j]]
                taken.add(free_r[j])
                if b - a <= RUN_LEN(slope[k]):          # one line's worth of ink -> trust the centre
                    meas[k], measured[k] = (a + b) / 2, True
                else:                                   # too long for one line (other ink) -> stay inside it
                    meas[k] = np.clip(pred[k], a, b)
        # reacquire: a lost line takes the nearest free run within S/2 only if no other line predicts closer to it
        for k in range(n):
            if measured[k] or merged[k]:
                continue
            cands = [(abs((a + b) / 2 - pred[k]), r) for r, (a, b) in enumerate(rr) if r not in taken and abs((a + b) / 2 - pred[k]) <= spacing / 2]
            if not cands:
                continue
            dist, r = min(cands)
            a, b = rr[r]
            if all(abs((a + b) / 2 - pred[m]) > dist for m in range(n) if m != k) and b - a <= RUN_LEN(slope[k]):
                meas[k], measured[k] = (a + b) / 2, True
                taken.add(r)
        for k in range(n):
            est[k, x] = meas[k]
            if measured[k]:
                hx[k].append(x); hy[k].append(meas[k]); coast[k] = 0; gap[k] = False
            else:
                coast[k] += 1
                gap[k] = not merged[k]
    return est


def tracker_greedy(ink, base, x0, x1, S):
    """the greedy baseline from digitize_baseline.py (one line at a time)"""
    y = np.full(ink.shape[1], np.nan)
    yp, v = float(base), 0.0
    for x in range(x0, x1):
        pred = yp + v
        c = [(a + b) / 2 for a, b in runs(ink[:, x]) if abs((a + b) / 2 - pred) <= S]
        if c:
            yc = min(c, key=lambda q: abs(q - pred)); v = 0.7 * v + 0.3 * (yc - yp); yp = yc; y[x] = yc
        else:
            yp = pred
    return y


def touch_cols(pen, i):
    return np.min(np.nan_to_num(np.abs(np.delete(pen, i, 0) - pen[i]), nan=99), axis=0) < 3.0


def score(est, pen, lines, C0, C1, S):
    """same definitions as exp_sam_crossings.py: fail = missing or |err| > S/2; touch zone = +-40 cols"""
    f_all, f_zone = [], []
    for i in lines:
        gt, e = pen[i][C0:C1], est[i][C0:C1]
        zone = cv2.dilate(touch_cols(pen, i)[C0:C1].astype(np.uint8)[None], np.ones((1, 81), np.uint8))[0] > 0
        ok = ~np.isnan(gt)
        fail = ok & (np.isnan(e) | (np.abs(np.nan_to_num(e, nan=1e9) - gt) > S / 2))
        f_all.append(fail[ok].mean() * 100)
        if (zone & ok).any():
            f_zone.append(fail[zone & ok].mean() * 100)
    return round(float(np.mean(f_all)), 1), (round(float(np.mean(f_zone)), 1) if f_zone else None)


if __name__ == "__main__":
    d_all, d_act = Path(sys.argv[1]), Path(sys.argv[2])
    wins = {r["name"]: r for r in json.loads(Path(sys.argv[3]).read_text())}
    meta = json.loads((d_all / "meta.json").read_text())
    S, MX, MY, N = meta["spacing"], meta["margin_x"], meta["margin_y"], meta["n_lines"]
    rows = []
    for d, g, win in [(d_all, 0.8, None), (d_all, 2.0, "touch_g2"), (d_all, 5.0, "touch_g5"),
                      (d_act, 1.5, "active_g1.5"), (d_act, 2.0, "active_g2")]:
        img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
        pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
        W = img.shape[1]
        x0, x1 = MX, W - MX
        y0 = pen[:, x0]
        ink_new = ink_mask(img)
        _, binv = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 0.8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ests = {"greedy (Otsu ink)": np.array([tracker_greedy(binv > 0, y0[i], x0, x1, S) for i in range(N)]),
                "greedy (new ink)": np.array([tracker_greedy(ink_new, y0[i], x0, x1, S) for i in range(N)]),
                "joint (new ink)": track_joint(ink_new, y0, x0, x1, S)}
        for name, est in ests.items():
            row = {"src": d.name, "gain": g, "method": name, "whole_image_fail": score(est, pen, range(N), x0, x1, S)[0]}
            if win:
                R0, _, C0, C1 = wins[win]["crop_rows_cols"]
                i0 = int(round((R0 - MY) / S + 0.5))
                row["window"] = win
                row["window_fail"], row["window_touch_zone_fail"] = score(est, pen, range(i0, i0 + 5), C0, C1, S)
                if win.startswith("active"):
                    act = json.loads((d_act / "meta.json").read_text())["active_line"]
                    row["active_line_fail"] = score(est, pen, [act], C0, C1, S)[0]
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    (d_all / "track_joint_results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
