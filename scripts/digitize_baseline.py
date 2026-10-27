"""Classical digitization baselines vs ground truth on the synthetic helicorders.

usage: python -I digitize_baseline.py <synth_dir>

M1 band-centroid : mean ink row inside each line's +-S/2 band (naive)
M2 tracker       : follow the ink run closest to a momentum prediction (DigitSeis/SKATE-like idea)
Metrics per line vs GT pen centreline: coverage, MAE, wrong-trace rate (|err| > S/2),
and correlation of the recovered signal with the true signal (time-mark columns excluded).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

d = Path(sys.argv[1])
meta = json.loads((d / "meta.json").read_text())
S, N, MX, MY = meta["spacing"], meta["n_lines"], meta["margin_x"], meta["margin_y"]


def runs(col):
    """contiguous True runs in a boolean column -> list of (start, end) inclusive."""
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[splits + 1]]
    ends = np.r_[idx[splits], idx[-1]]
    return list(zip(starts, ends))


def band_centroid(ink, base):
    H, W = ink.shape
    y = np.full(W, np.nan)
    lo, hi = int(base - S / 2), int(base + S / 2)
    for x in range(W):
        r = np.flatnonzero(ink[max(lo, 0):min(hi, H), x])
        if r.size:
            y[x] = r.mean() + max(lo, 0)
    return y


def tracker(ink, base, x0, x1, R=None):
    H, W = ink.shape
    R = R or S
    y = np.full(W, np.nan)
    yp, v = float(base), 0.0
    for x in range(x0, x1):
        pred = yp + v
        cands = [(a + b) / 2 for a, b in runs(ink[:, x]) if abs((a + b) / 2 - pred) <= R]
        if cands:
            yc = min(cands, key=lambda c: abs(c - pred))
            v = 0.7 * v + 0.3 * (yc - yp)
            yp = yc
            y[x] = yc
        else:
            yp = pred
    return y


rows = []
for g in meta["gains"]:
    tag = f"g{g:g}"
    img = cv2.imread(str(d / f"heli_{tag}.png"), 0)
    gt = np.load(d / f"gt_{tag}.npz")
    pen, sig, mark = gt["pen"], gt["sig"], gt["mark"]
    blur = cv2.GaussianBlur(img, (0, 0), 0.8)
    _, binv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binv > 0
    H, W = ink.shape
    x0, x1 = MX, W - MX
    # ambiguity from GT: columns where any pair of lines is within 2 px or out of order
    order_bad = np.zeros(W, bool)
    for i in range(N - 1):
        dd = pen[i + 1] - pen[i]
        order_bad |= np.nan_to_num(dd, nan=99) < 2.0
    amb = order_bad[x0:x1].mean() * 100
    est_all = {"M1": [], "M2": []}
    for i in range(N):
        base = MY + i * S
        est_all["M1"].append(band_centroid(ink, base))
        est_all["M2"].append(tracker(ink, base, x0, x1))
    for meth, ests in est_all.items():
        cov, mae, wrong, corr = [], [], [], []
        for i in range(N):
            e, p, s_, m = ests[i], pen[i], sig[i], mark[i]
            cols = np.arange(x0, x1)
            ok = ~np.isnan(e[cols]) & ~np.isnan(p[cols])
            cov.append(ok.mean() * 100)
            err = np.abs(e[cols][ok] - p[cols][ok])
            mae.append(err.mean() if err.size else np.nan)
            wrong.append((err > S / 2).mean() * 100 if err.size else np.nan)
            use = ok & ~m[cols]
            base = MY + i * S
            a, b = base - e[cols][use], base - s_[cols][use]
            corr.append(np.corrcoef(a, b)[0, 1] if use.sum() > 10 and a.std() > 0 and b.std() > 0 else np.nan)
        rows.append(dict(gain=g, method=meth, ambiguous_cols=round(amb, 1),
                         coverage=round(float(np.nanmean(cov)), 1),
                         mae_px=round(float(np.nanmean(mae)), 2),
                         wrong_trace_pct=round(float(np.nanmean(wrong)), 1),
                         corr_mean=round(float(np.nanmean(corr)), 3),
                         corr_worst_line=round(float(np.nanmin(corr)), 3)))
    # overlay for visual inspection (tracker), lines 3-6, 500 s window
    if g in (2.0, 5.0):
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        colors = [(0, 0, 255), (0, 160, 0), (255, 0, 0), (0, 140, 255), (200, 0, 200), (0, 200, 200), (120, 60, 0), (60, 0, 120)]
        for i in range(N):
            e = est_all["M2"][i]
            pts = np.array([(x, e[x]) for x in range(x0, x1) if not np.isnan(e[x])], np.int32)
            if len(pts) > 1:
                cv2.polylines(vis, [pts], False, colors[i], 1, cv2.LINE_AA)
        crop = vis[MY + 2 * S - 40: MY + 6 * S + 40, 1900:3100]
        cv2.imwrite(str(d / f"overlay_tracker_{tag}.png"), crop)

hdr = list(rows[0].keys())
print(" | ".join(hdr))
for r in rows:
    print(" | ".join(str(r[k]) for k in hdr))
(d / "baseline_results.json").write_text(json.dumps(rows, indent=1))
