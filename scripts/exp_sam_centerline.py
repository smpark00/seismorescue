"""Apples-to-apples: SAM 3.1 'wavy line' instances -> per-column centreline -> same metrics as baselines.

usage: python -I exp_sam_centerline.py <scripts_dir> <synth_dir>
Also evaluates the classical tracker on the very same crop for comparison.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
from metaapi import sam_image

d = Path(sys.argv[2])
meta = json.loads((d / "meta.json").read_text())
S, MY = meta["spacing"], meta["margin_y"]
R0, R1 = int(MY + 1.5 * S), int(MY + 6.5 * S)
C0, C1 = 1900, 3100


def runs(col):
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    sp = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[sp + 1]], np.r_[idx[sp], idx[-1]]))


def tracker(ink, base):
    H, W = ink.shape
    y = np.full(W, np.nan)
    yp, v = float(base), 0.0
    for x in range(W):
        pred = yp + v
        c = [(a + b) / 2 for a, b in runs(ink[:, x]) if abs((a + b) / 2 - pred) <= S]
        if c:
            yc = min(c, key=lambda q: abs(q - pred)); v = 0.7 * v + 0.3 * (yc - yp); yp = yc; y[x] = yc
        else:
            yp = pred
    return y


def metrics(est, gt_pen):
    ok = ~np.isnan(est) & ~np.isnan(gt_pen)
    err = np.abs(est[ok] - gt_pen[ok])
    return {"coverage": round(ok.mean() * 100, 1), "mae_px": round(float(err.mean()), 2) if err.size else None,
            "within_1px_pct": round(float((err <= 1).mean() * 100), 1) if err.size else None,
            "wrong_trace_pct": round(float((err > S / 2).mean() * 100), 1) if err.size else None}


rows = []
for g in [0.8, 2.0, 5.0]:
    tag = f"g{g:g}"
    img = cv2.imread(str(d / f"heli_{tag}.png"), 0)[R0:R1, C0:C1]
    gt = np.load(d / f"gt_{tag}.npz")
    pen = gt["pen"][:, C0:C1] - R0
    inst = gt["inst"][:, R0:R1, C0:C1]
    keep = [i for i in range(len(inst)) if inst[i].sum() > 200]
    h, w = img.shape
    r = sam_image(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), "wavy line")
    masks = [o["mask"] for o in r["objects"] if o["mask"] is not None]
    # classical tracker on the same crop
    _, binv = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 0.8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binv > 0
    sam_m, trk_m = [], []
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pal = [(0, 0, 255), (0, 170, 0), (255, 0, 0), (0, 140, 255), (200, 0, 200), (0, 200, 200)]
    used = set()
    for k, i in enumerate(keep):
        gt_i = pen[i]
        # SAM: pick the instance whose centreline best matches this GT line (oracle matching = upper bound)
        best, best_est = None, np.full(w, np.nan)
        for mi, m in enumerate(masks):
            est = np.array([np.flatnonzero(m[:, x]).mean() if m[:, x].any() else np.nan for x in range(w)])
            ok = ~np.isnan(est) & ~np.isnan(gt_i)
            if ok.sum() < 50:
                continue
            e = np.nanmean(np.abs(est[ok] - gt_i[ok])) + (1 - ok.mean()) * S
            if best is None or e < best[0]:
                best, best_est = (e, mi), est
        if best:
            used.add(best[1])
            col = pal[k % len(pal)]
            mm = masks[best[1]]
            vis[mm] = (0.35 * vis[mm] + 0.65 * np.array(col)).astype(np.uint8)
        sam_m.append(metrics(best_est, gt_i))
        base = float(np.nanmedian(gt_i[:20])) if np.isfinite(np.nanmedian(gt_i[:20])) else (MY + i * S - R0)
        trk_m.append(metrics(tracker(ink, base), gt_i))
    cv2.imwrite(str(d / f"sam_wavyline_{tag}.png"), vis)

    def agg(ms, k):
        v = [m[k] for m in ms if m[k] is not None]
        return round(float(np.mean(v)), 1) if v else None
    for name, ms in [("SAM3.1 'wavy line'", sam_m), ("classical tracker", trk_m)]:
        rows.append({"gain": g, "method": name, "n_sam_instances": len(masks) if "SAM" in name else None,
                     **{k: agg(ms, k) for k in ["coverage", "mae_px", "within_1px_pct", "wrong_trace_pct"]}})
for r_ in rows:
    print(json.dumps(r_, ensure_ascii=False))
(d / "sam_centerline_results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
