"""SAM 3.1 where lines actually touch: SAM 'wavy line' vs classical tracker on crops chosen by GT touch count.

usage: python -I exp_sam_crossings.py <scripts_dir> <synth_dir> <synth_active_dir>

Why (10-27 review): the crop in exp_sam_centerline.py had only 2 touch events at gain 2, so its "0% wrong-trace"
said nothing about crossings. Here every crop is the 1200-px window with the most touch events (lines < 3 px apart),
plus the original crop for comparison. Metrics add a touch zone (+-40 columns around touches) and count a column as
failed when it is missing or off by > S/2. All SAM masks are saved so scoring can be redone offline.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import cv2
import numpy as np
from metaapi import sam_image

d_all, d_act = Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads((d_all / "meta.json").read_text())
S, MY, MX, N = meta["spacing"], meta["margin_y"], meta["margin_x"], meta["n_lines"]
CW, TOUCH_PX, ZONE = 1200, 3.0, 40
ACTIVE = json.loads((d_act / "meta.json").read_text())["active_line"]


def runs(col):
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    sp = np.flatnonzero(np.diff(idx) > 1)
    return list(zip(np.r_[idx[0], idx[sp + 1]], np.r_[idx[sp], idx[-1]]))


def tracker(ink, base):
    y = np.full(ink.shape[1], np.nan)
    yp, v = float(base), 0.0
    for x in range(ink.shape[1]):
        pred = yp + v
        c = [(a + b) / 2 for a, b in runs(ink[:, x]) if abs((a + b) / 2 - pred) <= S]
        if c:
            yc = min(c, key=lambda q: abs(q - pred)); v = 0.7 * v + 0.3 * (yc - yp); yp = yc; y[x] = yc
        else:
            yp = pred
    return y


def touch_cols(pen, i):
    """columns where line i is within TOUCH_PX of any other line"""
    return np.min(np.nan_to_num(np.abs(np.delete(pen, i, 0) - pen[i]), nan=99), axis=0) < TOUCH_PX


def best_window(pen, lines):
    """1200-px column window with the most touch events (rising edges) over the given lines"""
    t = np.array([touch_cols(pen, i) for i in lines])
    best = (-1, MX)
    for c0 in range(MX, pen.shape[1] - MX - CW, 50):
        ev = int(np.sum(np.diff(t[:, c0:c0 + CW].astype(int), axis=1) == 1))
        best = max(best, (ev, c0))
    return best[1]


def metrics(est, gt, zone):
    ok = ~np.isnan(gt)
    hit = ~np.isnan(est) & ok
    err = np.abs(est[hit] - gt[hit])
    fail = ok & (np.isnan(est) | (np.abs(np.nan_to_num(est, nan=1e9) - gt) > S / 2))
    return {"coverage": round(hit.sum() / max(1, ok.sum()) * 100, 1),
            "within_1px_pct": round(float((err <= 1).mean() * 100), 1) if err.size else None,
            "wrong_trace_pct": round(float((err > S / 2).mean() * 100), 1) if err.size else None,
            "fail_pct": round(float(fail[ok].mean() * 100), 1),
            "touch_zone_fail_pct": round(float(fail[zone & ok].mean() * 100), 1) if (zone & ok).any() else None}


def run(name, d, g, i0, c0):
    tag = f"g{g:g}"
    R0, R1, C0, C1 = int(MY + (i0 - 0.5) * S), int(MY + (i0 + 4.5) * S), c0, c0 + CW
    img = cv2.imread(str(d / f"heli_{tag}.png"), 0)[R0:R1, C0:C1]
    gt = np.load(d / f"gt_{tag}.npz")
    pen_full = gt["pen"]
    pen = pen_full[:, C0:C1] - R0
    inst = gt["inst"][:, R0:R1, C0:C1]
    lines = [i for i in range(i0, i0 + 5) if inst[i].sum() > 200]
    h, w = img.shape
    r = sam_image(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), "wavy line")
    masks = [o["mask"] for o in r["objects"] if o["mask"] is not None]
    np.savez_compressed(d / f"sam_masks_{name}.npz", masks=np.array(masks, bool).reshape(-1, h, w),
                        crop=np.array([R0, R1, C0, C1]))
    _, binv = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 0.8), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = binv > 0
    cents = [np.array([np.flatnonzero(m[:, x]).mean() if m[:, x].any() else np.nan for x in range(w)]) for m in masks]
    per, picks = [], []
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    pal = [(0, 0, 255), (0, 170, 0), (255, 0, 0), (0, 140, 255), (200, 0, 200)]
    for k, i in enumerate(lines):
        gt_i = pen[i]
        zone = cv2.dilate(touch_cols(pen_full, i)[C0:C1].astype(np.uint8)[None], np.ones((1, 2 * ZONE + 1), np.uint8))[0] > 0
        # oracle matching (upper bound): the SAM instance whose centreline best fits this GT line
        best, sam_est = None, np.full(w, np.nan)
        for mi, est in enumerate(cents):
            ok = ~np.isnan(est) & ~np.isnan(gt_i)
            if ok.sum() < 50:
                continue
            e = np.nanmean(np.abs(est[ok] - gt_i[ok])) + (1 - ok.mean()) * S
            if best is None or e < best[0]:
                best, sam_est = (e, mi), est
        picks.append(best[1] if best else None)
        if best:
            vis[masks[best[1]]] = (0.35 * vis[masks[best[1]]] + 0.65 * np.array(pal[k % 5])).astype(np.uint8)
        base = float(np.nanmedian(gt_i[:20]))
        per.append({"line": i, "touch_events": int(np.sum(np.diff(touch_cols(pen_full, i)[C0:C1].astype(int)) == 1)),
                    "gt_outside_crop_pct": round(float(np.mean((gt_i < 0) | (gt_i >= h)) * 100), 1),
                    "sam": metrics(sam_est, gt_i, zone), "classical": metrics(tracker(ink, base), gt_i, zone)})
    cv2.imwrite(str(d / f"sam_crossings_{name}.png"), vis)

    def agg(meth, k, rows):
        v = [p[meth][k] for p in rows if p[meth][k] is not None]
        return round(float(np.mean(v)), 1) if v else None
    row = {"name": name, "gain": g, "crop_rows_cols": [R0, R1, C0, C1], "n_sam_instances": len(masks),
           "shared_sam_instances": len([p for p in picks if p is not None]) - len({p for p in picks if p is not None}),
           "touch_events": sum(p["touch_events"] for p in per)}
    for meth in ("sam", "classical"):
        row[meth] = {k: agg(meth, k, per) for k in ("coverage", "within_1px_pct", "wrong_trace_pct", "fail_pct", "touch_zone_fail_pct")}
        if name.startswith("active"):
            row[meth + "_active_line"] = next(p[meth] for p in per if p["line"] == ACTIVE)
    row["per_line"] = per
    print(json.dumps({k: v for k, v in row.items() if k != "per_line"}, ensure_ascii=False))
    return row


rows = []
for g in [0.8, 2.0, 5.0]:                                  # original crop, now with corrected GT
    rows.append(run(f"orig_g{g:g}", d_all, g, 2, 1900))
for g in [2.0, 5.0]:                                       # all-lines mode, crop with most touches
    pen = np.load(d_all / f"gt_g{g:g}.npz")["pen"]
    i0 = max(range(N - 4), key=lambda i: sum(touch_cols(pen, j).sum() for j in range(i, i + 5)))
    rows.append(run(f"touch_g{g:g}", d_all, g, i0, best_window(pen, range(i0, i0 + 5))))
for g in [1.0, 1.5, 2.0]:                                  # single active line between quiet neighbours
    pen = np.load(d_act / f"gt_g{g:g}.npz")["pen"]
    i0 = ACTIVE - 2
    rows.append(run(f"active_g{g:g}", d_act, g, i0, best_window(pen, range(i0, i0 + 5))))
(d_all / "sam_crossings_results.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
