"""Calibration scans for the sim-to-real work (research/31). KIP and MAT are NOT used here.

usage: python -I scripts/ml_sim2real_calib.py pairs [NAME ...] | layout [NAME ...]

pairs  : three WWSSN LPZ film chips with a co-located, unclipped HGLP digital record (CAL_ALQ73, CAL_KON73A, CAL_KON73B)
         run through real_paired_eval.main unchanged (layout, deskew, rescale to 38 px, A/B/judge/DP, independent DP
         time alignment, metrics) into out/ml/sim2real/calib_paired/.  They are calibration data, never test data.
layout : the same layout + rescale (+ the same photometric gain rule) for the unpaired calibration scans
         (WIN original, ALQ 1976 Guatemala, QUE/NUR 1975 Lice, ALQ 1971) into out/ml/sim2real/calib/layout/.
The calibration scans are local copies and are not distributed with this repository: only the settings below are.
To run this file, put your own copies at the paths below. Digital pairs: EarthScope FDSN dataselect/station services
(HG.ALQ / HG.KON LHZ, 25.5 h from ON - 30 min; HG.ALQ 1973-02-27 starts at 17:11 after a 235 s gap).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import real_paired_eval as rpe  # noqa: E402

CAL = ROOT / "data_local" / "sim2real_calib"
OUT = ROOT / "out" / "ml" / "sim2real"

PAIRED = {
    # label: STATION ALQ COMP Z MAG 3K, ON 2-27-73 1659, OFF 2-28-73 1612, CORR 0, To 15 Tg 100, GRND MOT up
    "CAL_ALQ73": {"image": "data_local/sim2real_calib/CAL_ALQ73.png", "negative": True,
                  "area": (790, 4685, 680, 14640), "px_per_s": 3.87,
                  "on": "1973-02-27T16:59:00", "off": "1973-02-28T16:12:00", "label_mag": 3000,
                  "label_note": "printed label read by eye: ALQ Z MAG 3K, ON 2-27-73 1659, OFF 2-28-73 1612, To 15 Tg 100",
                  "pair": "data_local/sim2real_calib/pairs/CAL_ALQ73__HG.ALQ",
                  "event": {"origin": "1973-02-28T06:37:49", "name": "Kuril Is. M7.2", "dist_deg": 67.7}},
    # label: KON Z MAG 1500, ON 2-5-73 2001 (CORR +210 ms), OFF 2-6-73 1845 (CORR -90 ms)
    "CAL_KON73A": {"image": "data_local/sim2real_calib/CAL_KON73A.png", "negative": True,
                   "area": (1450, 5215, 970, 14860), "px_per_s": 3.87,
                   "on": "1973-02-05T20:01:00", "off": "1973-02-06T18:45:00", "label_mag": 1500,
                   "label_note": "printed label read by eye: KON Z MAG 1500, ON 2-5-73 2001, OFF 2-6-73 1845, To 15 Tg 100",
                   "pair": "data_local/sim2real_calib/pairs/CAL_KON73A__HG.KON",
                   "event": {"origin": "1973-02-06T10:37:10", "name": "W China M7.4", "dist_deg": 63.7}},
    # label: KON Z MAG 1500, ON 6-23-73 16.12 (CORR -50 ms), OFF 6-24-73 13.21 (CORR +50 ms)
    "CAL_KON73B": {"image": "data_local/sim2real_calib/CAL_KON73B.png", "negative": True,
                   "area": (1410, 4780, 825, 14720), "px_per_s": 3.87,
                   "on": "1973-06-23T16:12:00", "off": "1973-06-24T13:21:00", "label_mag": 1500,
                   "label_note": "printed label read by eye: KON Z MAG 1500, ON 6-23-73 1612, OFF 6-24-73 1321, To 15 Tg 100",
                   "pair": "data_local/sim2real_calib/pairs/CAL_KON73B__HG.KON",
                   "event": {"origin": "1973-06-24T02:43:25", "name": "Shikotan M7.1", "dist_deg": 71.1}},
}

UNPAIRED = {
    "WIN74": {"image": "data_local/sim2real_calib/WIN74.png", "negative": True,
              "area": (900, 5050, 300, 14480), "px_per_s": 3.9, "res": "S3 3.9 px/s"},
    "ALQ76": {"image": "data_local/sim2real_calib/ALQ76.png", "negative": True,
              "area": (790, 4780, 1070, 14970), "px_per_s": 3.87, "res": "S3 3.9 px/s"},
    "QUE75": {"image": "data_local/sim2real_calib/QUE75.TIFF", "negative": False,
              "area": (860, 2960, 270, 8440), "px_per_s": 2.17, "res": "EarthScope 2.2 px/s"},
    "NUR75": {"image": "data_local/sim2real_calib/NUR75.TIFF", "negative": False,
              "area": (1255, 3420, 110, 8090), "px_per_s": 2.17, "res": "EarthScope 2.2 px/s"},
    "ALQ71": {"image": "data_local/sim2real_calib/ALQ71.TIFF", "negative": False,
              "area": (990, 3180, 215, 8410), "px_per_s": 2.17, "res": "EarthScope 2.2 px/s"},
}


_fit_orig = rpe.fit_time_model


def _fit_safe(P, lines, *a, **k):
    """real_paired_eval's even/odd-half check fails (All-NaN) when a scan has a single reference-active line; report
    the half as empty instead of crashing (the main alignment is unchanged)"""
    if len(lines) == 0:
        return {"r": float("nan"), "T0": float("nan"), "period": float("nan"), "pps": float("nan"),
                "pps_rel_nominal": float("nan"), "sign": 0, "lines": [], "note": "no lines in this half"}
    return _fit_orig(P, lines, *a, **k)


def run_pairs(names):
    out = OUT / "calib_paired"
    out.mkdir(parents=True, exist_ok=True)
    rpe.fit_time_model = _fit_safe
    for n in names:
        rpe.SCANS[n] = PAIRED[n]
        sys.argv = ["real_paired_eval.py", n, str(out)]
        rpe.main()


def layout_scan(cfg):
    """the layout part of real_paired_eval.main (unchanged logic): invert, crop, deskew, baselines, rescale to 38 px,
    darkness gain to the film renders' level when fainter"""
    img = cv2.imread(str(ROOT / cfg["image"]), 0)
    if cfg["negative"]:
        img = 255 - img
    r0, r1, c0, c1 = cfg["area"]
    sub = img[r0:r1, c0:c1]
    dark = np.clip(float(np.median(sub)) - sub.astype(np.float32), 0, None)
    S0 = rpe.window_spacing(dark)
    xs_w, tracks = rpe.track_line_field(dark, S0)
    field = rpe.fit_shift_field(xs_w, tracks)
    sub_d = rpe.deskew(sub, field)
    bases_full, S_full, filled = rpe.baselines(np.clip(float(np.median(sub_d)) - sub_d.astype(np.float32), 0, None), S0)
    f = rpe.TARGET_S / S_full
    S = rpe.TARGET_S
    pad = int(round(1.5 * S))
    img_s = cv2.copyMakeBorder(cv2.resize(sub_d, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), pad, pad, 0, 0, cv2.BORDER_REPLICATE)
    bases = bases_full * f + pad
    D_scan, D_film = rpe.trace_darkness(img_s, bases, S), rpe.film_render_darkness()
    gain = max(1.0, D_film / D_scan)
    if gain > 1.0:
        img_s = np.clip(245.0 - gain * np.clip(rpe.local_dark(img_s), 0, None), 0, 255).astype(np.uint8)
    W = img_s.shape[1]
    x0 = 3
    y0 = rpe.snap_start(img_s, bases, S, x0, int(round(120 * cfg["px_per_s"] * f)))
    info = {"spacing_full_px": S_full, "scale": f, "px_per_s_rescaled": cfg["px_per_s"] * f, "n_lines": len(bases),
            "filled": filled, "line_field_range_px": field["total_range_px"], "trace_darkness": D_scan,
            "render_darkness": D_film, "darkness_gain": gain, "rescaled_size": [W, img_s.shape[0]],
            "sec_per_spacing": S_full / cfg["px_per_s"]}
    return img_s, bases, y0, info


def run_layout(names):
    out = OUT / "calib" / "layout"
    out.mkdir(parents=True, exist_ok=True)
    allinfo = {}
    p = out / "layout_info.json"
    if p.exists():
        allinfo = json.loads(p.read_text())
    for n in names:
        img_s, bases, y0, info = layout_scan(UNPAIRED[n])
        cv2.imwrite(str(out / f"{n}_rescaled.png"), img_s)
        np.savez_compressed(out / f"{n}_layout.npz", bases=bases, y0=y0)
        allinfo[n] = {**info, "res": UNPAIRED[n]["res"]}
        print(n, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()}), flush=True)
    p.write_text(json.dumps(allinfo, indent=1))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "pairs":
        run_pairs(sys.argv[2:] or list(PAIRED))
    elif cmd == "layout":
        run_layout(sys.argv[2:] or list(UNPAIRED))
