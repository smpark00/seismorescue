"""Render a real digital waveform as a synthetic WWSSN-style analog helicorder with ground truth.

usage: python -I render_helicorder.py <mseed> <outdir> [all|active|film] [gains]
film: geometry measured on the real WWSSN long-period film chip (docs/RESULTS.md): 1 h per line at ~1 px/s, 38 px line
spacing, 0.01-0.2 Hz band so the 6-8 s microseism shows on quiet lines; feed it an 8 h record (fetch_film.py) and
the quake then falls on one or two lines while the others carry the station's real background noise.

Physics/realism knobs that mirror real analog records:
- helical layout: 15 min per line, lines stacked; large waves cross neighbouring lines
- minute time marks: the pen/light spot is offset for 2 s every minute (longer at the hour)
- photographic exposure: fast-moving parts of the trace are fainter (less light per pixel)
- paper texture, speckle noise, blur
Ground truth saved per line: pen centreline y(x) (incl. time marks), signal-only y(x),
time-mark mask, and per-line instance masks.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from obspy import read
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

mseed, outdir = sys.argv[1], Path(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"  # all: every line shows its own 15 min; active: only the strongest line moves
FILM = MODE == "film"
outdir.mkdir(parents=True, exist_ok=True)

LINE_SEC = 3600 if FILM else 900     # helicorder line length
PX_PER_SEC = 1 if FILM else 4        # horizontal scale
SPACING = 38 if FILM else 48         # px between line baselines
MARGIN_X, MARGIN_Y = 40, 70
SS = 4                  # supersampling for anti-aliased drawing
PEN_W = 1.6             # pen width in output px
GAINS = [0.25, 0.8, 2.0, 5.0] if MODE == "all" else [1.0, 1.5, 2.0]   # peak deflection in units of line spacing
if len(sys.argv) > 4:                    # optional comma-separated gains, e.g. "1,2,4,8" (real big quakes reach 5-9)
    GAINS = [float(v) for v in sys.argv[4].split(",")]

st = read(mseed)
tr = st[0].copy()
tr.detrend("demean"); tr.taper(0.02)
tr.filter("bandpass", freqmin=0.01, freqmax=0.2 if FILM else 0.1, corners=2, zerophase=False)  # WWSSN-LP-like band
sig = tr.data.astype(float)
sig /= np.max(np.abs(sig))
fs = tr.stats.sampling_rate
n_lines = int(len(sig) // (LINE_SEC * fs))
t0 = tr.stats.starttime
# "active" mimics the real film chip: the quake sits on one line and its neighbours stay quiet (pre-P noise)
L = int(LINE_SEC * fs)
active = int(np.argmax([np.abs(sig[i * L:(i + 1) * L]).max() for i in range(n_lines)]))

# theoretical P arrival for later VLM evaluation: event/station from the sidecar json (fetch_benchmark.py),
# defaulting to Tohoku -> ANMO for data/tohoku_IU_ANMO_LHZ.mseed
side = Path(mseed).with_suffix(".json")
ev = json.loads(side.read_text()) if side.exists() else {
    "event_lat": 38.297, "event_lon": 142.373, "event_depth_km": 29.0, "sta_lat": 34.9459, "sta_lon": -106.4572,
    "event_time": "2011-03-11T05:46:24"}
dist = locations2degrees(ev["event_lat"], ev["event_lon"], ev["sta_lat"], ev["sta_lon"])
arr = TauPyModel("iasp91").get_travel_times(max(0.0, ev["event_depth_km"]), dist, phase_list=["P"])
p_time = t0.__class__(ev["event_time"]) + arr[0].time

W = MARGIN_X * 2 + LINE_SEC * PX_PER_SEC
H = MARGIN_Y * 2 + SPACING * (n_lines - 1)
rng = np.random.default_rng(7)

# time marks (offset up by 0.18*spacing for 2 s each minute; 6 s at the hour)
t_sec = np.arange(len(sig)) / fs
abs_sec = np.array([(t0 + s).timestamp for s in t_sec])
sec_of_min = np.mod(abs_sec, 60)
sec_of_hour = np.mod(abs_sec, 3600)
mark = (sec_of_min < 2) | (sec_of_hour < 6)
MARK_OFF = 0.18 * SPACING

paper_lowfreq = cv2.GaussianBlur(rng.normal(0, 1, (H // 8 + 1, W // 8 + 1)), (0, 0), 3)
paper_lowfreq = cv2.resize(paper_lowfreq, (W, H))
paper = 236 + 10 * paper_lowfreq / (np.abs(paper_lowfreq).max() + 1e-9)

meta = {"n_lines": n_lines, "line_sec": LINE_SEC, "px_per_sec": PX_PER_SEC, "spacing": SPACING,
        "margin_x": MARGIN_X, "margin_y": MARGIN_Y, "start_utc": str(t0), "fs": fs,
        "p_arrival_utc": str(p_time), "p_arrival_line": int((p_time - t0) // LINE_SEC) + 1,
        "dist_deg": dist, "gains": GAINS, "mode": MODE, "active_line": active}

line_ink = np.zeros((H * SS, W * SS), np.float32)
for g in GAINS:
    ink = np.zeros((H * SS, W * SS), np.float32)          # darkness accumulator
    inst = np.zeros((n_lines, H, W), np.uint8)             # per-line instance masks
    gt_pen, gt_sig, gt_mark = [], [], []
    for i in range(n_lines):
        a, b = int(i * LINE_SEC * fs), int((i + 1) * LINE_SEC * fs)
        s = sig[a:b] if MODE in ("all", "film") or i == active else sig[:b - a]
        mk = mark[a:b]
        base = MARGIN_Y + i * SPACING
        y_sig = base - g * SPACING * s
        y_pen = y_sig - MARK_OFF * mk
        x = MARGIN_X + np.arange(len(s)) / fs * PX_PER_SEC
        # densify to 1 point per output px for smooth drawing
        xd = np.arange(x[0], x[-1], 0.25)
        yd = np.interp(xd, x, y_pen)
        speed = np.hypot(np.gradient(xd), np.gradient(yd)) / 0.25          # px moved per px advance
        # photographic fading; film mode uses the physical law (exposure ~ 1/writing speed), which also turns the
        # fast minute-mark offsets into detached dots as on the real film
        intensity = np.clip(1.0 / speed, 0.1, 1.0) if FILM else np.clip(1.0 / (1.0 + 0.08 * (speed - 1)), 0.25, 1.0)
        # +(SS-1)/2 = pixel-centre offset so INTER_AREA downsampling puts the ink centre exactly on GT (x, y)
        pts = np.round(np.stack([xd, yd], 1) * SS + (SS - 1) / 2).astype(np.int32)
        # draw short segments with per-segment intensity; each line on its own layer so exposures add at crossings
        line_ink.fill(0)
        for k in range(0, len(pts) - 1, 4):
            seg = pts[k:k + 5]
            lvl = float(intensity[k])
            cv2.polylines(line_ink, [seg], False, lvl, thickness=max(1, int(PEN_W * SS)), lineType=cv2.LINE_AA)
        ink += line_ink
        m = np.zeros((H * SS, W * SS), np.uint8)
        cv2.polylines(m, [pts], False, 1, thickness=max(1, int(PEN_W * SS)), lineType=cv2.LINE_8)
        inst[i] = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA) > 0
        cols = np.arange(W)
        gt_pen.append(np.interp(cols, x, y_pen, left=np.nan, right=np.nan))
        gt_sig.append(np.interp(cols, x, y_sig, left=np.nan, right=np.nan))
        gt_mark.append(np.interp(cols, x, mk.astype(float), left=0, right=0) > 0.5)
    ink = cv2.resize(np.clip(ink, 0, 1), (W, H), interpolation=cv2.INTER_AREA)
    img = paper - 190 * ink
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    img += rng.normal(0, 5, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    tag = f"g{g:g}"
    cv2.imwrite(str(outdir / f"heli_{tag}.png"), img)
    np.savez_compressed(outdir / f"gt_{tag}.npz", pen=np.array(gt_pen), sig=np.array(gt_sig),
                        mark=np.array(gt_mark), inst=inst)
    # crossing statistics: columns where a line's pen comes within 1 px of a neighbour's pen
    pen = np.array(gt_pen)
    cross = np.nanmean(np.abs(pen[1:] - pen[:-1]) < 2.0) * 100
    print(f"{tag}: image {W}x{H}, lines={n_lines}, columns with neighbour contact={cross:.2f}%")

(Path(outdir) / "meta.json").write_text(json.dumps(meta, indent=1))
print(json.dumps(meta))
