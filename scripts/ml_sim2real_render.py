"""Calibrated, randomised film renderer ("film2", research/31). Same geometry and GT format as render_helicorder.py film
mode (1 h per line, 1 px/s, 38 px spacing, margins 40/70, pen = pen centreline incl. time marks), with a physical
photographic recording model whose parameter ranges come from calibration scans (never KIP or MAT):

- instrument: WWSSN long-period response (To 15 s, Tg 100 s, critically damped approximation as in real_paired_eval)
  applied to the broadband LHZ velocity record (velocity-flat sensor assumed), instead of a 0.01-0.2 Hz band-pass.
- amplitude: quake-line peak P (line spacings) is a parameter (log-uniform for training; fixed gains for holdout4);
  all lines share one scale (one instrument magnification per record).
- exposure: the light spot (Gaussian, sigma_spot) is integrated over time along the path (exposure ~ time spent per
  unit area, so ~ 1 / writing speed), exposures of all lines add on the film, then the film/print response is a
  Hill curve in exposure D = E^g / (E^g + E50^g) (E = 1 for a stroke moving at the drum speed); it saturates slow
  strokes (thick, dark) and drops steep strokes below the noise.
- time marks: a fast (0.08-0.25 s) offset of the spot every minute (1.5-3 s; 4-8 s at the hour), 0.18-0.40 spacing up,
  so marks appear as detached dots like on the film chips.
- scan: Gaussian blur, spatially correlated grain, slow background, sparse specks and faint vertical scratches; the
  trace darkness is drawn over the measured range and real_paired_eval's darkness-gain rule is applied afterwards.

usage: python -I scripts/ml_sim2real_render.py one <mseed> <outdir> [P] [seed]
       python -I scripts/ml_sim2real_render.py set <mseed_dir> <out_root> train|holdout [n_variants] [seed] [jobs]
"""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

LINE_SEC, PPS, SPACING, MARGIN_X, MARGIN_Y, N_LINES = 3600, 1.0, 38, 40, 70, 8
SS = 4
FS_FINE = 8.0                      # path time resolution (samples/s) before arc-length densification
D_FILM = 146.0                     # real_paired_eval.film_render_darkness() (film-mode renders), for the gain rule
G_HOLD = (1, 2, 4, 6, 8)

# measured / documented ranges (research/31, out/ml/sim2real/calib/): see calib_summary.json
RANGES = {
    "P_train": (0.7, 24.0),        # quake-line peak, spacings (log-uniform); real: CAL_ALQ73 8.0 (next lines 5.9, 4.5);
                                   # the tail above the crop (+-10.5) gives neighbours of 4-8 spacings, as at MAG 1500-3000
    "s50": (1.2, 16.0),            # slope at which a straight stroke's density halves (log-uniform); CAL_ALQ73 fit 1.5-2,
                                   # QUE75/NUR75 (visual bracketing, thin visible flanks) ~8-16
    "gamma": (2.0, 3.5),           # film contrast (Hill exponent); CAL_ALQ73 fit 2.2-3
    "fwhm": (2.6, 4.4),            # slow-stroke FWHM target, film px (scans: 2.7-4.3); sets the spot sd given s50
    "blur": (0.3, 0.7),            # scan blur sd, film px
    "dark_raw": (20.0, 230.0),     # slow-stroke darkness before the gain rule (log-uniform); scans 19-194
    "noise_sd": (0.015, 0.055),    # grain sd / slow-stroke darkness (log-uniform); scans 0.018-0.054
    "noise_corr": (0.6, 1.3),      # grain correlation sd, film px; scans: lag-1/2/4 autocorrelation 0.53-0.72/0.27-0.40/0.06-0.23
    "micro_rms": (0.008, 0.05),    # 6-30 s background motion on quiet lines, spacings rms (log-uniform); scans 0.009-0.043
    "mark_off": (0.18, 0.40),      # minute-mark offset, spacings (up); scans: detached dots ~0.2-0.4 above the line
    "mark_len": (1.5, 3.0),        # s
    "hour_len": (4.0, 8.0),        # s
    "mark_rise": (0.08, 0.25),     # s (fast: the connecting strokes are invisible, as on the scans)
    "lamp_line": 0.15,             # per-line exposure factor exp(N(0, .15))
    "specks_per_mpx": (50, 600),
    "scratches_per_mpx": (0, 60),
}


def wwssn_velocity(x, fs=1.0):
    """broadband velocity record -> WWSSN LP trace shape: H_v(s) = s^2 / ((s + 2pi/15)^2 (s + 2pi/100)^2), causal"""
    x = np.asarray(x, float)
    x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
    n = len(x)
    tap = np.ones(n)
    m = int(0.02 * n)
    tap[:m] = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
    tap[-m:] = tap[:m][::-1]
    nf = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfftfreq(nf, 1 / fs)
    s = 2j * np.pi * f
    w1, w2 = 2 * np.pi / 15.0, 2 * np.pi / 100.0
    H = s ** 2 / ((s + w1) ** 2 * (s + w2) ** 2)
    return np.fft.irfft(np.fft.rfft(x * tap, nf) * H, nf)[:n]


def spot_for(fwhm, s50):
    """spot sd giving the target slow-stroke FWHM: FWHM ~ sigma * (2.47 + 1.563 ln s50), fitted on the CAL_ALQ73 grid
    renders (saturated slow strokes widen as s50 grows)"""
    return float(np.clip(fwhm / (2.47 + 1.563 * np.log(s50)), 0.45, 1.5))


def sample_params(rng, P=None):
    lu = lambda a, b: float(np.exp(rng.uniform(np.log(a), np.log(b))))
    R = RANGES
    s50, fwhm = lu(*R["s50"]), float(rng.uniform(*R["fwhm"]))
    return {"sigma_spot": spot_for(fwhm, s50), "fwhm_target": fwhm,
            "P": P if P is not None else lu(*R["P_train"]), "s50": s50, "gamma": float(rng.uniform(*R["gamma"])),
            "blur": float(rng.uniform(*R["blur"])),
            "dark_raw": lu(*R["dark_raw"]), "noise_sd": lu(*R["noise_sd"]), "noise_corr": float(rng.uniform(*R["noise_corr"])),
            "mark_off": float(rng.uniform(*R["mark_off"])) * (1 if rng.random() < 0.9 else -1),
            "mark_len": float(rng.uniform(*R["mark_len"])), "hour_len": float(rng.uniform(*R["hour_len"])),
            "mark_rise": float(rng.uniform(*R["mark_rise"])), "paper": float(rng.uniform(205, 245)),
            "specks": lu(*R["specks_per_mpx"]), "scratches": float(rng.uniform(*R["scratches_per_mpx"])),
            "micro_rms": lu(*R["micro_rms"]),
            "bg_lowfreq": float(rng.uniform(0.0, 0.04)), "seed": int(rng.integers(1 << 31))}


def smooth_mark(t, abs_t0, prm):
    """0..1 time-mark waveform on the fine time base (fast linear rise / fall)"""
    sec = np.mod(abs_t0 + t, 3600.0)
    smin = np.mod(sec, 60.0)
    dur = np.where(sec < 60.0, prm["hour_len"], prm["mark_len"])
    r = prm["mark_rise"]
    up = np.clip(smin / r, 0, 1)
    down = np.clip((dur - smin) / r, 0, 1)
    return np.where(smin < dur + r, np.minimum(up, down), 0.0)


def splat(canvas, xs, ys, w):
    """bilinear splat of weights w at film-px positions (xs, ys) into the SS-supersampled canvas"""
    H, W = canvas.shape
    X = xs * SS + (SS - 1) / 2
    Y = ys * SS + (SS - 1) / 2
    x0 = np.floor(X).astype(np.int64)
    y0 = np.floor(Y).astype(np.int64)
    fx, fy = X - x0, Y - y0
    flat = canvas.ravel()
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        flat += np.bincount((yi[ok] * W + xi[ok]), weights=(w * ww)[ok], minlength=H * W)


def path_points(x, y, t_step, step=0.25):
    """arc-length densified points along the polyline (x, y) sampled every t_step s; weight = time per point"""
    dx, dy = np.diff(x), np.diff(y)
    L = np.hypot(dx, dy)
    n = np.maximum(1, np.ceil(L / step).astype(np.int64))
    seg = np.repeat(np.arange(len(L)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    frac = (np.arange(n.sum()) - start + 0.5) / n[seg]
    return x[seg] + frac * dx[seg], y[seg] + frac * dy[seg], (t_step / n)[seg]


def expose(paths, H, W, prm, lamp=None):
    """paths: list of (x, y, t_step) fine polylines in film px (one per line). Returns film-scale density 0..1 and the
    density of a slow straight stroke (for normalising the darkness)."""
    canvas = np.zeros((H * SS, W * SS), np.float64)
    for i, (xp, yp, t_step) in enumerate(paths):
        px, py, pw = path_points(xp, yp, t_step)
        keep = (py > -3 * SPACING) & (py < H + 3 * SPACING)
        splat(canvas, px[keep], py[keep], pw[keep] * (1.0 if lamp is None else lamp[i]))
    sp = prm["sigma_spot"] * SS
    E = cv2.GaussianBlur(canvas.astype(np.float32), (0, 0), sp) * (SS * SS * np.sqrt(2 * np.pi) * prm["sigma_spot"] * PPS)
    e50 = 1.0 / np.sqrt(1 + prm["s50"] ** 2)
    g = prm["gamma"]
    Eg = np.power(np.clip(E, 0, None), g)
    Dn = cv2.resize(Eg / (Eg + e50 ** g), (W, H), interpolation=cv2.INTER_AREA)
    if prm["blur"] > 0:
        Dn = cv2.GaussianBlur(Dn, (0, 0), prm["blur"])
    return Dn, 1.0 / (1.0 + e50 ** g)


def finish(Dn, d_slow_density, prm, rng, bases):
    """density -> scanned uint8 image: darkness scale, specks, scratches, correlated grain, slow background, then
    real_paired_eval's darkness-gain rule"""
    H, W = Dn.shape
    dark = (prm["dark_raw"] / d_slow_density) * Dn
    n_sp = rng.poisson(prm["specks"] * H * W / 1e6)
    if n_sp:
        sx, sy = rng.integers(0, W, n_sp), rng.integers(0, H, n_sp)
        spk = np.zeros((H, W), np.float32)
        spk[sy, sx] = rng.uniform(0.2, 1.0, n_sp) * prm["dark_raw"]
        dark += cv2.GaussianBlur(spk, (0, 0), rng.uniform(0.5, 1.0)) * 3.0
    n_sc = rng.poisson(prm["scratches"] * H * W / 1e6)
    for _ in range(n_sc):
        x = int(rng.integers(0, W)); y0 = int(rng.integers(0, H)); ln = int(rng.uniform(0.3, 3.0) * SPACING)
        dark[y0:y0 + ln, x] += rng.uniform(0.1, 0.4) * prm["dark_raw"]
    nz = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), prm["noise_corr"])
    nz *= prm["noise_sd"] * prm["dark_raw"] / (nz.std() + 1e-9)
    low = cv2.resize(rng.normal(0, 1, (H // 40 + 2, W // 40 + 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    img = np.clip(prm["paper"] * (1 + prm["bg_lowfreq"] * low) - dark + nz, 0, 255).astype(np.uint8)
    import real_paired_eval as rpe
    D_scan = rpe.trace_darkness(img, bases, SPACING)
    gain = max(1.0, D_FILM / max(D_scan, 1e-3))
    if gain > 1.0:
        img = np.clip(245.0 - gain * np.clip(rpe.local_dark(img), 0, None), 0, 255).astype(np.uint8)
    return img, D_scan, gain


def render(y_rec, abs_t0, prm, n_lines=N_LINES, noise_rec=None):
    """y_rec: WWSSN-filtered 1-sps record (>= n_lines h); noise_rec: optional extra station noise (same units, already
    scaled). Returns uint8 image, gt dict, info."""
    rng = np.random.default_rng(prm["seed"])
    L = int(LINE_SEC)
    y_rec = y_rec[:n_lines * L]
    if len(y_rec) < n_lines * L:                     # a few samples short (fetch_film accepts >= 99 %): pad the end
        y_rec = np.pad(y_rec, (0, n_lines * L - len(y_rec)), mode="edge")
    sig = y_rec / np.max(np.abs(y_rec))
    if noise_rec is not None:
        sig = sig + noise_rec[:len(sig)]
    W = MARGIN_X * 2 + int(LINE_SEC * PPS)
    H = MARGIN_Y * 2 + SPACING * (n_lines - 1)
    from scipy.signal import resample
    n_f = int(len(sig) * FS_FINE)
    sig_f = resample(sig, n_f)                       # band-limited trace -> 8 samples/s
    t_f = np.arange(n_f) / FS_FINE
    mk_f = smooth_mark(t_f, abs_t0, prm)
    gt_pen, gt_sig, gt_mark, paths = [], [], [], []
    cols = np.arange(W)
    amp = prm["P"] * SPACING
    lamp = np.exp(rng.normal(0, RANGES["lamp_line"], n_lines))
    for i in range(n_lines):
        a, b = int(i * L * FS_FINE), int((i + 1) * L * FS_FINE)
        base = MARGIN_Y + i * SPACING
        ys = base - amp * sig_f[a:b]
        yp = ys - prm["mark_off"] * SPACING * mk_f[a:b]
        xp = MARGIN_X + t_f[:b - a] * PPS
        paths.append((xp, yp, 1.0 / FS_FINE))
        gt_pen.append(np.interp(cols, xp, yp, left=np.nan, right=np.nan))
        gt_sig.append(np.interp(cols, xp, ys, left=np.nan, right=np.nan))
        gt_mark.append(np.interp(cols, xp, mk_f[a:b], left=0, right=0) > 0.5)
    Dn, dsd = expose(paths, H, W, prm, lamp)
    bases = MARGIN_Y + SPACING * np.arange(n_lines)
    img, D_scan, gain = finish(Dn, dsd, prm, rng, bases)
    gt = {"pen": np.array(gt_pen, np.float32), "sig": np.array(gt_sig, np.float32), "mark": np.array(gt_mark)}
    act = int(np.argmax([np.abs(sig[j * L:(j + 1) * L]).max() for j in range(n_lines)]))
    return img, gt, {"active_line": act, "trace_darkness": D_scan, "gain_rule": gain}


def hp_rms(v):
    """rms after removing a 31-sample running median (the 6-30 s band seen on quiet lines; same as the scans' measure)"""
    from scipy.ndimage import median_filter
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean((v - median_filter(v, 31)) ** 2)))


def micro_template(y, n, rng):
    """stationary noise with the amplitude spectrum of the record's first 3000 s (pre-event: P is >= 1 h after the
    start) above 1/60 Hz, random phases; unit hp_rms"""
    seg = y[:3000] - y[:3000].mean()
    A = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
    f = np.fft.rfftfreq(n, 1.0)
    A[f < 1 / 60.0] = 0
    z = np.fft.irfft(A * np.exp(2j * np.pi * rng.random(len(A))), n)
    return z / (hp_rms(z[:20000]) + 1e-12)


def load_record(mseed):
    from obspy import read
    st = read(str(mseed))
    st.merge(fill_value="interpolate")
    tr = st[0]
    side = Path(mseed).with_suffix(".json")
    info = json.loads(side.read_text()) if side.exists() else {}
    return tr.data.astype(float), tr.stats.starttime, tr.stats.sampling_rate, info


def render_record(mseed, outdir, tags_params):
    """tags_params: list of (tag, prm); writes heli_<tag>.png, gt_<tag>.npz, meta.json (params per tag)"""
    x, t0, fs, info = load_record(mseed)
    assert abs(fs - 1.0) < 1e-6
    y = wwssn_velocity(x)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {"n_lines": N_LINES, "line_sec": LINE_SEC, "px_per_sec": PPS, "spacing": SPACING, "margin_x": MARGIN_X,
            "margin_y": MARGIN_Y, "start_utc": str(t0), "fs": fs, "mode": "film2", "renderer": "ml_sim2real_render.py",
            "event": {k: info.get(k) for k in ("event_time", "mag", "dist_deg", "net", "sta")}, "variants": {}}
    abs_t0 = float(t0.timestamp)
    act = None
    sig_all = y / np.max(np.abs(y[:N_LINES * LINE_SEC]))
    r_pre = hp_rms(sig_all[:3000])                       # quiet-line background motion of the record (peak = 1 units)
    for tag, prm in tags_params:
        rng = np.random.default_rng(prm["seed"] + 1)
        need = max(prm["micro_rms"] ** 2 - (prm["P"] * r_pre) ** 2, 0.0) ** 0.5 / prm["P"]
        extra = micro_template(y, max(len(y), N_LINES * LINE_SEC), rng) * need if need > 0 else None
        prm = {**prm, "micro_rms_record": prm["P"] * r_pre, "micro_added": need * prm["P"]}
        img, gt, inf = render(y, abs_t0, prm, noise_rec=extra)
        cv2.imwrite(str(outdir / f"heli_{tag}.png"), img)
        np.savez_compressed(outdir / f"gt_{tag}.npz", **gt)
        meta["variants"][tag] = {**prm, **{k: float(v) for k, v in inf.items() if k != "active_line"}}
        act = inf["active_line"]
    meta["active_line"] = act
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    return outdir


def _job(args):
    mseed, outdir, tags_params = args
    try:
        render_record(mseed, outdir, tags_params)
        return str(outdir), None
    except Exception as e:  # report, do not stop the pool
        return str(outdir), repr(e)


def seed_of(name, salt):
    return int(hashlib.md5(f"{name}|{salt}".encode()).hexdigest(), 16) % (1 << 31)


def cmd_set(src, out_root, mode, n_var=6, seed=0, jobs=6):
    src, out_root = Path(src), Path(out_root)
    tasks = []
    for ms in sorted(src.glob("*.mseed")):
        rng = np.random.default_rng(seed_of(ms.stem, seed))
        if mode == "holdout":
            tp = [(f"g{g:g}", sample_params(rng, float(g))) for g in G_HOLD]
        else:
            tp = [(f"v{j}", sample_params(rng)) for j in range(n_var)]
        od = out_root / ms.stem / "film"
        if (od / "meta.json").exists():
            continue
        tasks.append((str(ms), str(od), tp))
    print(f"{len(tasks)} records to render -> {out_root}", flush=True)
    with ProcessPoolExecutor(jobs) as ex:
        for od, err in ex.map(_job, tasks):
            print(od, "ERROR " + err if err else "ok", flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    cmd = sys.argv[1]
    if cmd == "one":
        P = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0
        sd = int(sys.argv[5]) if len(sys.argv) > 5 else 0
        prm = sample_params(np.random.default_rng(sd), P)
        print(json.dumps(prm))
        render_record(sys.argv[2], sys.argv[3], [(f"g{P:g}", prm)])
    elif cmd == "set":
        cmd_set(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]) if len(sys.argv) > 5 else 6,
                int(sys.argv[6]) if len(sys.argv) > 6 else 0, int(sys.argv[7]) if len(sys.argv) > 7 else 6)
