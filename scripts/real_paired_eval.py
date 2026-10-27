"""First quantitative evaluation of the digitizer on real analog records with co-located digital ground truth
(docs/RESULTS.md).

usage: python -I scripts/real_paired_eval.py KIP|MAT [out_root=out/real_paired]

Reference: the co-located HGLP LHZ channel (1 sps). Instrument response removed to ground displacement (pre-filter
0.003-0.005-0.4-0.45 Hz, water level 60 dB), then the WWSSN long-period instrument (To 15 s, Tg 100 s) simulated with
the critically damped approximation H(s) = s^3 / ((s + 2pi/15)^2 (s + 2pi/100)^2), normalised to peak gain 1
(seismometer-galvanometer coupling and the real damping are ignored). SEED dip -90: positive = ground up.
Layout: invert a negative; track row-profile peaks across column windows to measure the line-position field (lines
drift ~1 spacing per line = drum helix, plus a bow/fan); shear it out; baselines = row-profile peaks (gaps of the
earthquake lines filled evenly); isotropic rescale to 38 px spacing (film-mode geometry, docs/RESULTS.md). If the scan's
traces are fainter than the film-mode renders, their darkness is scaled up to the renders' level (image statistics
only, never the reference).
Digitize every line with proposers A and B (arbiter.proposals) and the learned judge (fit_lr once on all film dev items).
Time: t(x, line k) = T0 + P k + (x - xa) / pps, line 0 = first record line (the label ON time lies on it).
  (1) As specified: P = 3600 s, (T0, pps, polarity) maximise Pearson r between the digitized quake line (largest ink
      spread, image only) and the reference; pps within +-3% of nominal, T0 such that ON lies on line 0 (+-10 min).
      This fails on both scans (no method's quake line correlates), so it is reported, not used.
  (2) Used: an independent alignment from an offline DP trace (Viterbi max-darkness path per line at ~85 px spacing; an
      alignment tool, not an evaluated method) on the reference-active lines; P free in 3564-3636 s, both series
      band-passed to BAND (the HGLP response is >36 dB below its peak above 0.1 Hz, so the 6-8 s microseisms that
      dominate WWSSN LP records are not in the reference). Checked by even/odd line halves and per-line residual lags.
Metrics per line x method on the reference samples (1 s) inside the record's ON-OFF window: Pearson r in BAND (also
the unfiltered, detrended r as specified); positions low-passed at BAND[1], then after a least-squares amplitude fit
the fraction of samples with |error| < 0.1 spacing and > 0.5 spacing ("lost"). Two scalings: per line (d ~ a ref +
c0 + c1 t, as specified; a path that flattens can score well because a -> 0) and one global scale per scan (median
per-line slope of the DP tool on lines it follows, r >= 0.45; only offset/trend per line). DP and DP38 (the same DP on
the film-scale image) are reported as controls. The KIP scan is a local-only copy: every image stays under out/.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from obspy import UTCDateTime, read, read_inventory
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, fftconvolve, find_peaks, sosfiltfilt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from arbiter import assemble, event_table, proposals          # noqa: E402
from cv_judges import ctx_features, fit_lr, load_items         # noqa: E402
from film_dev_cv import FILM_CONDS                              # noqa: E402
from track_joint import ink_mask                                # noqa: E402

TARGET_S = 38.0                     # film-mode line spacing (px)
TAU = 6.0                           # event threshold used by the judge everywhere else
PRE_FILT = (0.003, 0.005, 0.4, 0.45)
LP_PAPER_MM_PER_S = 0.25            # WWSSN LP drum speed 15 mm/min
METHODS = ("A", "B", "judge")
BAND = (1 / 400.0, 0.08)            # comparison band (Hz); see bandpass()
DP_SPACING = 85.0                   # line spacing (px) at which the DP alignment tool works
SCANS = {
    # area = (row0, row1, col0, col1) of the trace field on the full-res scan, read off previews (excludes film
    # borders, the label box and the handwritten notes)
    "KIP": {"image": "data_local/real_scans/020175_1700_0053_04.png", "negative": True,
            "area": (960, 5030, 1150, 15050), "px_per_s": 3.85,
            # handwritten label (partly legible): "KIP Z 375 ... / 2-1-75 1700 -010 ... / 2-2-75 1706 -020 ... / 15 100"
            "on": "1975-02-01T17:00:00", "off": "1975-02-02T17:06:00", "label_mag": 375,
            "label_note": "handwritten label read by eye: MAG 375?, ON 2-1-75 1700, OFF 2-2-75 1706, To 15 Tg 100",
            "pair": "data_local/real_scans/pairs/020175_1700_0053_04__HG.KIP",
            "event": {"origin": "1975-02-02T08:43:39.1", "name": "Attu Ms7.6", "dist_deg": 38.4}},
    "MAT": {"image": "data/real_scans/MAT_73_12_28_0020_LHZ.TIFF", "negative": False,
            "area": (900, 3250, 60, 7930), "px_per_s": 2.17,
            "on": "1973-12-28T00:20:00", "off": "1973-12-29T00:13:00", "label_mag": 3000,
            "label_note": "printed label: MAG 3000, ON 12-28-73 0020, OFF 12-29-73 0013, To 15 Tg 100, 'GRND MOT. Up' arrow; "
                          "operator wrote '20' where the first (partial) line starts",
            "pair": "data_local/real_scans/pairs/MAT_73_12_28_0020_LHZ__HG.MAT",
            "event": {"origin": "1973-12-28T13:41:45.8", "name": "Vanuatu Ms7.5", "dist_deg": 57.6}},
}


# ---------------------------------------------------------------- reference
def wwssn_lp(f):
    s = 2j * np.pi * f
    w1, w2 = 2 * np.pi / 15.0, 2 * np.pi / 100.0
    return s ** 3 / ((s + w1) ** 2 * (s + w2) ** 2)


def reference(cfg, t_on):
    """digital LHZ -> ground displacement -> simulated WWSSN LP; times in s relative to the label ON time"""
    st = read(str(ROOT / (cfg["pair"] + ".mseed"))).select(channel="LHZ")
    inv = read_inventory(str(ROOT / (cfg["pair"] + ".stationxml")))
    assert len(st) == 1, f"expected one gap-free LHZ trace, got {st}"
    tr = st[0].copy()
    ch = inv.select(channel="LHZ", time=tr.stats.starttime)[0][0][0]
    tr.detrend("demean"); tr.detrend("linear"); tr.taper(max_percentage=0.005)
    tr.remove_response(inventory=inv, output="DISP", pre_filt=PRE_FILT, water_level=60)
    x = tr.data.astype(float)
    n = len(x)
    nf = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfftfreq(nf, tr.stats.delta)
    H = wwssn_lp(f)
    y = np.fft.irfft(np.fft.rfft(x, nf) * H / np.abs(H).max(), nf)[:n]
    t = tr.times() + (tr.stats.starttime - t_on)
    info = {"channel": tr.id, "start": str(tr.stats.starttime), "npts": n, "sps": tr.stats.sampling_rate, "gaps": 0,
            "dip": ch.dip, "sensor": ch.sensor.description if ch.sensor else None,
            "pre_filt_hz": PRE_FILT, "water_level_db": 60,
            "wwssn_lp": "critically damped approx: zeros 0,0,0; poles -2pi/15 (x2), -2pi/100 (x2); peak gain normalised to 1",
            "wwssn_peak_period_s": float(1 / f[1:][np.argmax(np.abs(H[1:]))]),
            "polarity": "SEED dip %.0f -> positive = ground up" % ch.dip}
    return t, y, info


# ---------------------------------------------------------------- layout
def window_spacing(dark, win=600):
    lags = []
    for c in range(0, dark.shape[1] - win, dark.shape[1] // 8):
        p = dark[:, c:c + win].mean(1)
        a = p - p.mean()
        ac = np.correlate(a, a, "full")[len(a) - 1:]
        lags.append(30 + int(np.argmax(ac[30:min(len(ac), 400)])))
    return float(np.median(lags))


def track_line_field(dark, S0, win=400, step=200):
    """row-profile peaks per column window, linked across windows -> per-line row vs column"""
    sig = max(1.5, S0 / 60)
    cols = list(range(0, dark.shape[1] - win, step))
    peaks = []
    for c in cols:
        pr = gaussian_filter1d(dark[:, c:c + win].mean(1), sig)
        pk, _ = find_peaks(pr, distance=0.6 * S0, prominence=0.5)
        peaks.append(pk.astype(float))
    ref_i = int(np.argmin([np.std(np.diff(p)) if len(p) > 3 else 1e9 for p in peaks]))
    tracks = {j: {ref_i: y} for j, y in enumerate(peaks[ref_i])}
    for direction in (1, -1):
        last = {j: y for j, y in enumerate(peaks[ref_i])}
        i = ref_i + direction
        while 0 <= i < len(cols):
            if len(peaks[i]):
                for j in list(last):
                    d = np.abs(peaks[i] - last[j])
                    m = int(np.argmin(d))
                    if d[m] <= 0.15 * S0:
                        last[j] = peaks[i][m]
                        tracks[j][i] = last[j]
            i += direction
    return np.array(cols) + win / 2, tracks


def fit_shift_field(xs, tracks, deg=3):
    """dev(x, y) = a(x) + b(x) (ybar - ymid), a and b cubic in x; robust (two passes, 3 MAD rejection)"""
    rows = []
    for t in tracks.values():
        if len(t) < 0.8 * len(xs):
            continue
        ybar = np.mean(list(t.values()))
        rows += [(xs[i], ybar, y - ybar) for i, y in t.items()]
    R = np.array(rows)
    ymid = float(np.median(R[:, 1]))
    xm, xh = float(xs.mean()), float(np.ptp(xs) / 2)
    V = np.vander((R[:, 0] - xm) / xh, deg + 1)
    M = np.c_[V, V * (R[:, 1] - ymid)[:, None]]
    keep = np.ones(len(R), bool)
    for _ in range(2):
        coef = np.linalg.lstsq(M[keep], R[keep, 2], rcond=None)[0]
        res = R[:, 2] - M @ coef
        keep = np.abs(res) < 3 * 1.4826 * np.median(np.abs(res[keep])) + 1e-6
    return {"coef": coef.tolist(), "deg": deg, "x_mean": xm, "x_half": xh, "ymid": ymid,
            "n_lines_used": int(len({r[1] for r in rows})), "resid_sd_px": float(res[keep].std()),
            "total_range_px": float(np.ptp(M @ coef))}


def deskew(sub, field):
    H, W = sub.shape
    xx = np.arange(W, dtype=float)
    V = np.vander((xx - field["x_mean"]) / field["x_half"], field["deg"] + 1)
    c = np.array(field["coef"])
    a = (V @ c[:field["deg"] + 1]).astype(np.float32)          # shift at the mid row
    b = (V @ c[field["deg"] + 1:]).astype(np.float32)          # extra shift per row away from it
    map_x, map_y = np.meshgrid(xx.astype(np.float32), np.arange(H, dtype=np.float32))
    map_y = map_y + a[None, :] + b[None, :] * (map_y - field["ymid"])
    return cv2.remap(sub, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def baselines(dark_d, S0):
    pr = gaussian_filter1d(dark_d.mean(1), max(1.5, S0 / 60))
    pk, _ = find_peaks(pr, distance=0.6 * S0, prominence=0.3)
    pk = pk.astype(float)
    gaps = np.diff(pk)
    S = float(np.median(gaps))
    out, filled = [pk[0]], []
    for a, g in zip(pk[:-1], gaps):
        n_in = int(round(g / S)) - 1
        for m in range(1, n_in + 1):                    # earthquake lines have no sharp profile peak
            out.append(a + g * m / (n_in + 1)); filled.append(len(out) - 1)
        out.append(a + g)
    return np.array(out), S, filled


def local_dark(img):
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    return bg.astype(np.float32) - img.astype(np.float32)


def trace_darkness(img, bases, S):
    """median over lines of the median per-column peak darkness within +-S/3 of the baseline"""
    dk = local_dark(img)
    vals = []
    for b in bases:
        lo, hi = int(round(b - S / 3)), int(round(b + S / 3))
        if lo < 0 or hi > img.shape[0]:
            continue
        vals.append(np.median(dk[lo:hi, 5:-5].max(0)))
    return float(np.median(vals))


def film_render_darkness():
    vals = []
    for rec in sorted((ROOT / "out" / "film_dev").iterdir())[:6]:
        meta = json.loads((rec / "film" / "meta.json").read_text())
        img = cv2.imread(str(rec / "film" / "heli_g1.png"), 0)
        vals.append(trace_darkness(img, meta["margin_y"] + meta["spacing"] * np.arange(meta["n_lines"]), meta["spacing"]))
    return float(np.median(vals))


def trace_coverage(ink, b, S):
    """fraction of columns with a thin ink run (<= S/2) centred within S/4 of the baseline (a trace, not a band)"""
    band = ink[int(max(0, b - S)):int(min(ink.shape[0], b + S))]
    off = int(max(0, b - S))
    hit = 0
    for col in band.T:
        idx = np.flatnonzero(col)
        if idx.size == 0:
            continue
        sp = np.flatnonzero(np.diff(idx) > 1)
        starts, ends = np.r_[idx[0], idx[sp + 1]], np.r_[idx[sp], idx[-1]]
        cen = (starts + ends) / 2 + off
        hit += bool(np.any((ends - starts + 1 <= S / 2) & (np.abs(cen - b) <= S / 4)))
    return hit / ink.shape[1]


def snap_start(img_s, bases, S, x0, ncols):
    """baselines at the start column: nearest darkness-profile peak over the first columns, within S/4"""
    pr = gaussian_filter1d(np.clip(local_dark(img_s)[:, x0:x0 + ncols], 0, None).mean(1), 1.0)
    pk, _ = find_peaks(pr, distance=0.5 * S, prominence=0.5)
    y0 = bases.copy()
    for k, b in enumerate(bases):
        if len(pk):
            j = int(np.argmin(np.abs(pk - b)))
            if abs(pk[j] - b) <= S / 4:
                y0[k] = float(pk[j])
    return y0




def dp_trace(D, b, S, lam=0.005, dmax=10):
    """offline alignment tool, not an evaluated method: max-darkness path within +-S/2 of the baseline (Viterbi)"""
    H, W = D.shape
    lo, hi = int(max(0, b - S / 2)), int(min(H, b + S / 2))
    sc = D[lo:hi]
    n = hi - lo
    acc = sc[:, 0].copy()
    back = np.zeros((n, W), np.int16)
    ii = np.arange(n)
    for x in range(1, W):
        best = np.full(n, -1e9)
        arg = np.zeros(n, np.int16)
        for d in range(-dmax, dmax + 1):
            src = ii - d
            ok = (src >= 0) & (src < n)
            cand = np.full(n, -1e9)
            cand[ok] = acc[src[ok]] - lam * abs(d)
            upd = cand > best
            best[upd], arg[upd] = cand[upd], src[upd]
        acc = best + sc[:, x]
        back[:, x] = arg
    y = np.zeros(W, int)
    y[-1] = int(np.argmax(acc))
    for x in range(W - 1, 0, -1):
        y[x - 1] = back[y[x], x]
    return (y + lo).astype(float)


# ---------------------------------------------------------------- time alignment
def hp(x, fc=1 / 400.0):
    return sosfiltfilt(butter(2, fc, "highpass", fs=1.0, output="sos"), x - np.mean(x))


def bandpass(x):
    """the band both instruments resolve: HGLP is >36 dB below its peak above 0.1 Hz (6-8 s microseisms are lost)"""
    return sosfiltfilt(butter(4, BAND, "bandpass", fs=1.0, output="sos"), x - np.mean(x))


def lowpass(x):
    return sosfiltfilt(butter(4, BAND[1], "lowpass", fs=1.0, output="sos"), x)


def detrend(v, t):
    X = np.c_[np.ones_like(t), t - t.mean()]
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def ncc(ref, d):
    """Pearson r of d against every window of ref (valid lags)"""
    d0 = d - d.mean()
    n = len(d0)
    num = fftconvolve(ref, d0[::-1], mode="valid")
    c1, c2 = np.r_[0, np.cumsum(ref)], np.r_[0, np.cumsum(ref * ref)]
    s1, s2 = c1[n:] - c1[:-n], c2[n:] - c2[:-n]
    return num / np.sqrt(np.clip(s2 - s1 * s1 / n, 1e-30, None) * (d0 * d0).sum())


def line_series(path_k, x0, x1, xa, pps, base):
    xs = np.arange(x0, x1, dtype=float)
    tau = (xs - xa) / pps
    tj = np.arange(np.ceil(tau[0]), np.floor(tau[-1]) + 1)
    return tj, -(np.interp(tj, tau, path_k[x0:x1]) - base)       # up on the scan = positive


def fit_time_model(P, lines, x0, x1, xa, bases, t_ref, ref_f, filt, pps_nom, t0_range, period=None, lo=None, hi=None):
    """t(x, k) = T0 + Pl k + (x - xa)/pps; grid over pps (+-3%), Pl (3564-3636 s, or fixed), T0 (1 s), polarity,
    maximising the mean per-line Pearson r of the filtered series; then a local fine search. Returns the fit and the
    runner-up (best solution > 30 s / > 4 s away in T0 / Pl, or of the other polarity)."""
    lo, hi = (x0 if lo is None else lo), (x1 if hi is None else hi)
    t0g = np.arange(np.floor(t0_range[0]), np.ceil(t0_range[1]) + 1, 1.0)
    pg = np.array([float(period)]) if period else np.arange(3564.0, 3637.0)
    cand = []
    for pps in pps_nom * np.linspace(0.97, 1.03, 61):
        curves = []
        for k in lines:
            tj, d = line_series(P[k], lo, hi, xa, pps, bases[k])
            c = ncc(ref_f, filt(d))
            curves.append((t_ref[:len(c)] - tj[0], c))
        for Pl in pg:
            acc, n = np.zeros(len(t0g)), np.zeros(len(t0g))
            for k, (Tl, c) in zip(lines, curves):
                v = np.interp(t0g + Pl * k, Tl, c, left=np.nan, right=np.nan)
                ok = ~np.isnan(v)
                acc[ok] += v[ok]; n[ok] += 1
            m = np.where(n > 0, acc / np.clip(n, 1, None), np.nan)
            for sg in (1, -1):
                i = int(np.nanargmax(sg * m))
                cand.append((float(sg * m[i]), float(t0g[i]), float(Pl), float(pps), sg, sg * m))
    cand.sort(key=lambda v: -v[0])
    r0, T0, Pl, pps, sg, curve = cand[0]
    runner = next(c for c in cand[1:] if c[4] != sg or abs(c[1] - T0) > 30 or abs(c[2] - Pl) > 4)
    i0 = int(np.flatnonzero(t0g == T0)[0])
    left, right = i0, i0
    while left > 0 and curve[left - 1] >= r0 / 2:
        left -= 1
    while right < len(curve) - 1 and curve[right + 1] >= r0 / 2:
        right += 1

    def score(T0_, Pl_, pps_):
        rs = []
        for k in lines:
            tj, d = line_series(P[k], lo, hi, xa, pps_, bases[k])
            rs.append(pearson(filt(d), np.interp(T0_ + Pl_ * k + tj, t_ref, ref_f)))
        return sg * float(np.mean(rs))
    best = (score(T0, Pl, pps), T0, Pl, pps)
    for pps_ in pps * (1 + np.linspace(-0.001, 0.001, 9)):
        for Pl_ in ([Pl] if period else Pl + np.linspace(-1, 1, 9)):
            for T0_ in T0 + np.linspace(-2, 2, 9):
                s = score(T0_, Pl_, pps_)
                if s > best[0]:
                    best = (s, T0_, Pl_, pps_)
    return {"r": round(best[0], 4), "T0": round(best[1], 2), "period": round(best[2], 2), "pps": best[3],
            "pps_rel_nominal": best[3] / pps_nom - 1, "sign": int(sg), "lines": list(lines),
            "T0_range": [float(t0g[0]), float(t0g[-1])], "T0_halfwidth_s": float((t0g[right] - t0g[left]) / 2),
            "at_range_edge": bool(i0 in (0, len(t0g) - 1)),
            "runner_up": {"r": round(runner[0], 4), "T0": runner[1], "period": runner[2],
                          "pps_rel_nominal": runner[3] / pps_nom - 1, "sign": runner[4]}}


# ---------------------------------------------------------------- metrics
def sample_path(P, k, x0, x1, xa, al, t_ref, t_lo, t_hi):
    """reference samples whose mapped column lies on line k inside [x0, x1) and inside the record window"""
    x = xa + (t_ref - al["T0"] - al["period"] * k) * al["pps"]
    idx = np.flatnonzero((x >= x0) & (x <= x1 - 1) & (t_ref >= t_lo) & (t_ref <= t_hi))
    y = np.interp(x[idx], np.arange(x0, x1, dtype=float), P[k, x0:x1]) if idx.size else None
    return idx, y, x


def line_metrics(d, rf, t, S, a_glob):
    """d: digitized deviation (px, up +, polarity applied); rf: reference (m) on the same 1-s samples"""
    db, rb = bandpass(d), bandpass(rf)
    dl, rl = lowpass(d), lowpass(rf)
    tt = t - t.mean()
    X = np.c_[rl, np.ones_like(tt), tt]
    e = dl - X @ np.linalg.lstsq(X, dl, rcond=None)[0]                       # per-line LS amplitude + offset + trend
    coef_a = np.linalg.lstsq(X, dl, rcond=None)[0][0]
    Xg = np.c_[np.ones_like(tt), tt]
    eg = dl - a_glob * rl - Xg @ np.linalg.lstsq(Xg, dl - a_glob * rl, rcond=None)[0]   # global amplitude
    return {"n": int(len(d)), "r_band": round(pearson(db, rb), 3), "r_raw_detrended": round(pearson(detrend(d, t), detrend(rf, t)), 3),
            "a_ls_over_global": round(float(coef_a / a_glob), 3),
            "in_0.1S": round(float(np.mean(np.abs(e) < 0.1 * S)), 3), "lost_0.5S": round(float(np.mean(np.abs(e) > 0.5 * S)), 3),
            "in_0.1S_global": round(float(np.mean(np.abs(eg) < 0.1 * S)), 3),
            "lost_0.5S_global": round(float(np.mean(np.abs(eg) > 0.5 * S)), 3),
            "rms_err_global_S": round(float(np.sqrt(np.mean(eg ** 2)) / S), 3)}


# ---------------------------------------------------------------- figures (local only)
def draw_overlays(out, img_s, paths, pred, kq, x0, x1, S, name):
    lines = [k for k in range(max(0, kq - 2), min(len(pred), kq + 3)) if not np.all(np.isnan(pred[k]))]
    if not lines:
        return
    ys = [np.nanmedian(pred[k]) for k in lines]
    r0, r1 = int(max(0, min(ys) - 1.5 * S)), int(min(img_s.shape[0], max(ys) + 1.5 * S))
    colors = {"A": (255, 120, 0), "B": (0, 0, 255), "judge": (200, 0, 200), "DP": (0, 140, 255), "DP38": (0, 120, 0)}
    for meth, P in paths.items():
        vis = cv2.cvtColor(img_s, cv2.COLOR_GRAY2BGR)
        for k in lines:
            for Q, col in ((pred, (0, 170, 0)), (P, colors[meth] if k == kq else tuple(int(c * 0.6) for c in colors[meth]))):
                pts = np.array([(x, Q[k, x]) for x in range(x0, x1) if not np.isnan(Q[k, x])], np.int32)
                if len(pts):
                    cv2.polylines(vis, [pts], False, col, 1, cv2.LINE_AA)
        for i, c0 in enumerate(range(x0, x1, 700)):
            tile = cv2.resize(vis[r0:r1, c0:c0 + 700], None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out / f"{name}_overlay_{meth}_{i}.png"), tile)


def plot_series(path, name, show, paths, bases, al, a_glob, S, t_on, xa, x0, x1, t_ref, ref, t_lo, t_hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(show), 1, figsize=(16, 3.0 * len(show)), squeeze=False)
    for ax, (k, label) in zip(axs[:, 0], show):
        idx, _, _ = sample_path(paths["B"], k, x0, x1, xa, al, t_ref, t_lo, t_hi)
        tk = t_ref[idx]
        ax.plot((tk - tk[0]) / 60, a_glob * bandpass(ref[idx]) / S, color="k", lw=1.4, label=f"reference x global scale ({BAND[0]:.4f}-{BAND[1]} Hz)")
        for m, col in (("A", "tab:blue"), ("B", "tab:red"), ("judge", "tab:purple"), ("DP", "tab:orange"), ("DP38", "tab:green")):
            _, y, _ = sample_path(paths[m], k, x0, x1, xa, al, t_ref, t_lo, t_hi)
            ax.plot((tk - tk[0]) / 60, bandpass(-(y - bases[k]) * al["sign"]) / S, color=col, lw=0.7, alpha=0.8,
                    label=m + {"DP": " (alignment tool, 2.2x res)", "DP38": " (control, same res)"}.get(m, ""))
        ax.set_title(f"{name} line {k} ({label}), first sample {str(t_on + tk[0])[:19]} UTC, same band")
        ax.set_xlabel("minutes"); ax.set_ylabel("line spacings"); ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main():
    name = sys.argv[1].upper()
    cfg = SCANS[name]
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "out" / "real_paired"
    figs = out / name
    figs.mkdir(parents=True, exist_ok=True)
    t_on, t_off = UTCDateTime(cfg["on"]), UTCDateTime(cfg["off"])
    clock = time.time()

    t_ref, ref, ref_info = reference(cfg, t_on)
    ref_hp, ref_bp = hp(ref), bandpass(ref)
    print(f"[{name}] reference {ref_info['channel']} {ref_info['npts']} samples; WWSSN peak period {ref_info['wwssn_peak_period_s']:.1f} s", flush=True)

    # ---- layout on the full-res scan
    img = cv2.imread(str(ROOT / cfg["image"]), 0)
    if cfg["negative"]:
        img = 255 - img
    r0, r1, c0, c1 = cfg["area"]
    sub = img[r0:r1, c0:c1]
    dark = np.clip(float(np.median(sub)) - sub.astype(np.float32), 0, None)
    S0 = window_spacing(dark)
    xs_w, tracks = track_line_field(dark, S0)
    field = fit_shift_field(xs_w, tracks)
    sub_d = deskew(sub, field)
    bases_full, S_full, filled = baselines(np.clip(float(np.median(sub_d)) - sub_d.astype(np.float32), 0, None), S0)
    f = TARGET_S / S_full
    S = TARGET_S
    pad = int(round(1.5 * S))
    img_s = cv2.copyMakeBorder(cv2.resize(sub_d, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), pad, pad, 0, 0, cv2.BORDER_REPLICATE)
    bases = bases_full * f + pad
    D_scan, D_film = trace_darkness(img_s, bases, S), film_render_darkness()
    gain = max(1.0, D_film / D_scan)              # photometric normalisation from image statistics only
    if gain > 1.0:
        img_s = np.clip(245.0 - gain * np.clip(local_dark(img_s), 0, None), 0, 255).astype(np.uint8)
    H, W = img_s.shape
    x0, x1 = 3, W - 3
    xa = float(x0)
    pps_nom = cfg["px_per_s"] * f
    n_expect = int(np.ceil((t_off - t_on) / 3600.0)) + 1
    ink0 = ink_mask(img_s)
    dropped = []
    while len(bases) > n_expect:                  # more baselines than ON-OFF allows: drop non-trace edge candidates
        cov = [trace_coverage(ink0, b, S) for b in (bases[0], bases[-1])]
        j = 0 if cov[0] < cov[1] else len(bases) - 1
        dropped.append({"row_rescaled": round(float(bases[j]), 1), "trace_coverage": round(min(cov), 3)})
        bases = np.delete(bases, j)
    bases_full = (bases - pad) / f
    y0 = snap_start(img_s, bases, S, x0, int(round(120 * pps_nom)))
    print(f"[{name}] spacing {S_full:.1f}px -> scale {f:.4f}; line-field range {field['total_range_px']:.0f}px resid sd {field['resid_sd_px']:.1f}px; "
          f"{len(bases)} lines (filled {filled}, dropped {dropped}); darkness {D_scan:.0f} vs renders {D_film:.0f} -> gain {gain:.2f}; "
          f"rescaled {W}x{H}, nominal pps {pps_nom:.4f}", flush=True)
    cv2.imwrite(str(figs / f"{name}_rescaled.png"), img_s)

    # ---- digitize (evaluated methods)
    A, B, ink = proposals(img_s, y0, x0, x1, S)
    dev = load_items(ROOT / "out" / "film_dev", ROOT / "out" / "cache_film_dev", FILM_CONDS)
    trn = [r for it in dev for r in it["rows"] if r["label"] >= 0]
    model = fit_lr(np.array([r["x"] for r in trn]), np.array([r["label"] for r in trn], float), np.array([r["stake"] for r in trn], float))
    rows_ev = event_table(A, B, ink, x0, x1, TAU)
    p = model(np.array([ctx_features({"A": A, "B": B, "S": S, "x0": x0, "x1": x1}, r) for r in rows_ev])) if rows_ev else np.array([])
    J = assemble(A, B, rows_ev, ["A" if v > 0.5 else "B" for v in p])

    # ---- alignment tool (not evaluated): DP trace of every line at ~85 px spacing, mapped to the rescaled frame
    ds = min(1.0, DP_SPACING / S_full)
    im_dp = cv2.resize(sub_d, None, fx=ds, fy=ds, interpolation=cv2.INTER_AREA)
    Ddp = np.clip(local_dark(im_dp) if ds < 1.0 else float(np.median(im_dp)) - im_dp.astype(np.float32), 0, None)
    Ddp = Ddp / (np.percentile(Ddp[Ddp > 0], 99) + 1e-6)
    DPs = np.full((len(bases), W), np.nan)
    xs_dp = np.arange(im_dp.shape[1]) * (f / ds)
    for k, bf in enumerate(bases_full):
        yk = dp_trace(Ddp, bf * ds, S_full * ds, dmax=max(4, int(round(5 * cfg["px_per_s"] * ds))))
        DPs[k] = np.interp(np.arange(W), xs_dp, yk * (f / ds) + pad)
    # control (not evaluated, not used for alignment): the same DP on the film-scale image the methods see
    D38 = np.clip(local_dark(img_s), 0, None)
    D38 = D38 / (np.percentile(D38[D38 > 0], 99) + 1e-6)
    DP38 = np.array([dp_trace(D38, b, S, dmax=max(3, int(round(5 * pps_nom)))) for b in bases])
    paths = {"A": A, "B": B, "judge": J, "DP": DPs, "DP38": DP38}
    print(f"[{name}] digitized ({time.time() - clock:.0f}s): judge trained on {len(trn)} events / {len(dev)} film dev items, "
          f"{len(rows_ev)} events here, {np.mean(p > 0.5) if len(p) else float('nan'):.2f} take A; DP at {ds:.3f} of full res", flush=True)
    np.savez_compressed(figs / f"{name}_paths.npz", A=A, B=B, judge=J, DP=DPs, DP38=DP38, bases=bases, y0=y0)

    # ---- lines
    inner = list(range(1, len(bases) - 1))        # the first and last lines are partial records
    spread = []
    for b in bases:
        band = ink[int(max(0, b - S / 2)):int(min(H, b + S / 2)), x0:x1]
        rr = np.nonzero(band)[0] + int(max(0, b - S / 2))
        spread.append(float(np.mean(np.abs(rr - b) > S / 8)) if rr.size else 0.0)
    kq = inner[int(np.argmax([spread[k] for k in inner]))]           # quake line from the image only
    line_s = (x1 - xa) / pps_nom
    rng_on0 = (-line_s * 1.03 - 600.0, 600.0)                        # label ON lies on line 0 (+-10 min)
    # reference-active inner lines (reference only): band-limited rms over a line window, median over the T0 range
    act = []
    for k in inner:
        v = [np.std(ref_bp[(t_ref >= T + 3600 * k) & (t_ref < T + 3600 * k + 3600)]) for T in np.arange(rng_on0[0], rng_on0[1], 120)]
        act.append(float(np.median(v)))
    act = np.array(act)
    active = [k for k, a in zip(inner, act) if a >= 2 * np.median(act)]

    # ---- (1) as specified: quake-line fit per method, 3600 s lines, full band (high-pass 400 s only)
    spec = {m: fit_time_model(paths[m], [kq], x0, x1, xa, bases, t_ref, ref_hp, hp, pps_nom, rng_on0, period=3600) for m in METHODS}
    for m, v in spec.items():
        print(f"[{name}] spec fit, quake line {kq}, {m}: r {v['r']:.3f} T0 {v['T0']:+.0f}s pps {v['pps_rel_nominal']:+.2%} sign {v['sign']} "
              f"(runner-up r {v['runner_up']['r']:.3f} at T0 {v['runner_up']['T0']:+.0f}s)", flush=True)
    spec_band = {m: fit_time_model(paths[m], [kq], x0, x1, xa, bases, t_ref, ref_bp, bandpass, pps_nom, rng_on0) for m in METHODS}
    # ---- (2) independent alignment from the DP tool on the reference-active lines, line period free
    al = fit_time_model(DPs, active, x0, x1, xa, bases, t_ref, ref_bp, bandpass, pps_nom, rng_on0)
    al3600 = fit_time_model(DPs, active, x0, x1, xa, bases, t_ref, ref_bp, bandpass, pps_nom, rng_on0, period=3600)
    half = {"lines_even": fit_time_model(DPs, active[0::2], x0, x1, xa, bases, t_ref, ref_bp, bandpass, pps_nom, rng_on0),
            "lines_odd": fit_time_model(DPs, active[1::2], x0, x1, xa, bases, t_ref, ref_bp, bandpass, pps_nom, rng_on0)}
    print(f"[{name}] DP alignment on active lines {active}: mean r {al['r']:.3f} T0 {al['T0']:+.1f}s period {al['period']:.2f}s "
          f"pps {al['pps_rel_nominal']:+.2%} sign {al['sign']} halfwidth {al['T0_halfwidth_s']:.0f}s; runner-up {al['runner_up']}", flush=True)
    print(f"[{name}]   with period fixed at 3600 s: mean r {al3600['r']:.3f} T0 {al3600['T0']:+.1f}s pps {al3600['pps_rel_nominal']:+.2%} sign {al3600['sign']}; "
          f"halves: even {half['lines_even']['T0']:+.1f}/{half['lines_even']['period']:.1f}/{half['lines_even']['pps_rel_nominal']:+.2%}, "
          f"odd {half['lines_odd']['T0']:+.1f}/{half['lines_odd']['period']:.1f}/{half['lines_odd']['pps_rel_nominal']:+.2%}", flush=True)
    # residual per-line lag at the chosen alignment (drift diagnostic), DP traces
    t_lo, t_hi = 0.0, float(t_off - t_on)
    resid = {}
    for k in inner:
        idx, y, _ = sample_path(DPs, k, x0, x1, xa, al, t_ref, t_lo, t_hi)
        if idx.size < 1200:
            continue
        d = bandpass(-(y - bases[k]) * al["sign"])
        rs = [pearson(d, np.interp(t_ref[idx] + L, t_ref, ref_bp)) for L in range(-60, 61)]
        resid[k] = {"lag_s": int(np.argmax(rs)) - 60, "r_at_lag": round(float(max(rs)), 3), "r_at_0": round(float(rs[60]), 3)}
    good_lag = [v["lag_s"] for v in resid.values() if v["r_at_lag"] >= 0.5]
    x_on = xa - al["T0"] * al["pps"]
    print(f"[{name}] residual lags (lines with DP r>=0.5): {good_lag}; label ON maps to line 0 col {x_on:.0f} (full-res col {c0 + x_on / f:.0f})", flush=True)

    # ---- global amplitude (px per m): median per-line LS slope of the DP tool on inner lines it follows (r >= 0.45);
    # lines with large swings are excluded implicitly because the DP clips them (low r) or biases them low
    slopes, used = [], []
    for k in inner:
        idx, y, _ = sample_path(DPs, k, x0, x1, xa, al, t_ref, t_lo, t_hi)
        if idx.size < 1200:
            continue
        d, rf = bandpass(-(y - bases[k]) * al["sign"]), bandpass(ref[idx])
        if pearson(d, rf) >= 0.45:
            slopes.append(float(d @ rf / (rf @ rf))); used.append(k)
    a_glob = float(np.median(slopes)) if slopes else float("nan")
    mag = a_glob * LP_PAPER_MM_PER_S / (al["pps"] * 1000.0)
    print(f"[{name}] global scale {a_glob:.4g} px/m from DP lines {used} -> implied magnification {mag:.0f} (label {cfg['label_mag']})", flush=True)

    # ---- per-line metrics at the independent alignment
    ev_origin = UTCDateTime(cfg["event"]["origin"]) - t_on
    win = (t_ref >= ev_origin) & (t_ref <= ev_origin + 5400)
    t_max = float(t_ref[win][np.argmax(np.abs(ref_bp[win]))])
    k_main = int(np.floor((t_max - al["T0"] - (x0 - xa) / al["pps"]) / al["period"]))
    lines, pred = [], np.full_like(B, np.nan)
    for k in range(len(bases)):
        idx, yB, xk = sample_path(B, k, x0, x1, xa, al, t_ref, t_lo, t_hi)
        if idx.size < 1200:
            continue
        tk, rf = t_ref[idx], ref[idx]
        rfb = bandpass(rf)
        row = {"line": k, "utc_first_sample": str(t_on + tk[0])[:19], "n_samples": int(idx.size),
               "ref_rms_S": round(float(np.std(rfb) * a_glob / S), 4), "ref_max_S": round(float(np.max(np.abs(rfb)) * a_glob / S), 3),
               "ink_spread": round(spread[k], 3), "active_for_alignment": k in active,
               "chance_r_ref_vs_ref_pm900": [round(pearson(rfb, bandpass(np.interp(tk + sh, t_ref, ref))), 3) for sh in (-900.0, 900.0)],
               "methods": {}}
        for m, P in paths.items():
            _, y, _ = sample_path(P, k, x0, x1, xa, al, t_ref, t_lo, t_hi)
            dk = -(y - bases[k]) * al["sign"]
            met = line_metrics(dk, rf, tk, S, a_glob)
            met["chance_r_vs_ref_pm900"] = [round(pearson(bandpass(dk), bandpass(np.interp(tk + sh, t_ref, ref))), 3) for sh in (-900.0, 900.0)]
            row["methods"][m] = met
        dl = lowpass(-(yB - bases[k]) * al["sign"])
        tt = tk - tk.mean()
        cg = np.linalg.lstsq(np.c_[np.ones_like(tt), tt], dl - a_glob * lowpass(rf), rcond=None)[0]
        cols = np.arange(x0, x1)
        inside = (cols >= xk[idx].min()) & (cols <= xk[idx].max())
        pred[k, cols[inside]] = np.interp(cols[inside], xk[idx], bases[k] - al["sign"] * (a_glob * lowpass(rf) + cg[0] + cg[1] * tt))
        lines.append(row)
    rms_med = float(np.median([r["ref_rms_S"] for r in lines]))
    for r in lines:
        k = r["line"]
        r["category"] = ("quake" if k == k_main else "neighbour" if abs(k - k_main) == 1
                         else "other_active" if r["ref_rms_S"] > 2 * rms_med else "quiet")
    summary = {}
    keys = ("r_band", "r_raw_detrended", "in_0.1S", "lost_0.5S", "in_0.1S_global", "lost_0.5S_global", "rms_err_global_S")
    for cat in ("quake", "neighbour", "other_active", "quiet"):
        rr = [r for r in lines if r["category"] == cat]
        if not rr:
            continue
        summary[cat] = {"n_lines": len(rr), "lines": [r["line"] for r in rr],
                        "ref_rms_S_mean": round(float(np.mean([r["ref_rms_S"] for r in rr])), 4),
                        "ref_max_S_max": round(float(np.max([r["ref_max_S"] for r in rr])), 3),
                        "chance_abs_r_ref_pm900": round(float(np.mean([np.mean(np.abs(r["chance_r_ref_vs_ref_pm900"])) for r in rr])), 3)}
        for m in paths:
            summary[cat][m] = {key: round(float(np.mean([r["methods"][m][key] for r in rr])), 3) for key in keys}
            summary[cat][m]["chance_abs_r_pm900"] = round(float(np.mean([np.mean(np.abs(r["methods"][m]["chance_r_vs_ref_pm900"])) for r in rr])), 3)
    print(f"[{name}] main-shock line {k_main} (max |ref| at {str(t_on + t_max)[11:19]}); image quake line {kq}", flush=True)
    for cat, v in summary.items():
        print(f"[{name}] {cat:12s} {v['lines']} ref rms {v['ref_rms_S_mean']:.3f}S max {v['ref_max_S_max']:.2f}S chance|r| {v['chance_abs_r_ref_pm900']:.2f} | " +
              " | ".join(f"{m}: r {v[m]['r_band']:+.2f} in {v[m]['in_0.1S_global']:.2f} lost {v[m]['lost_0.5S_global']:.2f}" for m in paths), flush=True)

    # ---- figures (local only)
    draw_overlays(figs, img_s, paths, pred, k_main, x0, x1, S, name)
    quiet = [r["line"] for r in lines if r["category"] == "quiet"]
    best_dp = max((k for k in resid if k != k_main), key=lambda k: resid[k]["r_at_0"], default=None)
    show = [(k_main, "main shock")] + ([(best_dp, "best-tracked active line")] if best_dp is not None else []) + ([(quiet[len(quiet) // 2], "quiet")] if quiet else [])
    try:
        plot_series(figs / f"{name}_timeseries.png", name, show, paths, bases, al, a_glob, S, t_on, xa, x0, x1, t_ref, ref, t_lo, t_hi)
    except Exception as e:  # figures are optional
        print("plot failed:", e)

    res = {"scan": name, "image": cfg["image"], "label": {k: cfg[k] for k in ("on", "off", "label_mag", "label_note")},
           "event": cfg["event"], "reference": ref_info, "comparison_band_hz": BAND,
           "layout": {"area_full": cfg["area"], "window_spacing_full_px": S0, "line_field": field, "spacing_full_px": S_full,
                      "scale": f, "n_lines": len(bases), "filled_lines": filled, "dropped_edge_baselines": dropped,
                      "baselines_rescaled": np.round(bases, 2).tolist(), "y0_start_col": np.round(y0, 2).tolist(),
                      "rescaled_size": [W, H], "x0": x0, "x1": x1, "xa": xa, "pps_nominal": pps_nom,
                      "trace_darkness": D_scan, "render_darkness": D_film, "darkness_gain": gain, "dp_scale_of_full_res": ds},
           "judge": {"train_events": len(trn), "train_items": len(dev), "events_here": len(rows_ev),
                     "frac_take_A": float(np.mean(p > 0.5)) if len(p) else None},
           "alignment": {"model": "t(x, line k) = T0 + period * k + (x - xa) / pps; t in s relative to label ON; line 0 = first record line",
                         "T0_search_range": rng_on0, "quake_line_by_ink_spread": kq, "ink_spread": [round(v, 3) for v in spread],
                         "spec_quake_line_fit_3600s_fullband": spec, "quake_line_fit_bandlimited_period_free": spec_band,
                         "reference_active_lines": active, "chosen_dp_fit": al, "dp_fit_period_fixed_3600": al3600,
                         "dp_fit_halves": half, "residual_lag_per_line": resid,
                         "label_on_col_line0": x_on, "label_on_col_line0_fullres": c0 + x_on / f},
           "amplitude": {"global_scale_px_per_m": a_glob, "from_dp_lines": used, "per_line_slopes": [round(v, 1) for v in slopes],
                         "implied_magnification": mag,
                         "label_magnification": cfg["label_mag"]},
           "main_shock_line": k_main, "main_shock_ref_max_utc": str(t_on + t_max),
           "lines": lines, "summary": summary, "runtime_s": round(time.time() - clock, 1)}
    (out / f"{name}_results.json").write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(f"[{name}] wrote {out / (name + '_results.json')} ({res['runtime_s']}s)")


if __name__ == "__main__":
    main()
