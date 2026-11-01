"""L0 (research/40): image-only geometry and time model for a WWSSN long-period film chip (LPZ).

usage: python -I scripts/l0_geometry.py run <chip image> [--on ISO] [--off ISO] [--area r0,r1,c0,c1] [--negative 0|1]
                                             [--out DIR]
       python -I scripts/l0_geometry.py validate [MAT KIP]  # development check against research/25's fitted time
       python -I scripts/l0_geometry.py test                # research/40 §4, once: the three calibration pairs
       python -I scripts/l0_geometry.py batch <folder>      # descriptive run on public chips (E4 set), no reference

No reference waveform is used. Inputs are the image and, when known, the label ON/OFF times (EarthScope file names
carry the ON date and time; OFF comes from the label).
1. polarity (a negative is inverted) and the trace-field box (largest block of bright paper rows / columns), unless
   given by hand; both are recorded as auto or manual;
2. layout exactly as research/25 (line-position field, deskew, baselines, isotropic rescale to 38 px spacing);
3. minute marks: on quiet lines, darkness in a band above the trace minus a band higher up (cancels vertical
   scratches), at full resolution; single-frequency DFTs at the mark period and harmonics 1-4, combined coherently
   across lines with the per-line phase drift of the line period P = 3600 + delta -> pps, P and the line-0 mark phase.
   The search range is pps in [S/48, S/36] (a line spacing is ~40-42 s of record on LPZ film), which excludes the
   mark-train harmonics;
4. T0: the line-0 trace start (the label ON time, written by the operator where the record starts) snapped to the
   nearest minute mark; the snap distance is reported (> 20 s is flagged). With OFF, the last line's trace end gives
   an independent P.
Time model (rescaled frame, as research/25): t(x, k) = ON + T0 + P k + (x - xa) / pps_r, line 0 = first record line.
Outputs in out/l0/<name>/: geometry.json, <name>_rescaled.png, layout.npz (bases, y0, x0, x1, xa, S).
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from obspy import UTCDateTime
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import real_paired_eval as rpe                                    # noqa: E402
from track_joint import ink_mask                                  # noqa: E402

S_TARGET = rpe.TARGET_S
SEC_PER_S = (36.0, 48.0)            # seconds of record per line spacing on LPZ film (MAT 42.1, KIP 40.4)
HARM = (1, 2, 3, 4)
DELTAS = np.arange(-30.0, 30.0, 0.05)
QUIET_SPREAD = 0.10                 # path spread (std about a 301-px running median) / S below which a line is quiet
SNAP_FLAG_S = 20.0


# ---------------------------------------------------------------- 1. polarity and trace field
def polarity_and_area(img, area=None, negative=None):
    h, w = img.shape
    small = cv2.resize(img, (w // 8, h // 8), interpolation=cv2.INTER_AREA)
    ctr = small[small.shape[0] // 4:3 * small.shape[0] // 4, small.shape[1] // 4:3 * small.shape[1] // 4]
    auto_neg = negative is None
    if auto_neg:                                           # the trace is the rare extreme: darker than the paper on a print,
        p1, med, p99 = np.percentile(ctr, [1, 50, 99])     # brighter than the film on a negative. A level threshold (< 128)
        negative = bool(p99 - med > med - p1)              # read ALQ 1975's mid-grey print as a negative (research/53)
    if negative:
        img, small = 255 - img, 255 - small
    auto_area = area is None
    if auto_area:                                          # paper rows / columns: 75th percentile above Otsu threshold
        thr, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        def longest(med):
            ok = np.r_[False, med > thr, False].astype(np.int8)
            e = np.flatnonzero(np.diff(ok))
            s, t = max(zip(e[::2], e[1::2]), key=lambda p: p[1] - p[0])
            return int(s), int(t)
        ra, rb = longest(np.percentile(small, 75, 1))
        ca, cb = longest(np.percentile(small[ra:rb], 75, 0))
        m = 2                                                      # small-image pixels kept clear of the border
        area = (8 * (ra + m), min(h, 8 * (rb - m)), 8 * (ca + m), min(w, 8 * (cb - m)))
    return img, bool(negative), tuple(int(v) for v in area), {"negative": "auto" if auto_neg else "manual",
                                                               "area": "auto" if auto_area else "manual"}


# ---------------------------------------------------------------- 2. layout (research/25 code)
def layout(img, area, n_expect=None):
    r0, r1, c0, c1 = area
    sub = img[r0:r1, c0:c1]
    dark = np.clip(float(np.median(sub)) - sub.astype(np.float32), 0, None)
    S0 = rpe.window_spacing(dark)
    xs_w, tracks = rpe.track_line_field(dark, S0)
    field = rpe.fit_shift_field(xs_w, tracks)
    sub_d = rpe.deskew(sub, field)
    bases_full, S_full, filled = rpe.baselines(np.clip(float(np.median(sub_d)) - sub_d.astype(np.float32), 0, None), S0)
    f = S_TARGET / S_full
    pad = int(round(1.5 * S_TARGET))
    img_s = cv2.copyMakeBorder(cv2.resize(sub_d, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), pad, pad, 0, 0,
                               cv2.BORDER_REPLICATE)
    bases = bases_full * f + pad
    D_scan, D_film = rpe.trace_darkness(img_s, bases, S_TARGET), rpe.film_render_darkness()
    gain = max(1.0, D_film / D_scan)
    if gain > 1.0:
        img_s = np.clip(245.0 - gain * np.clip(rpe.local_dark(img_s), 0, None), 0, 255).astype(np.uint8)
    ink0 = ink_mask(img_s)
    cov = np.array([rpe.trace_coverage(ink0, b, S_TARGET) for b in bases])
    dropped = []
    keep = np.ones(len(bases), bool)
    if n_expect is not None:                                       # research/25 rule: drop low-coverage edge lines
        while keep.sum() > n_expect:
            idx = np.flatnonzero(keep)
            j = idx[0] if cov[idx[0]] < cov[idx[-1]] else idx[-1]
            dropped.append({"row_rescaled": round(float(bases[j]), 1), "trace_coverage": round(float(cov[j]), 3)})
            keep[j] = False
    else:                                                          # no ON/OFF: drop edge lines that are not traces
        while keep.sum() > 2 and min(cov[np.flatnonzero(keep)[[0, -1]]]) < 0.02:
            idx = np.flatnonzero(keep)
            j = idx[0] if cov[idx[0]] < cov[idx[-1]] else idx[-1]
            dropped.append({"row_rescaled": round(float(bases[j]), 1), "trace_coverage": round(float(cov[j]), 3)})
            keep[j] = False
    bases = bases[keep]
    H, W = img_s.shape
    x0, x1 = 3, W - 3
    y0 = rpe.snap_start(img_s, bases, S_TARGET, x0, 120)
    return {"sub_d": sub_d, "img_s": img_s, "bases": bases, "y0": y0, "S_full": float(S_full), "f": float(f), "pad": pad,
            "x0": x0, "x1": x1, "xa": float(x0), "field": field, "filled": filled, "dropped": dropped, "gain": float(gain),
            "coverage": cov[keep].round(3).tolist()}


# ---------------------------------------------------------------- 3. minute marks
def trace_paths(img_s, bases):
    D = np.clip(rpe.local_dark(img_s), 0, None)
    paths, spread = {}, {}
    for k, b in enumerate(bases):
        p = rpe.dp_trace(D, b, S_TARGET, dmax=4)
        spread[k] = float(np.std(p - median_filter(p, 301)) / S_TARGET)
        paths[k] = p
    return paths, spread


def mark_signals(L, paths, quiet, side):
    """full-res darkness in [0.12, 0.45] S beyond the trace minus mid-gap [0.46, 0.54] S (side +1 above, -1 below);
    the mid-gap band holds scratches but neither this line's marks nor the neighbour's"""
    sub_d, f, pad, Sf = L["sub_d"], L["f"], L["pad"], L["S_full"]
    D = np.clip(rpe.local_dark(sub_d), 0, None)
    H, W = D.shape
    xs = np.arange(10, W - 10)
    hp = max(15, int(round(0.67 * Sf)) | 1)
    sig = []
    for k in quiet:
        y = (np.interp(xs * f, np.arange(len(paths[k])), paths[k]) - pad) / f

        def band(lo, hi):
            a = np.round(y - side * (hi if side > 0 else lo) * Sf).astype(int)
            b = np.round(y - side * (lo if side > 0 else hi) * Sf).astype(int)
            a, b = np.clip(np.minimum(a, b), 0, H), np.clip(np.maximum(a, b), 0, H)
            cs = np.vstack([np.zeros((1, W)), np.cumsum(D, 0)])
            return (cs[b, xs] - cs[a, xs]) / np.maximum(b - a, 1)
        m = band(0.12, 0.45) - band(0.46, 0.54)
        sig.append(np.clip(m - median_filter(m, hp), 0, None))
    return xs.astype(float), np.array(sig)


def joint_search(xs, sig, ks, pps_grid):
    ks = np.asarray(ks, float)
    rot1 = np.exp(-2j * np.pi * np.outer(DELTAS, ks) / 60.0)

    def at(pps):
        T = 60.0 * pps
        J = np.zeros(len(DELTAS))
        for h in HARM:
            C = sig @ np.exp(-2j * np.pi * h * xs / T)
            J += np.abs((rot1 ** h) @ C) ** 2
        j = int(np.argmax(J))
        return J[j], DELTAS[j]
    coarse = np.array([at(p) for p in pps_grid])
    i = int(np.argmax(coarse[:, 0]))
    fine_grid = np.arange(pps_grid[i] - 2 * (pps_grid[1] - pps_grid[0]), pps_grid[i] + 2 * (pps_grid[1] - pps_grid[0]), 2e-5)
    fine = np.array([at(p) for p in fine_grid])
    j = int(np.argmax(fine[:, 0]))
    pps, delta = float(fine_grid[j]), float(fine[j, 1])
    T = 60.0 * pps
    z = np.exp(-2j * np.pi * ks * delta / 60.0) @ (sig @ np.exp(-2j * np.pi * xs / T))
    phi0 = float((-np.angle(z) / (2 * np.pi) * T) % T)
    # peak sharpness: best over the coarse grid vs the median of the per-pps maxima outside +-1% of the peak
    off = np.abs(pps_grid / pps - 1) > 0.01
    contrast = float(coarse[i, 0] / np.median(coarse[off, 0])) if off.any() else float("nan")
    return {"pps_full": pps, "delta": delta, "P": 3600.0 + delta, "phi0_full": phi0, "contrast": contrast}


def marks(L, paths, spread):
    quiet = [k for k, s in spread.items() if s < QUIET_SPREAD]
    if len(quiet) < 3:
        return {"found": False, "reason": f"only {len(quiet)} quiet lines"}
    Sf = L["S_full"]
    pps_grid = np.arange(Sf / SEC_PER_S[1], Sf / SEC_PER_S[0], 1e-3 * Sf / 91.0)
    best = None
    for side in (1, -1):
        xs, sig = mark_signals(L, paths, quiet, side)
        r = joint_search(xs, sig, quiet, pps_grid)
        if best is None or r["contrast"] > best["contrast"]:
            best = {**r, "side": "above" if side > 0 else "below", "xs": xs, "sig": sig}
    xs, sig = best.pop("xs"), best.pop("sig")
    loo = []                                                     # leave-one-line-out spread
    for i in range(len(quiet)):
        m = np.ones(len(quiet), bool); m[i] = False
        g = np.arange(best["pps_full"] * 0.998, best["pps_full"] * 1.002, 1e-3 * Sf / 91.0)
        r = joint_search(xs, sig[m], np.asarray(quiet)[m], g)
        loo.append((r["pps_full"], r["P"]))
    loo = np.array(loo)
    n = len(quiet)
    best.update({"found": bool(best["contrast"] > 2.0), "quiet_lines": quiet,
                 "pps_full_sd_jackknife": float(np.sqrt((n - 1) / n * np.sum((loo[:, 0] - loo[:, 0].mean()) ** 2))),
                 "P_sd_jackknife": float(np.sqrt((n - 1) / n * np.sum((loo[:, 1] - loo[:, 1].mean()) ** 2)))})
    return best


# ---------------------------------------------------------------- 4. T0 from the line-0 trace start
def presence(L, paths, k, level, pps_r):
    """trace present per column: darkness within +-4 px of the line's DP path above half the typical trace level,
    smoothed over 20 s"""
    D, p = L["D_s"], paths[k]
    v = np.array([D[int(max(0, p[x] - 4)):int(p[x] + 5), x].max() for x in range(D.shape[1])])
    w = max(3, int(round(20 * pps_r)))
    return np.convolve((v > 0.5 * level).astype(float), np.ones(w) / w, "same") > 0.5


def edge_run(pres, x0, x1, gap, from_right):
    """the run of present columns touching the right (left) edge, gaps up to `gap` columns bridged"""
    idx = np.flatnonzero(pres[x0:x1]) + x0
    if idx.size == 0:
        return None
    if not from_right:
        idx = idx[::-1]
    x = idx[-1]
    for i in range(len(idx) - 2, -1, -1):
        if abs(idx[i + 1] - idx[i]) > gap:
            break
        x = idx[i]
    return int(x)


def time_model(L, mk, paths, spread, on=None, off=None):
    """ON mark = the first minute mark at or after the line-0 trace start (MAT and KIP traces start 25-35 s before
    the labelled ON minute, research/40); minute ambiguity is flagged when the start lies within 5 s of a mark"""
    bases, f = L["bases"], L["f"]
    pps_r = mk["pps_full"] * f
    T = 60.0 * pps_r
    phi0_r = mk["phi0_full"] * f                                 # the DFT used absolute area columns: same origin
    D = L["D_s"]
    full = [k for k in range(1, len(bases) - 1) if L["coverage"][k] > 0.5] or list(range(len(bases)))
    level = float(np.median([np.median([D[int(max(0, paths[k][x] - 4)):int(paths[k][x] + 5), x].max()
                                         for x in range(L["x0"], L["x1"], 7)]) for k in full]))
    gap = int(round(120 * pps_r))
    xs0 = edge_run(presence(L, paths, 0, level, pps_r), L["x0"], L["x1"], gap, from_right=True)
    out = {"pps_r": pps_r, "P": mk["P"], "trace_level": level, "line0_trace_start_x": xs0}
    if xs0 is None:
        out["T0"] = None
        return out
    tol = 3.0 * pps_r
    n = int(np.ceil((xs0 - tol - phi0_r) / T))
    x_on = phi0_r + n * T
    lead = (x_on - xs0) / pps_r                                  # seconds the trace starts before the ON mark
    out.update({"x_on": x_on, "start_lead_s": lead, "minute_ambiguous": bool(lead < 5.0 or lead > 55.0),
                "T0": -(x_on - L["xa"]) / pps_r})
    if on is not None and off is not None:
        xe = edge_run(presence(L, paths, len(bases) - 1, level, pps_r), L["x0"], L["x1"], gap, from_right=False)
        if xe is not None:
            dur = UTCDateTime(off) - UTCDateTime(on)
            out["last_line_trace_end_x"] = xe
            out["P_from_off"] = (dur - out["T0"] - (xe - L["xa"]) / pps_r) / (len(bases) - 1)
    return out


# ---------------------------------------------------------------- driver
def parse_name(name):
    m = re.match(r"([A-Z0-9]+)_(\d\d)_(\d\d)_(\d\d)_(\d\d)(\d\d)_([A-Z]{3})", name)
    if not m:
        return None
    sta, yy, mo, dd, hh, mi, cha = m.groups()
    year = 1900 + int(yy)
    return {"station": sta, "on": f"{year}-{mo}-{dd}T{hh}:{mi}:00", "channel": cha}


def run(path, on=None, off=None, area=None, negative=None, out=None, verbose=True):
    t_start = time.time()
    path = Path(path)
    name = path.stem
    meta = parse_name(name)
    on = on or (meta or {}).get("on")
    img = cv2.imread(str(path), 0)
    img, negative, area, how = polarity_and_area(img, area, negative)
    n_expect = int(np.ceil((UTCDateTime(off) - UTCDateTime(on)) / 3600.0)) + 1 if on and off else None
    try:
        L = layout(img, area, n_expect)
    except (IndexError, ValueError) as e:                         # e.g. no line track over 80% of the windows
        geo = {"image": str(path), "name": name, "from_name": meta, "on": on, "off": off, "how": how, "negative": negative,
               "area_full": list(area), "failed": f"layout: {type(e).__name__}: {e}", "seconds": round(time.time() - t_start, 1)}
        if out is not None:
            Path(out).mkdir(parents=True, exist_ok=True)
            (Path(out) / "geometry.json").write_text(json.dumps(geo, indent=1))
        if verbose:
            print(f"[{name}] {how} negative {negative} area {area} | FAILED {geo['failed']}", flush=True)
        return geo, None
    L["D_s"] = np.clip(rpe.local_dark(L["img_s"]), 0, None)
    paths, spread = trace_paths(L["img_s"], L["bases"])
    mk = marks(L, paths, spread)
    if mk.get("found"):                          # 60-min lines only; 30-min-line chips alias to ~120 here (research/40 §6)
        mk["minutes_per_line"] = float(L["sub_d"].shape[1] / (60.0 * mk["pps_full"]))
        mk["format_ok"] = bool(55.0 <= mk["minutes_per_line"] <= 66.0)
    if not mk.get("found"):
        tm = {"T0": None, "reason": "minute marks not found"}
    elif not mk["format_ok"]:
        tm = {"T0": None, "reason": f"unsupported line format: {mk['minutes_per_line']:.1f} min per line (60 expected)"}
    else:
        tm = time_model(L, mk, paths, spread, on, off)
    geo = {"image": str(path), "name": name, "from_name": meta, "on": on, "off": off, "how": how, "negative": negative,
           "area_full": list(area), "S_full": L["S_full"], "scale": L["f"], "pad": L["pad"], "n_lines": int(len(L["bases"])),
           "n_expect": n_expect, "filled_lines": L["filled"], "dropped_edge_baselines": L["dropped"], "gain": L["gain"],
           "line_coverage": L["coverage"], "line_spread": {str(k): round(v, 3) for k, v in spread.items()},
           "x0": L["x0"], "x1": L["x1"], "xa": L["xa"], "marks": {k: v for k, v in mk.items()}, "time": tm,
           "model": "t(x, k) = ON + T0 + P k + (x - xa) / pps_r, rescaled frame, line 0 = first record line",
           "seconds": round(time.time() - t_start, 1)}
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"{name}_rescaled.png"), L["img_s"])
        np.savez_compressed(out / "layout.npz", bases=L["bases"], y0=L["y0"], x0=L["x0"], x1=L["x1"], xa=L["xa"], S=S_TARGET)
        (out / "geometry.json").write_text(json.dumps(geo, indent=1, default=float))
    if verbose:
        m, t = geo["marks"], geo["time"]
        print(f"[{name}] {how} negative {negative} area {area} S {L['S_full']:.1f}px lines {geo['n_lines']} | marks found "
              f"{m.get('found')} ({m.get('side')}, contrast {m.get('contrast', float('nan')):.1f}, {len(m.get('quiet_lines', []))} quiet) "
              f"pps {m.get('pps_full', float('nan')):.4f}+-{m.get('pps_full_sd_jackknife', float('nan')):.4f} "
              f"P {m.get('P', float('nan')):.2f}+-{m.get('P_sd_jackknife', float('nan')):.2f} | T0 {t.get('T0')} lead {t.get('start_lead_s')} "
              f"ambiguous {t.get('minute_ambiguous')} P_from_off {t.get('P_from_off')} ({geo['seconds']} s)", flush=True)
    return geo, L


def ref_of(name):
    """config and reference-fitted time model: research/25 (MAT, KIP) or the research/31 calibration pairs"""
    if name in rpe.SCANS:
        return rpe.SCANS[name], json.loads((ROOT / "out" / "real_paired" / f"{name}_results.json").read_text())
    import ml_sim2real_calib as cal
    return cal.PAIRED[name], json.loads((ROOT / "out" / "ml" / "sim2real" / "calib_paired" / f"{name}_results.json").read_text())


def compare(name):
    """L0 (manual area and fully automatic) vs the reference-fitted model: pps, P, and seconds at 3 columns per line"""
    cfg, ref = ref_of(name)
    RL, a = ref["layout"], ref["alignment"]["chosen_dp_fit"]
    c0_gt, r0_gt = RL["area_full"][2], RL["area_full"][0]
    rows_gt = r0_gt + (np.array(RL["baselines_rescaled"]) - 57) / RL["scale"]
    res = {}
    for mode, area, neg in (("manual_area", tuple(cfg["area"]), cfg["negative"]), ("auto", None, None)):
        geo, L = run(ROOT / cfg["image"], cfg["on"], cfg["off"], area, neg, ROOT / "out" / "l0" / f"{name}_{mode}")
        if L is None:
            res[mode] = {"failed": geo["failed"], "how": geo["how"], "area": geo["area_full"]}
            continue
        t, c0_me = geo["time"], geo["area_full"][2]
        rows_me = geo["area_full"][0] + (L["bases"] - geo["pad"]) / geo["scale"]   # lines matched by full-res row
        k_off = int(round(np.median([np.argmin(np.abs(rows_me - y)) - i for i, y in enumerate(rows_gt)])))
        row = {"pps_full": geo["marks"].get("pps_full"), "pps_full_ref": a["pps"] / RL["scale"],
               "pps_rel_err": geo["marks"].get("pps_full", np.nan) / (a["pps"] / RL["scale"]) - 1,
               "P": geo["marks"].get("P"), "P_ref": a["period"], "P_from_off": t.get("P_from_off"),
               "marks_found": geo["marks"].get("found"), "contrast": geo["marks"].get("contrast"),
               "line_index_offset_vs_ref": k_off, "start_lead_s": t.get("start_lead_s"),
               "minute_ambiguous": t.get("minute_ambiguous"), "how": geo["how"], "area": geo["area_full"]}
        if t.get("T0") is not None:
            dts = []
            for k_gt in range(len(rows_gt)):
                k_me = k_gt + k_off
                if not 0 <= k_me < len(L["bases"]):
                    continue
                for frac in (0.05, 0.5, 0.95):
                    c = c0_gt + frac * (RL["area_full"][3] - c0_gt)                   # full-res column
                    t_gt = a["T0"] + a["period"] * k_gt + ((c - c0_gt) * RL["scale"] - 3) / a["pps"]
                    t_me = t["T0"] + geo["marks"]["P"] * k_me + ((c - c0_me) * geo["scale"] - geo["xa"]) / t["pps_r"]
                    dts.append(t_me - t_gt)
            dts = np.array(dts)
            row.update({"dt_median_s": float(np.median(dts)), "dt_max_abs_s": float(np.max(np.abs(dts))),
                        "dt_line0_mid_s": float(dts[1]) if len(dts) > 1 else None,
                        "x_on_full": c0_me + t["x_on"] / geo["scale"],
                        "x_on_full_ref": c0_gt + ref["alignment"]["label_on_col_line0"] / RL["scale"]})
        res[mode] = row
        print(name, mode, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
    return res


def validate(names=("MAT", "KIP")):
    """development check on the scans L0 was built on (research/40 §3)"""
    (ROOT / "out" / "l0" / "validate.json").write_text(json.dumps({n: compare(n) for n in names}, indent=1, default=float))


TEST = ("CAL_ALQ73", "CAL_KON73A", "CAL_KON73B")      # research/40 §4: only CAL_ALQ73 has a stable reference fit
CRIT = {"pps_rel": 0.003, "P_s": 1.0, "dt_median_s": 10.0, "dt_max_s": 30.0}


def cmd_test():
    """one run with the frozen code on the calibration pairs; CAL_ALQ73 (auto mode) is scored against CRIT"""
    out = ROOT / "out" / "l0"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "test_calib_started.json", "x") as f:
        f.write(json.dumps({"started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}))
    res = {n: compare(n) for n in TEST}
    r = res[TEST[0]]["auto"]
    checks = {"marks_found": bool(r["marks_found"]), "pps": abs(r["pps_rel_err"]) <= CRIT["pps_rel"],
              "P": abs(r["P"] - r["P_ref"]) <= CRIT["P_s"], "lines": r["line_index_offset_vs_ref"] == 0,
              "dt_median": abs(r.get("dt_median_s", np.inf)) <= CRIT["dt_median_s"],
              "dt_max": r.get("dt_max_abs_s", np.inf) <= CRIT["dt_max_s"]}
    res["primary"] = {"scan": TEST[0], "mode": "auto", "criteria": CRIT, "checks": checks, "pass": all(checks.values())}
    (out / "test_calib.json").write_text(json.dumps(res, indent=1, default=float))
    print("PRIMARY", TEST[0], "PASS" if all(checks.values()) else "FAIL", checks)


def cmd_batch(folder):
    """descriptive robustness run on public chips without a reference (E4 set): ON from the file name, OFF unknown"""
    rows = []
    for p in sorted(Path(folder).glob("*.TIFF")):
        geo, L = run(p, out=ROOT / "out" / "l0" / "batch" / p.stem)
        m, t = geo.get("marks", {}), geo.get("time", {})
        rows.append({"name": p.stem, "failed": geo.get("failed"), "negative": geo["negative"], "how": geo["how"],
                     "area": geo["area_full"], "S_full": geo.get("S_full"), "n_lines": geo.get("n_lines"),
                     "marks_found": m.get("found"), "side": m.get("side"), "contrast": m.get("contrast"),
                     "n_quiet": len(m.get("quiet_lines", [])), "pps_full": m.get("pps_full"),
                     "sec_per_spacing": geo["S_full"] / m["pps_full"] if m.get("pps_full") else None,
                     "P": m.get("P"), "P_sd": m.get("P_sd_jackknife"), "T0": t.get("T0"),
                     "start_lead_s": t.get("start_lead_s"), "minute_ambiguous": t.get("minute_ambiguous")})
    (ROOT / "out" / "l0" / "batch.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "validate":
        validate(sys.argv[2:] or ("MAT", "KIP"))
    elif sys.argv[1] == "test":
        cmd_test()
    elif sys.argv[1] == "batch":
        cmd_batch(sys.argv[2])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("image")
        ap.add_argument("--on")
        ap.add_argument("--off")
        ap.add_argument("--area")
        ap.add_argument("--negative", type=int)
        ap.add_argument("--out")
        a = ap.parse_args()
        area = tuple(int(v) for v in a.area.split(",")) if a.area else None
        run(a.image, a.on, a.off, area, None if a.negative is None else bool(a.negative),
            a.out or ROOT / "out" / "l0" / Path(a.image).stem)
