"""Tools for the seismologist agent (docs/RESULTS.md stage 3): digitized series, catalogue + TauP, onset detection,
AIC picking, plots, digitization confidence. Tools measure; the agent only chooses (docs/RESULTS.md P1).

CLI:
  python -I seismo_tools.py prefetch <data_bench_dir> <cache_dir>       # USGS catalogue per record (network)
  python -I seismo_tools.py series <bench_out_dir> <cache_props> <event_predictions.json> <out_dir>
      -> per record/condition npz: digitized series (from the CV-held-out learned judge), true signal, low-confidence mask
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

LINE_SEC, PPS = 900, 4
CONF_BAND = 0.2                   # |p_A - 0.5| below this = low-confidence digitization


# ---------- series ----------
def remove_time_marks(a, start_ts, fs=1.0):
    """minute marks (2 s) and hour marks (6 s) are known from the start time: bridge them by interpolation"""
    t = start_ts + np.arange(len(a)) / fs
    bad = (np.mod(t, 60) < 3) | (np.mod(t, 60) > 59) | (np.mod(t, 3600) < 7)
    out = a.copy()
    ok = ~bad & ~np.isnan(out)
    out[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), out[ok])
    return out


def paths_to_series(P, meta, base=None):
    """(n_lines, W) pixel paths -> 1 Hz amplitude in line-spacing units (up = positive)"""
    S, MX, MY, n = meta["spacing"], meta["margin_x"], meta["margin_y"], meta["n_lines"]
    cols = MX + PPS * np.arange(LINE_SEC)
    out = []
    for k in range(n):
        y = P[k, cols]
        b = (MY + k * S) if base == "grid" else float(np.nanmedian(y))
        out.append((b - y) / S)
    return np.concatenate(out)


# ---------- catalogue ----------
def prefetch(bench_dir, cache_dir):
    from obspy import UTCDateTime
    from obspy.clients.fdsn import Client
    from obspy.geodetics import locations2degrees
    from obspy.taup import TauPyModel
    usgs, taup = Client("USGS"), TauPyModel("iasp91")
    cache_dir.mkdir(parents=True, exist_ok=True)
    for side in sorted(bench_dir.glob("*.json")):
        info = json.loads(side.read_text())
        t0 = UTCDateTime(info["start_utc"])
        cat = usgs.get_events(starttime=t0 - 5400, endtime=t0 + 7200, minmagnitude=5.5)
        rows = []
        for ev in cat:
            o = ev.preferred_origin() or ev.origins[0]
            mag = (ev.preferred_magnitude() or ev.magnitudes[0]).mag
            dist = locations2degrees(o.latitude, o.longitude, info["sta_lat"], info["sta_lon"])
            dep = max(0.0, (o.depth or 0) / 1000)
            arr = {a.name: a.time for a in taup.get_travel_times(dep, dist, phase_list=["P", "S", "Pdiff"])}
            p = arr.get("P", arr.get("Pdiff"))
            rows.append({"id": f"ev{len(rows)}_{o.time.strftime('%Y%m%dT%H%M%S')}_M{mag:.1f}", "time": str(o.time), "mag": mag,
                         "lat": o.latitude, "lon": o.longitude, "depth_km": round(dep, 1), "dist_deg": round(dist, 2),
                         "P_s": None if p is None else round(float(o.time + p - t0), 1),
                         "S_s": None if "S" not in arr else round(float(o.time + arr["S"] - t0), 1),
                         "R1_s": round(float(o.time + dist * 111.19 / 3.9 - t0), 1),
                         "is_target": abs(float(o.time - UTCDateTime(info["event_time"]))) < 2})
        (cache_dir / f"{side.stem}.json").write_text(json.dumps({"station": f"{info['net']}.{info['sta']}",
                                                                  "sta_lat": info["sta_lat"], "sta_lon": info["sta_lon"],
                                                                  "start_utc": info["start_utc"], "events": rows}, indent=1))
        print(side.stem, len(rows), "events", sum(r["is_target"] for r in rows), "target", flush=True)


# ---------- measuring tools ----------
def sta_lta(a, sta=10, lta=120, on=3.0):
    """classic STA/LTA on the squared, demeaned series; returns onsets (s) with peak ratio"""
    x = np.nan_to_num(a - np.nanmedian(a)) ** 2
    c = np.r_[0, np.cumsum(x)]
    n = len(x)
    ratio = np.zeros(n)
    for i in range(lta, n):
        s_ = (c[i + 1] - c[i + 1 - sta]) / sta
        l_ = max((c[i + 1 - sta] - c[i + 1 - lta]) / (lta - sta), 1e-4)   # floor: digitized quiet spans can be exactly flat
        ratio[i] = s_ / l_
    trig = ratio > on
    starts = np.flatnonzero(np.diff(np.r_[0, trig.astype(int)]) == 1)
    out = []
    for s in starts:
        e = s
        while e < n and trig[e]:
            e += 1
        out.append({"onset_s": int(s), "peak_ratio": round(float(ratio[s:e].max()), 2), "duration_s": int(e - s)})
    return out


def aic_pick(a, t_approx, window=60):
    """Akaike information criterion picker (Maeda 1985) inside [t_approx - window, t_approx + window]"""
    i0, i1 = max(1, int(t_approx - window)), min(len(a) - 1, int(t_approx + window))
    x = np.nan_to_num(a[i0:i1] - np.nanmedian(a[i0:i1]))
    n = len(x)
    if n < 10:
        return {"pick_s": float(t_approx), "quality": 0.0}
    aic = np.full(n, np.inf)
    for k in range(2, n - 2):
        v1, v2 = np.var(x[:k]), np.var(x[k:])
        aic[k] = k * np.log(v1 + 1e-12) + (n - k - 1) * np.log(v2 + 1e-12)
    k = int(np.argmin(aic))
    snr = float(np.std(x[k:k + 30]) / max(np.std(x[max(0, k - 30):k]), 0.005))   # floor: flat digitized noise
    return {"pick_s": float(i0 + k), "snr_after_before": round(snr, 2)}


def plot_png(a, lowconf, t0, t1, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0, t1 = max(0, int(t0)), min(len(a), int(t1))
    t = np.arange(t0, t1)
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=100)
    ax.plot(t, a[t0:t1], lw=0.8, color="k")
    lc = lowconf[t0:t1]
    if lc.any():
        ax.fill_between(t, np.nanmin(a[t0:t1]), np.nanmax(a[t0:t1]), where=lc, color="orange", alpha=0.25,
                        label="low-confidence digitization")
        ax.legend(loc="upper right", fontsize=8)
    step = 60 if t1 - t0 <= 1800 else 300
    ax.set_xticks(np.arange((t0 // step) * step, t1 + 1, step))
    ax.grid(alpha=0.3)
    ax.set_xlabel("seconds since record start"); ax.set_ylabel("amplitude (line spacings)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return np.asarray(Image.open(buf).convert("RGB"))


# ---------- series building from the multi-agent digitizer ----------
def build_series(bench_out, cache_props, preds_file, out_dir):
    sys.path.insert(0, str(Path(__file__).parent))
    from arbiter import assemble, event_table
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {(p["record"], p["mode"], p["gain"], p["line"], p["s"]): p for p in json.loads(Path(preds_file).read_text())}
    from obspy import UTCDateTime
    for rec in sorted(p for p in bench_out.iterdir() if p.is_dir()):
        for mode, g in [("all", 2.0), ("all", 5.0), ("active", 1.5), ("active", 2.0)]:
            d = rec / mode
            meta = json.loads((d / "meta.json").read_text())
            z = np.load(cache_props / f"{rec.name}_{mode}_g{g:g}.npz")
            gt = np.load(d / f"gt_g{g:g}.npz")
            S, x0 = meta["spacing"], meta["margin_x"]
            x1 = gt["pen"].shape[1] - x0
            rows = event_table(z["A"], z["B"], z["ink"], x0, x1, 6.0)
            ps = [preds[(rec.name, mode, g, r["line"], r["s"])]["p_A"] for r in rows]
            est = assemble(z["A"], z["B"], rows, ["A" if p > 0.5 else "B" for p in ps])
            low = np.zeros_like(est, bool)
            for r, p in zip(rows, ps):
                if abs(p - 0.5) < CONF_BAND:
                    low[r["line"], r["s"]:r["e"]] = True
            start_ts = UTCDateTime(meta["start_utc"]).timestamp
            digi = remove_time_marks(paths_to_series(est, meta), start_ts)
            true = paths_to_series(gt["sig"], meta, base="grid")
            cols = meta["margin_x"] + PPS * np.arange(LINE_SEC)
            lowc = np.concatenate([low[k, cols] for k in range(meta["n_lines"])])
            # where the digitization is really wrong (GT, for scoring the QC flags only)
            fail = np.concatenate([np.abs(est[k, cols] - gt["pen"][k, cols]) > S / 2 for k in range(meta["n_lines"])])
            # the agent reads only <out_dir>; ground truth goes to a sibling <out_dir>_gt used by the evaluator
            gt_dir = out_dir.parent / (out_dir.name + "_gt")
            gt_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out_dir / f"{rec.name}_{mode}_g{g:g}.npz", digi=digi, lowconf=lowc, start_ts=start_ts)
            np.savez_compressed(gt_dir / f"{rec.name}_{mode}_g{g:g}.npz", true=true, fail=fail)
        print(rec.name, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "prefetch":
        prefetch(Path(sys.argv[2]), Path(sys.argv[3]))
    elif sys.argv[1] == "series":
        build_series(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5]))
