"""L6 export (research/46): the rescued trace of one film chip as miniSEED, with its provenance and a quick-look plot.

usage: python -I scripts/export_mseed.py export <rescue folder> [--net XX] [--loc ""]
       python -I scripts/export_mseed.py paircheck <rescue folder> MAT|KIP|ALQ     # timing against the co-located digital
                                                                              # record (local only, numbers only)
Input: the rescue folder (paths.npz, l0/geometry.json, l1/record.json if present) and its review state (research/45:
review/decisions.json replayed through review_server.Review, which refuses decisions made for another paths.npz).
Exported samples: only columns that are auto-accepted (both SAM 3 passes agree) or accepted after a human-guided SAM 3 re-track;
disputed, held and no-record columns become gaps. The path is resampled at 1 Hz on L0's time model
t(x, k) = ON + T0 + P k + (x - xa) / pps_r; each time lies on exactly one line (line k holds [x0, x0 + P pps_r)).
Amplitude: line spacings, up on the film positive, per-line median removed; no instrument correction (research/36 §2).
Minute marks are bridged by linear interpolation inside each data run: seconds :58-:02 of every minute and :58-:06 at
the hour (the pen ticks of seismo_tools.remove_time_marks, 2 s / 6 s, widened by 2 s before for L0's timing error; the
measured footprint on MAT peaks at :59, research/46 §5); marks at a run edge are dropped, not extrapolated. Network code XX: WWSSN has no FDSN code, and WW is the Idaho Seismic Network.
Outputs in <folder>/export/: <NET>.<STA>.<LOC>.<CHA>.mseed, export.json, quicklook.png. The file is read back with
ObsPy and compared with what was written.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from obspy import Stream, Trace, UTCDateTime, read               # noqa: E402
from review_server import AUTO, HUMAN, S, Review                  # noqa: E402

MIN_SEG = 10                                                      # samples: shorter accepted runs are left out
CHANNEL = {"LPZ": "LHZ", "LPN": "LHN", "LPE": "LHE"}
NET_NOTE = "XX: WWSSN film digitizations have no FDSN network code (WW is the Idaho Seismic Network)"


def runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(e[::2].tolist(), e[1::2].tolist()))


def resample(rv):
    """1 Hz samples of the reviewed path on L0's time model -> start time, amplitude (NaN = gap), source code, line"""
    tm, geo = rv.tm, rv.geo
    if rv.on is None:
        raise SystemExit("no time model in l0/geometry.json: nothing can be timed, nothing exported")
    P, pps, T0, xa, n = tm["P"], tm["pps_r"], tm["T0"], geo["xa"], len(rv.bases)
    xs0 = tm.get("line0_trace_start_x") if tm.get("line0_trace_start_x") is not None else rv.x0
    xe = tm.get("last_line_trace_end_x") if tm.get("last_line_trace_end_x") is not None else rv.x1
    t_first, t_last = rv.on + T0 + (xs0 - xa) / pps, rv.on + T0 + P * (n - 1) + (xe - xa) / pps
    start = UTCDateTime(np.ceil(t_first.timestamp))
    N = int(np.floor(t_last - start)) + 1
    s = (start - rv.on) - T0 + np.arange(N, dtype=float)          # seconds on the model's clock
    k = np.floor((s - (rv.x0 - xa) / pps) / P).astype(int)
    x = xa + (s - P * k) * pps
    ok = (k >= 0) & (k < n) & (x >= rv.x0) & (x < rv.x1 - 1)                # columns i and i + 1 both inside
    kk, i = np.where(ok, k, 0), np.where(ok, np.floor(x), 0).astype(int)
    f = np.where(ok, x - i, 0.0)
    st_a, st_b = rv.status[kk, i], rv.status[kk, i + 1]
    ya, yb = rv.path[kk, i], rv.path[kk, i + 1]
    good = ok & np.isin(st_a, (AUTO, HUMAN)) & np.isin(st_b, (AUTO, HUMAN)) & np.isfinite(ya) & np.isfinite(yb)
    amp = np.where(good, (rv.bases[kk] - ((1 - f) * ya + f * yb)) / S, np.nan)
    for line in np.unique(kk[good]):                              # per-line median removed
        m = good & (kk == line)
        amp[m] -= np.median(amp[m])
    src = np.where(good, np.where((st_a == HUMAN) | (st_b == HUMAN), HUMAN, AUTO), 0)
    return start, amp, src, np.where(ok, k, -1)


def bridge_marks(start, amp):
    """minute marks: seconds :58-:02 (:58-:06 at the hour) are interpolated linearly inside each data run; marks at a
    run edge have no data on one side and are dropped"""
    t = np.round(start.timestamp + np.arange(len(amp)))
    sec60, sec3600 = np.mod(t, 60), np.mod(t, 3600)
    mark = (sec60 >= 58) | (sec60 < 3) | (sec3600 >= 3598) | (sec3600 < 7)
    out, nb, nd = amp.copy(), 0, 0
    for a, b in runs(np.isfinite(amp)):
        m = mark[a:b]
        keep = np.flatnonzero(~m)
        seg = out[a:b]                                            # a view: writes go to out
        if len(keep) < 2:
            nd += int(m.sum())
            seg[m] = np.nan
            continue
        inner = m & (np.arange(b - a) > keep[0]) & (np.arange(b - a) < keep[-1])
        seg[inner] = np.interp(np.flatnonzero(inner), keep, seg[keep])
        edge = m & ~inner
        seg[edge] = np.nan
        nb, nd = nb + int(inner.sum()), nd + int(edge.sum())
    return out, nb, nd


def export(folder, net="XX", loc=""):
    folder = Path(folder).resolve()
    t0 = time.time()
    rv = Review(folder)                                           # replays review decisions, checks the paths.npz hash
    rec = rv.record or {}
    sta = (rec.get("station") or {}).get("code") or rv.geo["name"].split("_")[0]
    comp, comp_src = (rec.get("component") or {}).get("name"), "L1 record (film border)"
    if comp is None:                                              # no L1: the archive file name's channel, flagged
        comp = {"LHZ": "LPZ", "LHN": "LPN", "LHE": "LPE"}.get((rv.geo.get("from_name") or {}).get("channel"))
        comp_src = "archive file name (no L1 record)"
    if comp not in CHANNEL:
        raise SystemExit(f"component {comp}: only long-period chips with a known component are exported")
    cha = CHANNEL[comp]
    start, amp, src, line = resample(rv)
    amp, n_bridged, n_dropped = bridge_marks(start, amp)
    segs = [(a, b) for a, b in runs(np.isfinite(amp)) if b - a >= MIN_SEG]
    st = Stream([Trace(data=amp[a:b].astype(np.float32),
                       header={"network": net, "station": sta, "location": loc, "channel": cha, "sampling_rate": 1.0,
                               "starttime": start + a}) for a, b in segs])
    out = folder / "export"
    out.mkdir(exist_ok=True)
    mseed = out / f"{net}.{sta}.{loc}.{cha}.mseed"
    st.write(str(mseed), format="MSEED", encoding="FLOAT32", reclen=4096)
    back = read(str(mseed))                                       # read back and compare with what was written
    back.sort()
    diff = max((float(np.max(np.abs(b.data.astype(np.float64) - a.data.astype(np.float64)))) for a, b in zip(st, back)), default=0.0)
    same_times = all(abs(a.stats.starttime - b.stats.starttime) < 1e-6 and a.stats.npts == b.stats.npts for a, b in zip(st, back))
    exported = np.zeros(len(amp), bool)
    for a, b in segs:
        exported[a:b] = True
    me = rec.get("main_event")
    info = {
        "chip": rv.geo["name"], "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "file": mseed.name,
        "nslc": f"{net}.{sta}.{loc}.{cha}", "network_note": NET_NOTE, "sampling_rate_hz": 1.0, "encoding": "FLOAT32",
        "units": "line spacings of the film record, up on the film positive, per-line median removed; "
                 "no instrument correction (research/36 §2)",
        "record_utc": [str(start)[:19], str(start + len(amp) - 1)[:19]], "samples_in_record": int(len(amp)),
        "samples_exported": int(exported.sum()), "fraction_exported": round(float(exported.mean()), 4),
        "by_source": {"auto_accepted": int((exported & (src == AUTO)).sum()), "human_reviewed": int((exported & (src == HUMAN)).sum())},
        "segments": len(segs), "gaps": max(0, len(segs) - 1), "minute_mark_samples_bridged": n_bridged,
        "minute_mark_samples_dropped_at_run_edges": n_dropped, "component_source": comp_src,
        "min_segment_samples": MIN_SEG,
        "time_model": {k: rv.tm.get(k) for k in ("T0", "P", "pps_r", "minute_ambiguous")} | {"on": rv.geo.get("on"), "xa": rv.geo["xa"]},
        "on_source": rec.get("on_source") or "label or file name given to rescue.py (no L1 record)",
        "station": rec.get("station"), "component": rec.get("component"),
        "main_event": None if not me else {k: me.get(k) for k in ("id", "origin_utc", "mag", "mag_type", "place", "dist_deg", "P_utc", "surface_utc")},
        "inputs": {"paths_npz_sha256": rv.paths_sha, "review_decisions": len(rv.decisions),
                   "agreement_rule": (rv.prov.get("L3") or {}).get("rule_sha256"),
                   "sam3_ckpt_sha256": ((rv.prov.get("L2") or {}).get("SAM3_film2") or {}).get("ckpt_sha256")},
        "readback": {"traces": len(back), "samples": int(sum(tr.stats.npts for tr in back)), "max_abs_diff": diff,
                     "same_start_times_and_lengths": same_times},
        "seconds": None,
    }
    quicklook(back, info, out / "quicklook.png")
    info["seconds"] = round(time.time() - t0, 1)
    (out / "export.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[{info['chip']}] {mseed.name}: {info['samples_exported']} of {info['samples_in_record']} s exported "
          f"({100 * info['fraction_exported']:.1f}%; auto {info['by_source']['auto_accepted']}, reviewed "
          f"{info['by_source']['human_reviewed']}), {info['segments']} segments, {n_bridged} mark samples bridged; "
          f"read back {info['readback']['traces']} traces, max diff {diff}, same times {same_times} ({info['seconds']} s)", flush=True)
    return info


def quicklook(st, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    me = info["main_event"]
    t_on = UTCDateTime(info["record_utc"][0])
    fig, axes = plt.subplots(2 if me else 1, 1, figsize=(12, 6.2 if me else 3.2), dpi=110, squeeze=False)
    ax = axes[0, 0]
    for tr in st:
        ax.plot((tr.times() + (tr.stats.starttime - t_on)) / 3600, tr.data, lw=0.35, color="k")
    ax.set_xlabel(f"hours since {info['record_utc'][0]} UTC")
    ax.set_ylabel("line spacings")
    ax.set_title(f"{info['nslc']} · {info['samples_exported']} of {info['samples_in_record']} s exported "
                 f"({100 * info['fraction_exported']:.0f}%); gaps = spans left for review", fontsize=9)
    if me and me.get("P_utc"):
        tp = UTCDateTime(me["P_utc"])
        ax2 = axes[1, 0]
        a, b = tp - 600, tp + 3600
        for tr in st.slice(a, b):
            ax2.plot((tr.times() + (tr.stats.starttime - tp)) / 60, tr.data, lw=0.6, color="k")
        ax2.axvline(0, color="#d08a00", ls="--", lw=1)
        ax2.text(0.3, 0.92, "P (TauP)", transform=ax2.get_xaxis_transform(), color="#9a6b00", fontsize=8)
        if me.get("surface_utc"):
            r0, r1 = (UTCDateTime(u) - tp for u in me["surface_utc"])
            ax2.axvspan(r0 / 60, r1 / 60, color="#d08a00", alpha=0.12, lw=0)
            ax2.text(r0 / 60, 0.92, " surface waves", transform=ax2.get_xaxis_transform(), color="#9a6b00", fontsize=8)
        ax2.set_xlim(-10, 60)
        ax2.set_xlabel(f"minutes after the predicted P of M{me['mag']} {me.get('place') or ''} ({me['dist_deg']}°)")
        ax2.set_ylabel("line spacings")
    for a in axes.ravel():
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def paircheck(folder, name, max_lag=90):
    """lag of the exported miniSEED against the co-located digital record simulated through a WWSSN LP response
    (research/25 reference): band 1/400-0.08 Hz, Pearson r over every exported segment of >= 15 min at each lag.
    Numbers only; the digital record never leaves this machine and no picture of it is made."""
    import real_paired_eval as rpe
    folder = Path(folder).resolve()
    cfg = rpe.SCANS[name]
    info = json.loads((folder / "export" / "export.json").read_text())
    st = read(str(folder / "export" / info["file"]))
    t_on = UTCDateTime(cfg["on"])
    t_ref, y_ref, _ = rpe.reference(cfg, t_on)
    assert np.allclose(np.diff(t_ref), 1.0), "the digital reference is not 1 sample/s"
    ref = rpe.bandpass(y_ref)
    rows = []                                                     # one fixed segment set, valid at every lag
    for tr in st:
        if tr.stats.npts < 900:
            continue
        i0 = int(round((tr.stats.starttime - t_on) - t_ref[0]))
        if i0 + 60 - max_lag >= 0 and i0 + tr.stats.npts - 60 + max_lag <= len(ref):
            rows.append((i0, rpe.bandpass(tr.data.astype(float))))
    lags = np.arange(-max_lag, max_lag + 1)
    r = []
    for L in lags:                                                # film sample at t pairs with the digital at t + L
        a = [film[60:-60] for _, film in rows]                    # trim filter edges
        b = [ref[i0 + L + 60:i0 + L + len(film) - 60] for i0, film in rows]
        r.append(rpe.pearson(np.concatenate(a), np.concatenate(b)) if rows else np.nan)
    r = np.asarray(r)
    best = int(lags[int(np.nanargmax(r))])
    res = {"pair": name, "segments_used": len(rows), "samples_used": int(sum(len(f) - 120 for _, f in rows)),
           "band_hz": list(rpe.BAND), "r_at_zero_lag": round(float(r[lags == 0][0]), 3), "best_lag_s": best,
           "r_at_best_lag": round(float(np.nanmax(r)), 3), "max_lag_s": max_lag,
           "note": "the film sample at time t is paired with the digital sample at t + lag; best lag > 0 means the "
                   "film's time labels are early by that many seconds"}
    info["pair_check"] = res
    (folder / "export" / "export.json").write_text(json.dumps(info, indent=1, default=str))
    print(f"[{info['chip']}] pair check vs digital {name}: r(0 s) {res['r_at_zero_lag']}, best lag {best} s "
          f"(r {res['r_at_best_lag']}), {res['segments_used']} segments, {res['samples_used']} s", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "paircheck"])
    ap.add_argument("folder")
    ap.add_argument("pair", nargs="?")
    ap.add_argument("--net", default="XX")
    ap.add_argument("--loc", default="")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.folder, a.net, a.loc)
    else:
        paircheck(a.folder, a.pair)
