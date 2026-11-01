"""SeismoRescue product pipeline prototype (research/43 §4, research/55): one WWSSN long-period film chip -> green / red traces.

usage: HF_HUB_OFFLINE=1 python -I scripts/rescue.py <chip image> [--on ISO] [--off ISO] [--area r0,r1,c0,c1]
                                                    [--negative 0|1] [--out DIR] [--reuse] [--l1]
--reuse: take both SAM 3 passes from an earlier paths.npz of the same chip (L0 is deterministic) and redo L3 and the outputs.
--l1: first run L1 (l1_context, research/44; Muse Spark, needs MODEL_API_KEY, public chips only): the ON read from the
      film border and the OFF read from the label replace --on / --off, and record.json joins the outputs.

[L1 ->] L0 (l0_geometry: layout and time from the image and the label times only) -> L2 (the one fine-tuned SAM 3 traces every
line twice, from the left edge and from the right edge, with its frozen checkpoint and decoding, on L0's rescaled image and
baselines) -> L3 (the frozen agreement rule, out/ml/fb/frozen_rule.json) -> out/rescue/<name>/:
  overlay.png      the whole record, SAM 3 path drawn green where auto-accepted and red where it needs review; columns
                   before the line-0 trace start (ON) and after the last line's trace end (OFF) hold no record and are
                   left out (not drawn, not counted)
  quake.png        zoom on the line with the widest swings
  paths.npz        SAM3 (forward), SAM3_back (backward), accepted, bases, y0, x0, x1 (rescaled frame)
  provenance.json  L0 geometry/time and flags, model and rule hashes, per line the accepted fraction and the review
                   segments with their UTC times from L0's time model; with --l1 the L1 summary (station, ON source,
                   main event and its predicted lines)
  l1/              with --l1: record.json and l1_transcript.json (research/44)
No reference waveform is read. Files made from local-only scans (KIP) stay under out/.
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")              # numpy's small solves: 13 s here, 3.5 min with a thread per core on a busy 16-core Linux box
import cv2                                                       # noqa: E402
import numpy as np                                               # noqa: E402

os.environ.setdefault("SEISMO_SAM3_OUT", "sam3_film2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")                     # never resolve model revisions online (research/45 §5)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import l0_geometry as l0                                         # noqa: E402
import ml_fb as fb                                               # noqa: E402
import ml_sam3 as m3                                             # noqa: E402
from ml_consensus import runs_ge, sha_file                       # noqa: E402
from obspy import UTCDateTime                                    # noqa: E402

SAM3_FROZEN = ROOT / "out" / "ml" / "sam3_film2" / "frozen_method.json"
RULE = fb.RULE
GREEN, RED = (40, 150, 40), (40, 40, 210)                         # BGR


def l2(img, bases, x0, x1, line0_x=None, last_x=None):
    """SAM 3 film2 (research/37 frozen method) from both ends of every line, on the same rescaled image (research/55).
    line0_x: line 0 starts at the label ON time; from the left edge the tracker would first cross empty paper (it lost the
    MAT line 0 that way), so its forward pass starts there. last_x: the same for the backward pass of a partial last line."""
    tracker = fb.load_model()
    H = img.shape[0]
    tops = np.clip(np.round(bases).astype(int) - m3.S.WIN_H // 2, 0, H - m3.S.WIN_H)     # as ml_sam3.eval_real
    P1 = fb.trace(tracker, img, bases, x0, x1, l0.S_TARGET, tops=tops)
    P2 = fb.trace(tracker, img, bases, x0, x1, l0.S_TARGET, tops=tops, mirror=True)
    if line0_x is not None and line0_x - x0 > 100:                # a span shorter than one window cannot be traced: left to review
        P1[0] = fb.trace(tracker, img, bases[:1], line0_x, x1, l0.S_TARGET, tops=tops[:1])[0] if x1 - line0_x >= m3.S.WIN_W else np.nan
    if last_x is not None and x1 - 1 - last_x > 100:
        P2[-1] = (fb.trace(tracker, img, bases[-1:], x0, last_x + 1, l0.S_TARGET, tops=tops[-1:], mirror=True)[0]
                  if last_x + 1 - x0 >= m3.S.WIN_W else np.nan)
    return P1, P2


def l3(P1, P2, x0, x1, rule):
    with np.errstate(invalid="ignore"):
        d = np.where(np.isfinite(P1[:, x0:x1]) & np.isfinite(P2[:, x0:x1]), np.abs(P1[:, x0:x1] - P2[:, x0:x1]) / l0.S_TARGET, np.inf)
    acc = np.zeros(P1.shape, bool)
    acc[:, x0:x1] = np.stack([runs_ge(a, rule["min_run_cols"]) for a in d < rule["tau_S"]])
    return acc


def draw(img, P, acc, x0, x1, rows=None, cols=None, scale=1, valid=None):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).astype(np.float32)
    vis = (0.5 * vis + 0.5 * 255).astype(np.uint8)                # lighter scan so the colours read
    if scale != 1:
        vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for k in range(P.shape[0]):
        for x in range(x0, x1 - 1):
            a, b = P[k, x], P[k, x + 1]
            if np.isfinite(a) and np.isfinite(b) and (valid is None or valid[k, x]):
                cv2.line(vis, (int(x * scale), int(round(a * scale))), (int((x + 1) * scale), int(round(b * scale))),
                         GREEN if acc[k, x] else RED, max(1, scale), cv2.LINE_AA)
    r0, r1 = rows or (0, vis.shape[0] // scale)
    c0, c1 = cols or (0, vis.shape[1] // scale)
    return vis[r0 * scale:r1 * scale, c0 * scale:c1 * scale]


def segments(mask):
    """[start, end) runs of True"""
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(e[::2].tolist(), e[1::2].tolist()))


def rescue(path, on=None, off=None, area=None, negative=None, out=None, reuse=False, with_l1=False):
    t_start = time.time()
    path = Path(path)
    out = Path(out or ROOT / "out" / "rescue" / path.stem)
    out.mkdir(parents=True, exist_ok=True)
    rec = None
    if with_l1:
        import l1_context
        rec = l1_context.l1(path, out / "l1")
        if rec["on_utc"]:                                         # the chip's own border and label, not the file name
            on, off = rec["on_utc"], rec["off"]["used_by_L0"]
    geo, L = l0.run(path, on, off, area, negative, out / "l0")
    if L is None:
        raise SystemExit(f"L0 failed: {geo['failed']} (give --area by hand)")
    img, bases, x0, x1 = L["img_s"], L["bases"], L["x0"], L["x1"]
    rule = json.loads(RULE.read_text())
    fz3 = json.loads(SAM3_FROZEN.read_text())
    tm = geo["time"]
    if reuse and (out / "paths.npz").exists():
        z = np.load(out / "paths.npz")
        assert np.allclose(z["bases"], bases), "saved paths come from a different layout"
        P1, P2 = z["SAM3"].astype(float), z["SAM3_back"].astype(float)
    else:
        xs, xe = tm.get("line0_trace_start_x"), tm.get("last_line_trace_end_x")
        P1, P2 = l2(img, bases, x0, x1, None if xs is None else int(xs), None if xe is None else int(xe))
    acc = l3(P1, P2, x0, x1, rule)
    valid = np.zeros(acc.shape, bool)                             # record span: ON (line 0) .. OFF (last line)
    valid[:, x0:x1] = True
    if tm.get("line0_trace_start_x") is not None:
        valid[0, :int(tm["line0_trace_start_x"])] = False
    if tm.get("last_line_trace_end_x") is not None:
        valid[-1, int(tm["last_line_trace_end_x"]) + 1:] = False
    acc &= valid
    np.savez_compressed(out / "paths.npz", SAM3=P1.astype(np.float32), SAM3_back=P2.astype(np.float32), accepted=acc,
                        bases=bases, y0=L["y0"], x0=x0, x1=x1)
    on_t = UTCDateTime(geo["on"]) if geo.get("on") else None

    def utc(k, x):
        if tm.get("T0") is None or on_t is None:
            return None
        return str(on_t + tm["T0"] + tm["P"] * k + (x - L["xa"]) / tm["pps_r"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # a line with no forward path has no spread
        spread = np.array([np.nanstd((P1[k] - bases[k])[valid[k]]) if valid[k].any() else 0.0 for k in range(len(bases))])
    kq = int(np.argmax(np.nan_to_num(spread)))
    lines = []
    for k in range(len(bases)):
        rev = [(a + x0, b + x0) for a, b in segments((~acc[k] & valid[k])[x0:x1])]
        lines.append({"line": k, "accepted_frac": round(float(acc[k][valid[k]].mean()), 3) if valid[k].any() else None,
                      "record_cols": int(valid[k].sum()),
                      "review_segments": [{"x": [a, b], "utc": [utc(k, a), utc(k, b)]} for a, b in rev if b - a >= 3]})
    prov = {"chip": str(path), "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "seconds": None,
            "L0": {k: geo.get(k) for k in ("how", "negative", "area_full", "S_full", "scale", "n_lines", "on", "off")}
                  | {"marks": {k: geo["marks"].get(k) for k in ("found", "contrast", "pps_full", "P", "minutes_per_line", "format_ok")},
                     "time": {k: tm.get(k) for k in ("T0", "pps_r", "P", "start_lead_s", "minute_ambiguous", "reason")}},
            "L2": {"SAM3_film2": {"ckpt_sha256": fz3["ckpt_sha256"], "lam": fz3["lam_final"],
                                  "passes": "forward from the left edge; backward on the mirrored image from the right edge",
                                  "restarted_at_x": {"line_0_forward": tm.get("line0_trace_start_x"), "last_line_backward": tm.get("last_line_trace_end_x")}}},
            "L3": {"rule": RULE.name, "rule_sha256": sha_file(RULE), "tau_S": rule["tau_S"], "min_run_cols": rule["min_run_cols"]},
            "L1": None if rec is None else {
                "status": rec["status"], "station": rec["station"], "component": rec["component"], "on_utc": rec["on_utc"],
                "on_source": rec["on_source"], "off": rec["off"], "flags": rec["flags"], "cost_usd": rec["l1"]["cost_usd"],
                "main_event": None if rec["main_event"] is None else {k: rec["main_event"].get(k) for k in (
                    "id", "origin_utc", "mag", "mag_type", "place", "dist_deg", "P_utc", "P_line", "P_lines_pm1",
                    "surface_utc", "surface_lines", "surface_lines_pm1")},
                "record": "l1/record.json"},
            "widest_swing_line": kq, "accepted_frac_all": round(float(acc[valid].mean()), 3), "lines": lines,
            "note": f"forward SAM 3 path exported where accepted. Accepted: the two passes are within {rule['tau_S']} line spacing for at least "
                    f"{rule['min_run_cols']} columns in a row. Review segments are the other columns; segments under 3 columns are not listed"}
    cv2.imwrite(str(out / "overlay.png"), draw(img, P1, acc, x0, x1, valid=valid))
    xm = int(np.nanargmax(np.where(valid[kq], np.abs(P1[kq] - bases[kq]), -1.0)))
    r0, r1 = int(max(0, bases[kq] - 3.5 * l0.S_TARGET)), int(min(img.shape[0], bases[kq] + 3.5 * l0.S_TARGET))
    c0, c1 = max(x0, xm - 300), min(x1, xm + 300)
    cv2.imwrite(str(out / "quake.png"), draw(img, P1, acc, x0, x1, (r0, r1), (c0, c1), scale=2, valid=valid))
    prov["seconds"] = round(time.time() - t_start, 1)
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, default=float))
    print(f"[{path.stem}] accepted {100 * prov['accepted_frac_all']:.1f}% of trace columns; widest-swing line {kq} "
          f"accepted {100 * lines[kq]['accepted_frac']:.1f}%; per line: "
          + " ".join(f"{d['line']}:{'-' if d['accepted_frac'] is None else round(100 * d['accepted_frac'])}" for d in lines)
          + f" ({prov['seconds']} s) -> {out}", flush=True)
    if rec is not None and rec["main_event"]:
        me = rec["main_event"]
        print(f"[{path.stem}] L1: {(rec['station'] or {}).get('code')} {rec['on_utc']} | main event {me['id']} M{me['mag']} "
              f"{me['place']} {me['dist_deg']} deg: P line {me['P_line']}, surface waves lines {me['surface_lines']} "
              f"(widest swing line {kq})", flush=True)
    return prov


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--on")
    ap.add_argument("--off")
    ap.add_argument("--area")
    ap.add_argument("--negative", type=int)
    ap.add_argument("--out")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--l1", action="store_true")
    a = ap.parse_args()
    rescue(a.image, a.on, a.off, tuple(int(v) for v in a.area.split(",")) if a.area else None,
           None if a.negative is None else bool(a.negative), a.out, a.reuse, a.l1)
