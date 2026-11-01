"""SeismoBench: score traced lines on synthetic seismogram film with exact ground truth. Needs NumPy only.

A split folder holds, per record, film/meta.json, film/heli_g<gain>.png (the film) and film/gt_g<gain>.npz (the truth:
pen, sig; one row per line, one column per pixel). Fetch the splits with bench/fetch_bench.py.

A submission folder holds <record>_g<gain>.npz for every image of the split:
    path    float, lines x columns: the row of the trace in each column; NaN where you give no answer
    accept  optional bool, same shape: the columns you stand behind. Without it, every column with a value counts as accepted

usage: python -I bench/score.py <split folder> <submission folder> [--name NAME] [--out result.json]

Scored: columns margin_x .. width - margin_x where the pen position is known. S is the line spacing.
- right: |path - pen| <= S/4. A missing value is wrong.
- flat line: the line's baseline as the answer, on the same columns. It shows how much of a score comes from a quiet pen.
- moved: |pen - baseline| > S/4. A flat line gets 0% of these by definition.
- without minute marks: columns with |pen - sig| > 0.1 S left out (the pen's ticks, which no tracer follows so far).
- r: on the earthquake line, (baseline - path)/S with missing values as 0, against (baseline - sig)/S. Mean over images.
- r_accepted: the same on accepted columns only, on images with at least 200 of them.
Intervals: 95% percentile bootstrap over earthquakes, 10,000 draws.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np

TOL, TICK, MIN_R_COLS, REPS, MAX_ROW = 0.25, 0.1, 200, 10000, 1e6
FIELDS = ("cols", "acc", "acc_right", "acc_flat", "right", "flat", "moved", "moved_right", "moved_acc", "moved_acc_right")


def tally(pen, base, S, path, acc, keep):
    """column counts of one line"""
    with np.errstate(invalid="ignore"):
        right = np.abs(path - pen) <= TOL * S
        flat = np.abs(pen - base) <= TOL * S
    moved = keep & ~flat
    a = acc & keep
    return (keep.sum(), a.sum(), (a & right).sum(), (a & flat).sum(), (keep & right).sum(), (keep & flat).sum(),
            moved.sum(), (moved & right).sum(), (moved & a).sum(), (moved & a & right).sum())


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def rates(t):
    """t: dict or arrays of the FIELDS"""
    return {"accepted_share": pct(t["acc"], t["cols"]), "accepted_right": pct(t["acc_right"], t["acc"]), "accepted_flat_line": pct(t["acc_flat"], t["acc"]),
            "all_right": pct(t["right"], t["cols"]), "all_flat_line": pct(t["flat"], t["cols"]), "moved_share": pct(t["moved"], t["cols"]),
            "moved_right": pct(t["moved_right"], t["moved"]), "moved_accepted_share": pct(t["moved_acc"], t["moved"]),
            "moved_accepted_right": pct(t["moved_acc_right"], t["moved_acc"])}


def interval(per_event, seed=0):
    """events x fields -> 95% intervals of every rate"""
    rng = np.random.default_rng(seed)
    n = per_event.shape[0]
    idx = rng.integers(0, n, (REPS, n))
    tot = per_event[idx].sum(1)
    t = {f: tot[:, i].astype(float) for i, f in enumerate(FIELDS)}
    with np.errstate(invalid="ignore", divide="ignore"):
        r = {"accepted_share": 100 * t["acc"] / t["cols"], "accepted_right": 100 * t["acc_right"] / t["acc"], "accepted_flat_line": 100 * t["acc_flat"] / t["acc"],
             "accepted_right_minus_flat_line": 100 * (t["acc_right"] - t["acc_flat"]) / t["acc"], "all_right": 100 * t["right"] / t["cols"],
             "moved_right": 100 * t["moved_right"] / t["moved"], "moved_accepted_right": 100 * t["moved_acc_right"] / t["moved_acc"]}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)               # nothing accepted: no interval
        return {k: [round(float(x), 2) for x in np.nanpercentile(v, [2.5, 97.5])] for k, v in r.items()}


def score(split, sub):
    split, sub = Path(split), Path(sub)
    records = sorted(p for p in split.iterdir() if (p / "film" / "meta.json").exists())
    if not records:
        raise SystemExit(f"no records in {split}")
    events = sorted({r.name[:8] for r in records})
    per = {m: {c: np.zeros((len(events), len(FIELDS)), np.int64) for c in ("quake", "all")} for m in ("with_marks", "without_marks")}
    rs, rs_acc, rs_same, sha, n_images = [], [], [], hashlib.sha256(), 0
    for rec in records:
        meta = json.loads((rec / "film" / "meta.json").read_text())
        S, x0, n, q = float(meta["spacing"]), int(meta["margin_x"]), int(meta["n_lines"]), int(meta["active_line"])
        base = meta["margin_y"] + S * np.arange(n)
        ev = events.index(rec.name[:8])
        for g in meta["gains"]:
            gain = f"{g:g}"
            gt = np.load(rec / "film" / f"gt_g{gain}.npz", allow_pickle=False)
            pen, sig = gt["pen"].astype(float), gt["sig"].astype(float)
            f = sub / f"{rec.name}_g{gain}.npz"
            if not f.exists():
                raise SystemExit(f"missing from the submission: {f.name}")
            sha.update(f.read_bytes())
            z = np.load(f, allow_pickle=False)
            if "path" not in z.files:
                raise SystemExit(f"{f.name}: no array named path")
            path = z["path"].astype(float)
            if path.shape != pen.shape:
                raise SystemExit(f"{f.name}: path has shape {path.shape}, expected {pen.shape}")
            path[~np.isfinite(path)] = np.nan                         # inf counts as no answer
            if np.nanmax(np.abs(path), initial=0) > MAX_ROW:
                raise SystemExit(f"{f.name}: path has rows beyond {MAX_ROW:g}; use NaN where you give no answer")
            acc = z["accept"].astype(bool) if "accept" in z.files else np.isfinite(path)
            if acc.shape != pen.shape or (acc & ~np.isfinite(path)).any():
                raise SystemExit(f"{f.name}: accept must have the shape of path and only mark columns with a value")
            sl = slice(x0, pen.shape[1] - x0)
            n_images += 1
            for k in range(n):
                ok = np.isfinite(pen[k, sl])
                with np.errstate(invalid="ignore"):
                    no_tick = ok & ~(np.abs(pen[k, sl] - sig[k, sl]) > TICK * S)
                for m, keep in (("with_marks", ok), ("without_marks", no_tick)):
                    t = tally(pen[k, sl], base[k], S, path[k, sl], acc[k, sl], keep)
                    per[m]["all"][ev] += t
                    if k == q:
                        per[m]["quake"][ev] += t
            pq, sq, aq = pen[q, sl], sig[q, sl], acc[q, sl]
            have = np.isfinite(pq) & np.isfinite(sq)
            a, s = np.nan_to_num((base[q] - path[q, sl]) / S), (base[q] - sq) / S
            r = pearson(a[have], s[have])
            rs.append(r)
            if (have & aq).sum() >= MIN_R_COLS:
                rs_acc.append(pearson(a[have & aq], s[have & aq]))
                rs_same.append(r)
    out = {"split": split.name, "submission_sha256": sha.hexdigest(), "n_images": n_images, "n_earthquakes": len(events), "r_quake_line": round(float(np.mean(rs)), 4),
           "r_accepted": round(float(np.mean(rs_acc)), 4) if rs_acc else None, "r_accepted_images": len(rs_acc), "r_same_images_all_columns": round(float(np.mean(rs_same)), 4) if rs_same else None}
    for m in per:
        out[m] = {}
        for c in ("quake", "all"):
            tot = dict(zip(FIELDS, (int(v) for v in per[m][c].sum(0))))
            out[m][c] = {"rates": {k: round(v, 4) for k, v in rates(tot).items()}, "ci95": interval(per[m][c]), "columns": tot}
    return out


def plain(o):
    """NaN has no JSON form: a rate with nothing to count is null"""
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, list):
        return [plain(v) for v in o]
    return None if isinstance(o, float) and o != o else o


def num(v, d):
    return "–" if v is None else f"{v:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("submission")
    ap.add_argument("--name")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = plain({"name": a.name or Path(a.submission).name, **score(a.split, a.submission)})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1, allow_nan=False))
    q, w = res["with_marks"]["quake"]["rates"], res["with_marks"]["all"]["rates"]
    print(f"{res['name']} on {res['split']}: {res['n_images']} images, {res['n_earthquakes']} earthquakes")
    print(f"  earthquake line: r {res['r_quake_line']:.3f} | accepted {num(q['accepted_share'], 1)}%, right {num(q['accepted_right'], 2)}% (flat line {num(q['accepted_flat_line'], 2)}%) | moved: right {num(q['moved_right'], 2)}%, accepted right {num(q['moved_accepted_right'], 2)}%")
    print(f"  all lines:       accepted {num(w['accepted_share'], 1)}%, right {num(w['accepted_right'], 2)}% (flat line {num(w['accepted_flat_line'], 2)}%)")


if __name__ == "__main__":
    main()
