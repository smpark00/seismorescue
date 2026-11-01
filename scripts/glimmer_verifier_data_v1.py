"""Glimmer verifier data (research/38 §8.3, T2): synthetic A/B "line identity at a crossing" examples from the film2
training renders, in the glimmer_task.py format. Local CPU only, no network, deterministic for a given seed.

Per image (quake line a = meta active_line, every other line j, d = pen[a] - pen[j]):
  events  "cross": sign(d[x-1]) != sign(d[x]), both nonzero, xc = x. "kiss": |d| < 2 px local minimum within +-3
          columns and no sign change there, xc = argmin. Events of one pair chained closer than 8 columns count once
          (the first).
  window  x0 = xc - round(u * 192), u ~ U(0.35, 0.65), inside [margin_x, W - margin_x); both lines finite on it and
          inside rows [4, H - 4) on columns [xc - 32, xc + 32). crop = img[:, x0:x0 + 192].
  paths   TRUE = pen[a]; SWAP = pen[a] before xc, pen[j] from xc. Both get the same NaN-aware Gaussian along x
          (sigma ~ U(0.5, 2) columns); SWAP is then reset to TRUE left of xc, where the smoothing would leak pen[j]
          a few columns early. Events whose paths differ by < 3 px right of xc are dropped.
Sampling: <= 10 events per image, >= 96 columns apart, about 25 % kiss; a pick never lowers what its image can
still hold, so a group reaches min(goal, capacity).
  train  dates not in VAL_GROUPS -> subset "train"
  val    VAL_GROUPS dates -> "v1a" (200, stage 1) and "v1b" (the rest of --n-val, stage 2) on disjoint images
Outputs in --out: train|val.npz + .jsonl (glimmer_task.load_split), manifest.json (counts, sha256, path-only
baselines), preview.png (8 composites, not data). Reads only out/film2_train (asserted).

usage: .venv/bin/python -I scripts/glimmer_verifier_data.py [--n-train 3000 --n-val 600 --seed 20261030 --out out/glimmer_verifier]
"""
import argparse
import hashlib
import json
import math
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import glimmer_task as gt                                    # noqa: E402

SRC = ROOT / "out" / "film2_train"
WIN, H = gt.WIN, 406                       # crop width; film2 render height (asserted on every image read)
CAP, GAP, GAP_AB = 10, 96, 192             # per image: <= 10 events, >= 96 columns apart; v1a vs v1b >= 192
KISS_PX, KISS_HW, MERGE = 2.0, 3, 8
NEAR, PAD, MIN_DIFF = 32, 4, 3.0
U_RANGE, SIG_RANGE, KISS_FRAC = (0.35, 0.65), (0.5, 2.0), 0.25
N_V1A = 200
P_BINS = (("P<2", 0, 2), ("2-5", 2, 5), ("5-10", 5, 10), ("P>=10", 10, math.inf))   # = ml_glimmer_modal bins
TYPES = ("cross", "kiss")


def src(p):
    """every data read goes through here: out/film2_train only"""
    p = Path(p).resolve()
    assert p.is_relative_to(SRC.resolve()), f"read outside out/film2_train: {p}"
    return p


def pbin(P):
    return next(n for n, lo, hi in P_BINS if lo <= P < hi)


def smooth(v, sigma):
    """NaN-aware Gaussian along x (normalized convolution)"""
    r = int(np.ceil(4 * sigma))
    g = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    m = np.isfinite(v)
    num = np.convolve(np.where(m, v, 0.0), g, "same")
    den = np.convolve(m.astype(np.float64), g, "same")
    return np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)


def find_events(pen, a):
    """[(j, xc, type)] between the quake line a and every other line j"""
    out, h = [], KISS_HW
    for j in range(len(pen)):
        if j == a:
            continue
        d = pen[a] - pen[j]
        ok = np.isfinite(d)
        s = np.sign(np.where(ok, d, 0.0))
        cross = np.flatnonzero(ok[:-1] & ok[1:] & (s[:-1] * s[1:] < 0)) + 1
        ad = np.abs(d)
        c, sc = ad[h:-h], s[h:-h]                                         # window centres x = h .. W - h - 1
        kiss = np.flatnonzero(sliding_window_view(ok, 2 * h + 1).all(1) & (c < KISS_PX)
                              & (c <= sliding_window_view(ad, 2 * h + 1).min(1)) & (sc != 0)
                              & (sliding_window_view(s, 2 * h + 1) == sc[:, None]).all(1)) + h
        prev = None
        for x, t in sorted([(int(x), "cross") for x in cross] + [(int(x), "kiss") for x in kiss]):
            if prev is None or x - prev >= MERGE:
                out.append((j, x, t))
            prev = x
    return out


def image_pool(rec, k, meta, seed, stats):
    """valid events of one image, each with its window and smoothed TRUE / SWAP paths (crop columns)"""
    with np.load(src(rec / "film" / f"gt_g{k:g}.npz")) as z:
        pen = z["pen"].astype(np.float64)
    a, mx, W = meta["active_line"], meta["margin_x"], pen.shape[1]
    P = round(float(meta["variant_of_k"][f"{k:g}"]["P"]), 4)
    rng = np.random.default_rng([seed, zlib.crc32(rec.name.encode()), int(k)])
    pool = []
    for j, xc, typ in find_events(pen, a):
        u, sig = rng.uniform(*U_RANGE), rng.uniform(*SIG_RANGE)
        x0 = xc - int(round(u * WIN))
        stats[typ]["events"] += 1
        if x0 < mx or x0 + WIN > W - mx or not np.isfinite(pen[[a, j], x0:x0 + WIN]).all():
            stats[typ]["drop_window"] += 1
            continue
        near = pen[[a, j], xc - NEAR:xc + NEAR]
        if not ((near >= PAD) & (near < H - PAD)).all():
            stats[typ]["drop_rows"] += 1
            continue
        r = int(np.ceil(4 * sig))                                         # smooth with real data beyond the crop
        cols = np.arange(x0 - r, x0 + WIN + r)
        assert cols[0] >= 0 and cols[-1] < W, (rec.name, k, xc)
        xi = xc - x0
        true = smooth(pen[a, cols], sig)[r:-r].astype(np.float32)
        swap = smooth(np.where(cols < xc, pen[a, cols], pen[j, cols]), sig)[r:-r].astype(np.float32)
        swap[:xi] = true[:xi]
        if np.abs(true[xi:] - swap[xi:]).max() < MIN_DIFF:
            stats[typ]["drop_same"] += 1
            continue
        stats[typ]["valid"] += 1
        pa = pen[a, x0:x0 + WIN]
        dev = np.abs(true - pa)                                           # smoothing vs the pen line, drawn columns
        pool.append({"record": rec.name, "group": rec.name[:8], "k": int(k), "P": P, "active": a, "neighbor": j,
                     "xc": xc, "x0": x0, "xc_in_crop": xi, "type": typ,
                     "slope_true": float(pen[a, xc + 1] - pen[a, xc - 1]) / 2,
                     "slope_other": float(pen[j, xc + 1] - pen[j, xc - 1]) / 2, "sigma": float(sig),
                     "true": true, "swap": swap,
                     "dev": (float(dev[(pa >= 0) & (pa <= H - 1)].max()), float(dev[xi - NEAR:xi + NEAR].max()))})
    return pool


def pack(xs):
    """most events with pairwise gaps >= GAP among sorted columns xs (leftmost-first greedy is optimal on a line)"""
    n, last = 0, -math.inf
    for x in xs:
        if x - last >= GAP:
            n, last = n + 1, x
    return n


def capacity(imgs):
    return sum(min(CAP, pack([e["xc"] for e in im["events"]])) for im in imgs)


def pick(im, types, rng):
    """one more event of this image (types in order of preference) that keeps the image's reachable count"""
    room = CAP - len(im["chosen"])
    comp = [e for e in im["events"] if all(abs(e["xc"] - c["xc"]) >= GAP for c in im["chosen"])]
    if room <= 0 or not comp:
        return None
    m = pack([e["xc"] for e in comp])
    perm = [comp[i] for i in rng.permutation(len(comp))]
    for e in (e for t in types for e in perm if e["type"] == t):
        # a pick removes at most 2 from a packing, so with m > room every pick is safe
        if m > room or 1 + min(room - 1, pack([c["xc"] for c in comp if abs(c["xc"] - e["xc"]) >= GAP])) == min(room, m):
            return e
    return None


def select(imgs, n_goal, rng):
    """round robin over images: kisses first (up to KISS_FRAC of the goal), then fill preferring crosses"""
    for im in imgs:
        im["chosen"] = []
    got = []
    for types, goal in ((("kiss",), round(KISS_FRAC * n_goal)), (("cross", "kiss"), n_goal)):
        while len(got) < goal:
            before = len(got)
            for i in rng.permutation(len(imgs)):
                if len(got) >= goal:
                    break
                e = pick(imgs[i], types, rng)
                if e is not None:
                    imgs[i]["chosen"].append(e)
                    got.append(e)
            if len(got) == before:
                break
    return got


def split_val(imgs, rng):
    """v1a / v1b on disjoint images (so no shared windows): P-stratified random order, v1a takes the shortest
    prefix whose capacity covers N_V1A"""
    key = {}
    for _, lo, hi in P_BINS:
        ims = [im for im in imgs if lo <= im["P"] < hi]
        for r, i in enumerate(rng.permutation(len(ims))):
            key[ims[i]["id"]] = ((r + 0.5) / len(ims), rng.random())
    order = sorted(imgs, key=lambda im: key[im["id"]])
    a, cap = [], 0
    for im in order:
        if cap >= N_V1A:
            break
        a.append(im)
        cap += capacity([im])
    ids = {im["id"] for im in a}
    return a, [im for im in order if im["id"] not in ids]


def finalize(events, subset, split, rng):
    """random order, exactly balanced labels; label 0 -> A = TRUE, B = SWAP; label 1 -> A = SWAP, B = TRUE"""
    assert len(events) % 2 == 0, (subset, len(events))
    order = rng.permutation(len(events))
    labels = rng.permutation(np.repeat(np.uint8([0, 1]), len(events) // 2))
    rows, cands = [], np.empty((len(events), 2, WIN), np.float32)
    for i, (o, lab) in enumerate(zip(order, labels)):
        e = events[o]
        cands[i] = (e["true"], e["swap"]) if lab == 0 else (e["swap"], e["true"])
        rows.append({"id": f"{subset}_{i:04d}", "split": split, "subset": subset, "record": e["record"],
                     "group": e["group"], "k": e["k"], "P": e["P"], "active": e["active"], "neighbor": e["neighbor"],
                     "xc": e["xc"], "x0": e["x0"], "xc_in_crop": e["xc_in_crop"], "type": e["type"],
                     "slope_true": round(e["slope_true"], 4), "slope_other": round(e["slope_other"], 4),
                     "sigma": e["sigma"], "label": int(lab)})              # sigma unrounded: paths re-derivable
    return rows, cands, labels, [events[o]["dev"] for o in order]


def crops_for(rows):
    """img[:, x0:x0 + WIN] per row, each image read once"""
    crops = np.empty((len(rows), H, WIN), np.uint8)
    by_img = defaultdict(list)
    for i, r in enumerate(rows):
        by_img[r["record"], r["k"]].append(i)
    for (rec, k), idx in sorted(by_img.items()):
        img = cv2.imread(str(src(SRC / rec / "film" / f"heli_g{k:g}.png")), cv2.IMREAD_GRAYSCALE)
        assert img is not None and img.shape[0] == H, (rec, k)
        for i in idx:
            crops[i] = img[:, rows[i]["x0"]:rows[i]["x0"] + WIN]
    return crops


def check(rows, cands, labels, crops):
    """the dataset contract, asserted before anything is written"""
    assert crops.shape == (len(rows), H, WIN) and crops.dtype == np.uint8
    assert cands.shape == (len(rows), 2, WIN) and np.isfinite(cands).all()
    for r, c, lab in zip(rows, cands, labels):
        xi = r["xc_in_crop"]
        assert lab == r["label"] and np.array_equal(c[0, :xi], c[1, :xi]), r["id"]
        assert np.abs(c[0, xi:] - c[1, xi:]).max() >= MIN_DIFF, r["id"]
    per = defaultdict(list)
    for r in rows:
        per[r["record"], r["k"]].append(r)
    for rs in per.values():
        assert len(rs) <= CAP, rs[0]["record"]
        for i, p in enumerate(rs):
            for q in rs[i + 1:]:
                assert abs(p["xc"] - q["xc"]) >= (GAP if p["subset"] == q["subset"] else GAP_AB), (p["id"], q["id"])
    for s in sorted({r["subset"] for r in rows}):
        lab = [r["label"] for r in rows if r["subset"] == s]
        assert 2 * sum(lab) == len(lab), s


def kink(c, xi):
    return abs((c[xi + 10] - c[xi + 2]) / 8 - (c[xi - 2] - c[xi - 10]) / 8)


def curvature(c, xi):
    s = c[xi - 9:xi + 10].astype(np.float64)                             # second differences at xi - 8 .. xi + 8
    return np.abs(s[2:] - 2 * s[1:-1] + s[:-2]).max()


def acc(hits):
    """mean hit rate (ties count 1/2) with a Wilson 95 % interval, as in ml_glimmer_modal._summary"""
    n = len(hits)
    if not n:
        return None
    p, z = float(np.mean(hits)), 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return {"n": n, "acc": round(p, 4), "ci95": [round(c - h, 4), round(c + h, 4)]}


def baselines(rows, cands):
    """path-only rules: the candidate with the smaller kink / curvature at xc is called the true one"""
    out = {}
    for name, f in (("kink", kink), ("curvature", curvature)):
        hits = []
        for r, c in zip(rows, cands):
            sa, sb = f(c[0], r["xc_in_crop"]), f(c[1], r["xc_in_crop"])
            hits.append(0.5 if sa == sb else float((sb < sa) == r["label"]))
        res = {}
        for s in sorted({r["subset"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r["subset"] == s]
            res[s] = {"all": acc([hits[i] for i in idx]),
                      "by_type": {t: acc([hits[i] for i in idx if rows[i]["type"] == t]) for t in TYPES},
                      "by_P": {b: acc([hits[i] for i in idx if pbin(rows[i]["P"]) == b]) for b, _, _ in P_BINS}}
        out[name] = res
    return out


def counts(rows):
    n = len(rows)
    nk = sum(r["type"] == "kiss" for r in rows)
    per_img = Counter((r["record"], r["k"]) for r in rows)
    by_p = {b: [r for r in rows if pbin(r["P"]) == b] for b, _, _ in P_BINS}
    return {"n": n, "labels": {"A_true": sum(r["label"] == 0 for r in rows), "B_true": sum(r["label"] == 1 for r in rows)},
            "by_type": {t: sum(r["type"] == t for r in rows) for t in TYPES}, "kiss_frac": round(nk / max(1, n), 4),
            "by_P": {b: len(rs) for b, rs in by_p.items()},
            "kiss_frac_by_P": {b: round(sum(r["type"] == "kiss" for r in rs) / len(rs), 4) if rs else None
                               for b, rs in by_p.items()},
            "images": len(per_img), "max_per_image": max(per_img.values(), default=0),
            "xc_in_crop": [min((r["xc_in_crop"] for r in rows), default=None), max((r["xc_in_crop"] for r in rows), default=None)]}


def spread(v):
    return {**{f"p{q}": round(float(np.percentile(v, q)), 2) for q in (50, 90, 99, 100)},
            "frac_gt_3px": round(float(np.mean(np.asarray(v) > 3)), 3)}


def preview(split_data, path):
    """8 composites: train 2 cross + 2 kiss (left column); v1a and v1b one cross + one kiss each (right column)"""
    cols = []
    for split, wants in (("train", [("train", "cross")] * 2 + [("train", "kiss")] * 2),
                         ("val", [("v1a", "cross"), ("v1a", "kiss"), ("v1b", "cross"), ("v1b", "kiss")])):
        rows, cands, crops = split_data[split]
        used, tiles = set(), []
        for sub, typ in wants:
            i = next((i for i, r in enumerate(rows) if r["subset"] == sub and r["type"] == typ and i not in used),
                     next(i for i, r in enumerate(rows) if i not in used))
            used.add(i)
            r = rows[i]
            im = np.asarray(gt.compose(crops[i], cands[i, 0], cands[i, 1]))
            cap = np.full((44, im.shape[1], 3), 255, np.uint8)
            txt = f"{r['id']} | {r['subset']} | {r['type']} | P {r['P']:.1f} | true: {gt.ANSWERS[r['label']]}"
            cv2.putText(cap, txt, (10, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
            tiles.append(np.vstack([cap, im, np.full((16, im.shape[1], 3), 255, np.uint8)]))
        cols.append(np.vstack(tiles))
    sheet = np.hstack([cols[0], np.full((cols[0].shape[0], 24, 3), 255, np.uint8), cols[1]])
    Image.fromarray(sheet).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--out", default=str(ROOT / "out" / "glimmer_verifier"))
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    recs = sorted(p for p in SRC.iterdir() if (p / "film" / "meta.json").exists())
    assert set(gt.VAL_GROUPS) <= {r.name[:8] for r in recs}, gt.VAL_GROUPS
    split_recs = {"train": [r for r in recs if r.name[:8] not in gt.VAL_GROUPS],
                  "val": [r for r in recs if r.name[:8] in gt.VAL_GROUPS]}
    imgs, stats = {}, {}
    for split, rs in split_recs.items():
        imgs[split] = []
        stats[split] = {t: Counter(dict.fromkeys(("events", "drop_window", "drop_rows", "drop_same", "valid"), 0))
                        for t in TYPES}
        for r in rs:
            meta = json.loads(src(r / "film" / "meta.json").read_text())
            for k in meta["gains"]:
                ev = sorted(image_pool(r, k, meta, args.seed, stats[split]), key=lambda e: (e["xc"], e["neighbor"]))
                imgs[split].append({"id": f"{r.name}/g{k:g}", "P": round(float(meta["variant_of_k"][f"{k:g}"]["P"]), 4),
                                    "events": ev})
    print(f"pool {time.time() - t0:.1f}s: " + ", ".join(f"{s} {len(imgs[s])} images {sum(len(i['events']) for i in imgs[s])} "
                                                      f"events, capacity {capacity(imgs[s])}" for s in imgs), flush=True)

    sel, rngs = {}, {s: np.random.default_rng([args.seed, zlib.crc32(s.encode())]) for s in ("train", "val", "v1a", "v1b")}
    n_tr = min(args.n_train, capacity(imgs["train"])) // 2 * 2
    sel["train"] = select(imgs["train"], n_tr, rngs["train"])
    va, vb = split_val(imgs["val"], rngs["val"])
    n_a = min(N_V1A, args.n_val, capacity(va)) // 2 * 2
    sel["v1a"] = select(va, n_a, rngs["v1a"])
    n_b = max(0, min(args.n_val - n_a, capacity(vb))) // 2 * 2
    sel["v1b"] = select(vb, n_b, rngs["v1b"])
    assert len(sel["train"]) == n_tr and len(sel["v1a"]) == n_a and len(sel["v1b"]) == n_b

    split_data, devs, all_rows, all_cands = {}, {}, [], []
    for split, subs in (("train", ("train",)), ("val", ("v1a", "v1b"))):
        parts = [finalize(sel[s], s, split, rngs[s]) for s in subs]
        rows = [r for p in parts for r in p[0]]
        cands = np.concatenate([p[1] for p in parts])
        labels = np.concatenate([p[2] for p in parts])
        devs[split] = [d for p in parts for d in p[3]]
        crops = crops_for(rows)
        check(rows, cands, labels, crops)
        np.savez_compressed(out / f"{split}.npz", crops=crops, cands=cands, labels=labels)
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        split_data[split] = (rows, cands, crops)
        all_rows += rows
        all_cands.append(cands)
    assert not {(r["record"], r["k"]) for r in all_rows if r["subset"] == "v1a"} & \
        {(r["record"], r["k"]) for r in all_rows if r["subset"] == "v1b"}
    preview(split_data, out / "preview.png")

    base = baselines(all_rows, np.concatenate(all_cands))
    base["note"] = ("Path-only rules that never look at the ink: kink = |mean slope over [xc+2, xc+10] - mean slope over "
                    "[xc-10, xc-2]|, curvature = max |second difference| on [xc-8, xc+8], both on the smoothed candidate; "
                    "the candidate with the smaller value is called the true one (ties count 1/2). A high path-only score "
                    "means the synthetic task can be solved from the path shape without looking at the ink, which a VLM "
                    "could also exploit. A score far below 0.5 is just as exploitable: the opposite rule (call the larger "
                    "value true) scores 1 - acc.")
    n_req = {"train": args.n_train, "v1a": min(N_V1A, args.n_val), "v1b": max(0, args.n_val - N_V1A)}
    sub_rows = {s: [r for r in all_rows if r["subset"] == s] for s in ("train", "v1a", "v1b")}
    val_rows = sub_rows["v1a"] + sub_rows["v1b"]
    man = {
        "task": "Glimmer line-identity verifier (research/38): [film | A | B], which candidate stays on the quake line",
        "seed": args.seed, "args": {"n_train": args.n_train, "n_val": args.n_val},
        "params": {"WIN": WIN, "H": H, "cap_per_image": CAP, "min_gap": GAP, "min_gap_v1a_v1b": GAP_AB,
                   "kiss_px": KISS_PX, "kiss_halfwin": KISS_HW, "merge_cols": MERGE, "near_cols": NEAR, "row_pad": PAD,
                   "min_diff_px": MIN_DIFF, "u": U_RANGE, "sigma": SIG_RANGE, "kiss_frac_target": KISS_FRAC,
                   "P_bins": [b for b, _, _ in P_BINS]},
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (Path(__file__).resolve(), HERE / "glimmer_task.py")},
        "VAL_GROUPS": list(gt.VAL_GROUPS),
        "records": {s: [r.name for r in rs] for s, rs in split_recs.items()},
        "pool": {s: {"images": len(imgs[s]), "images_with_events": sum(bool(i["events"]) for i in imgs[s]),
                     "capacity": capacity(imgs[s]), "by_type": {t: dict(stats[s][t]) for t in TYPES}} for s in imgs},
        "val_images": {"v1a": [i["id"] for i in va], "v1b": [i["id"] for i in vb],
                       "capacity": {"v1a": capacity(va), "v1b": capacity(vb)}},
        "requested": n_req,
        "counts": {"train": counts(sub_rows["train"]), "val": counts(val_rows),
                   **{s: counts(sub_rows[s]) for s in ("v1a", "v1b")}},
        "true_path_vs_pen_px": {s: {where: spread([d[w] for d in devs[s]]) for w, where in enumerate(("in_frame", "near_xc"))}
                                for s in devs},
        "baselines": base,
        "notes": [
            "SWAP is smoothed with the same Gaussian as TRUE and then reset to TRUE left of xc (smoothing would leak "
            "pen[j] up to 4 sigma columns before xc); the two candidates are identical left of xc.",
            "Chained events of one pair (consecutive gaps < 8 columns) count once, at the first.",
            "v1a and v1b use disjoint images (P-stratified); this meets the >= 192-column rule trivially.",
            "true_path_vs_pen_px: per example, max |smoothed TRUE - pen[a]| over the in-frame crop columns (what "
            "compose draws) and over [xc-32, xc+32): how far the smoothing moves the drawn true path off the pen line "
            "(mostly at sharp peaks).",
        ] + [f"{s}: requested {n_req[s]}, got {len(sub_rows[s])} (the per-image cap and spacing limit this split's "
             f"images to {capacity(va if s == 'v1a' else vb if s == 'v1b' else imgs['train'])} events)"
             for s in n_req if len(sub_rows[s]) < n_req[s]],
    }
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    for s in ("train", "v1a", "v1b"):
        c = man["counts"][s]
        print(f"{s}: n {c['n']} (requested {n_req[s]}), kiss {c['kiss_frac']}, by P {c['by_P']}, labels {c['labels']}; "
              f"kink {base['kink'][s]['all']['acc']}, curvature {base['curvature'][s]['all']['acc']}")
    print(f"done in {time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
