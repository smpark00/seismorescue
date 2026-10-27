"""Row-anchor sequence model + Viterbi decoding for target-conditioned helicorder trace extraction (docs/RESULTS.md).

Input (target-conditioned): a crop centred on the target line's baseline (layout), +-10.5 line spacings, at half
resolution (19 px per spacing, 2 s per column); channels = local ink darkness (normalised by the image's typical trace
darkness) and a start hint (Gaussian at the target's GT row at the first column = the trackers' start information).
Encoder: small CNN pooling rows 400 -> 25, per-column flatten -> Transformer along x (bidirectional context, tokens of
4 s). Head: per-column classification over 200 row bins (4 px at 38 px spacing) from the upsampled context map fused
with the 200-row CNN features; Gaussian soft labels (sigma 1 bin). Decoding: Viterbi over columns, emissions = alpha *
log p, transition = empirical bin-step distribution of the training GT, start prior at the hint; argmax = ablation;
both refined by the local expectation within +-2 bins.
Training composites: other random train images are stacked above / below the target image (line grid continued), so
crops are filled with lines as on a real 24-line record instead of 8 lines on blank paper.

usage: python -I scripts/ml_rowanchor.py split | cacheB val|holdout | bench | train [minutes] | val | freeze | holdout | real |
       figs [all|curve|val|holdout|real]
"""
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from track_joint import score as tj_score  # noqa: E402

OUT = ROOT / "out" / "ml" / "rowanchor"
TRAIN_DIR, HOLD_DIR = ROOT / "out" / "film_train", ROOT / "out" / "film_holdout3"
HOLD_CACHE = ROOT / "out" / "cache_film_holdout3"
G_TRAIN, G_HOLD = (1, 2, 3, 4, 6, 8), (1, 2, 4, 6, 8)
CH, NB, WIN, DSTEP = 400, 200, 1024, 64          # crop rows (half res), row bins, train window (cols), max bin step
BATCH = 6
DEV = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.set_num_threads(4)
cv2.setNumThreads(2)


# ---------------------------------------------------------------- data
def prep(img, x0, x1):
    """full-res uint8 image -> half-res local ink darkness / typical trace darkness (float16), row/col factors"""
    bg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    d = np.clip(bg.astype(np.float32) - img.astype(np.float32), 0, None)
    H, W = d.shape
    h = cv2.resize(d, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    fx, fy = W / h.shape[1], H / h.shape[0]
    s = float(np.median(h[:, int(x0 / fx):int(x1 / fx)].max(0)))
    return np.clip(h / max(s, 1e-3), 0, 4).astype(np.float16), fy, fx


def make_crop(D, fy, fx, base, x0, c0, ncol, sy=1.0, sx=1.0, skew=0.0, dy=0.0, xmid=0.0):
    """crop row r <-> full-res y = base + dy + 2 sy (r - (CH-1)/2) + skew (x - xmid); col c <-> x = x0 + (2(c0+c) + .5) sx"""
    x = x0 + (2 * np.arange(c0, c0 + ncol, dtype=np.float32) + 0.5) * sx
    r = np.arange(CH, dtype=np.float32) - (CH - 1) / 2
    y = base + dy + 2 * sy * r[:, None] + skew * (x - xmid)[None]
    mx = np.ascontiguousarray(np.broadcast_to((x + 0.5) / fx - 0.5, y.shape), np.float32)
    my = ((y + 0.5) / fy - 0.5).astype(np.float32)
    return cv2.remap(D.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0), x


def gt_at(pen_k, x):
    """pen row interpolated at float columns x (NaN if a neighbour is NaN or outside)"""
    W = len(pen_k)
    xf = np.floor(x).astype(int)
    a = (x - xf).astype(np.float32)
    y = pen_k[np.clip(xf, 0, W - 1)] * (1 - a) + pen_k[np.clip(xf + 1, 0, W - 1)] * a
    y[(x < 0) | (x > W - 1)] = np.nan
    return y


def to_bin(y, base, x, sy=1.0, skew=0.0, dy=0.0, xmid=0.0):
    r = (y - base - dy - skew * (x - xmid)) / (2 * sy) + (CH - 1) / 2
    return (r - 0.5) / 2


def from_bin(u, base):
    return base + 4 * u - 398.0


def hint_map(u0, ncol):
    h = np.zeros((CH, ncol), np.float32)
    if u0 is not None and np.isfinite(u0):
        h[:, :6] = np.exp(-0.5 * ((np.arange(CH) - (2 * u0 + 0.5)) / 1.5) ** 2)[:, None]
    return h


def load_item(rec_dir, g):
    d = rec_dir / "film"
    meta = json.loads((d / "meta.json").read_text())
    img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
    pen = np.load(d / f"gt_g{g:g}.npz")["pen"].astype(np.float32)
    x0 = meta["margin_x"]
    x1 = img.shape[1] - x0
    D, fy, fx = prep(img, x0, x1)
    S = float(meta["spacing"])
    return {"name": rec_dir.name, "gain": g, "dir": str(d), "D": D, "fy": fy, "fx": fx, "pen": pen, "S": S, "x0": x0,
            "x1": x1, "W": img.shape[1], "bases": meta["margin_y"] + S * np.arange(meta["n_lines"]), "act": meta["active_line"]}


def split():
    f = OUT / "split.json"
    if f.exists():
        return json.loads(f.read_text())
    recs = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    groups = sorted({r[:8] for r in recs})
    val_g = sorted(np.random.default_rng(20261027).permutation(groups)[:4].tolist())
    sp = {"seed": 20261027, "val_groups": val_g, "train": [r for r in recs if r[:8] not in val_g],
          "val": [r for r in recs if r[:8] in val_g]}
    OUT.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(sp, indent=1))
    return sp


def photometric(c, rng):
    c = c * rng.uniform(0.6, 1.5)
    if rng.random() < 0.5:
        c = c ** rng.uniform(0.7, 1.3)
    if rng.random() < 0.5:
        c = cv2.GaussianBlur(c, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.3:                           # faint stretches
        c = c * cv2.resize(rng.uniform(0.3, 1.0, (1, 12)).astype(np.float32), (c.shape[1], 1), interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.3:                           # smooth background
        c = c + rng.uniform(0, 0.15) * cv2.resize(rng.normal(0, 1, (6, 12)).astype(np.float32), (c.shape[1], c.shape[0]),
                                                  interpolation=cv2.INTER_CUBIC)
    if rng.random() < 0.3:                           # speckle
        n = int(rng.integers(5, 60))
        c[rng.integers(0, c.shape[0], n), rng.integers(0, c.shape[1], n)] += rng.uniform(0.3, 1.2, n).astype(np.float32)
    c = c + rng.normal(0, rng.uniform(0, 0.12), c.shape).astype(np.float32)
    return np.clip(c, 0, 4)


def train_sample(items, idx_tr, i, k, rng):
    it = items[i]
    S, x0, x1 = it["S"], it["x0"], it["x1"]
    sy, sx = rng.uniform(0.9, 1.1), rng.uniform(0.88, 1.08)
    skew, dy = rng.uniform(-0.004, 0.004), rng.uniform(-4, 4)
    xmid = (x0 + x1) / 2
    ncmax = int((x1 - x0) / (2 * sx)) - 1
    u = rng.random()
    if u < 0.25:
        c0 = 0
    elif u < 0.7:                                    # around the quake
        dev = np.nan_to_num(np.abs(it["pen"][it["act"]] - it["bases"][it["act"]]))
        c0 = int(np.clip((np.argmax(dev) - x0) / (2 * sx) - rng.uniform(0.15, 0.85) * WIN, 0, ncmax - WIN))
    else:
        c0 = int(rng.integers(0, ncmax - WIN + 1))
    base = it["bases"][k]
    crop, x = make_crop(it["D"], it["fy"], it["fx"], base, x0, c0, WIN, sy, sx, skew, dy, xmid)
    nl = len(it["bases"])
    for sgn in (-1, 1):                              # composite: another image continues the line grid
        if rng.random() < 0.75:
            jt = items[idx_tr[int(rng.integers(len(idx_tr)))]]
            crop = np.maximum(crop, make_crop(jt["D"], jt["fy"], jt["fx"], base - sgn * nl * S, x0, c0, WIN, sy, sx, skew, dy, xmid)[0])
    crop = photometric(crop, rng)
    pen_k = it["pen"][k]
    u0 = to_bin(pen_k[x0], base, float(x0), sy, skew, dy, xmid) if c0 == 0 else None
    Y = np.stack([gt_at(p, x) for p in it["pen"]])
    ub = to_bin(Y[k], base, x, sy, skew, dy, xmid).astype(np.float32)
    dmin = np.min(np.nan_to_num(np.abs(np.delete(Y, k, 0) - Y[k]), nan=99), axis=0)
    steep = np.abs(np.gradient(np.nan_to_num(ub))) > 3
    w = (1 + 2 * (dmin < S / 2) + steep).astype(np.float32)
    return np.stack([crop, hint_map(u0, WIN)]), ub, w


def make_batch(items, idx_tr, plan, seed):
    rng = np.random.default_rng(seed)
    out = [train_sample(items, idx_tr, i, k, rng) for i, k in plan]
    return tuple(torch.from_numpy(np.stack(v)) for v in zip(*out))


# ---------------------------------------------------------------- model
def cbr(ci, co, k=(3, 3)):
    return [nn.Conv2d(ci, co, k, padding=(k[0] // 2, k[1] // 2), bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True)]


class RowAnchor(nn.Module):
    def __init__(s, d=128, nl=4):
        super().__init__()
        s.e1 = nn.Sequential(*cbr(2, 12), *cbr(12, 12))
        s.e2 = nn.Sequential(*cbr(12, 24), *cbr(24, 24))
        s.e3 = nn.Sequential(*cbr(24, 32), *cbr(32, 32))
        s.e4 = nn.Sequential(*cbr(32, 48, (3, 5)), *cbr(48, 48, (3, 5)))
        s.e5 = nn.Sequential(*cbr(48, 48, (3, 5)))
        s.proj = nn.Linear(48 * 25, d)
        s.pos = nn.Conv1d(d, d, 9, padding=4, groups=d)
        s.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True), nl,
                                     enable_nested_tensor=False)
        s.ctx = nn.Linear(d, 24 * 25)
        s.head = nn.Sequential(nn.Conv2d(48, 24, 1, bias=False), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                               nn.Conv2d(24, 12, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(12, 1, 1))

    def forward(s, x):                               # (B, 2, 400, W) -> logits (B, 200, W)
        f2 = s.e2(F.max_pool2d(s.e1(x), (2, 1)))
        f5 = s.e5(F.max_pool2d(s.e4(F.max_pool2d(s.e3(F.max_pool2d(f2, (2, 1))), (2, 1))), (2, 1)))
        B, C, H, W = f5.shape
        t = s.proj(f5.permute(0, 3, 1, 2).reshape(B, W, C * H)).transpose(1, 2)        # (B, d, W)
        t = F.avg_pool1d(t, 2, 2, ceil_mode=True)
        t = s.tf((t + s.pos(t)).transpose(1, 2)).transpose(1, 2)
        t = F.interpolate(t, size=W, mode="linear", align_corners=False)
        c = s.ctx(t.transpose(1, 2)).reshape(B, W, 24, 25).permute(0, 2, 3, 1)
        c = F.interpolate(c, size=(NB, W), mode="bilinear", align_corners=False)
        return s.head(torch.cat([f2, c], 1))[:, 0]


def soft_ce(logits, ub, w, sigma=1.0):
    valid = torch.isfinite(ub) & (ub >= 0) & (ub <= NB - 1)
    u = torch.where(valid, ub, torch.zeros_like(ub))
    j = torch.arange(NB, device=logits.device, dtype=torch.float32)[None, :, None]
    t = torch.exp(-0.5 * ((j - u[:, None, :]) / sigma) ** 2)
    t = t / t.sum(1, keepdim=True)
    wv = w * valid
    return ((-(t * F.log_softmax(logits, 1)).sum(1)) * wv).sum() / wv.sum().clamp(min=1)


# ---------------------------------------------------------------- inference + decoding
@torch.no_grad()
def infer(model, D, fy, fx, bases, y0s, x0, x1, bs=8):
    """log p over row bins for every line: (n_lines, NB, Wc) float32"""
    Wc = int(math.ceil((x1 - x0) / 2))
    out = []
    for i in range(0, len(bases), bs):
        X = []
        for b, y0 in zip(bases[i:i + bs], y0s[i:i + bs]):
            crop, _ = make_crop(D, fy, fx, b, x0, 0, Wc)
            X.append(np.stack([crop, hint_map(to_bin(y0, b, float(x0)), Wc)]))
        out.append(torch.log_softmax(model(torch.from_numpy(np.stack(X)).to(DEV)), 1).float().cpu().numpy())
    return np.concatenate(out)


def viterbi(logp, logT, u0, alpha=1.0, s0=1.5):
    """max over paths of alpha * sum log p + sum logT[step] (+ start prior at u0) -> (N, Wc) bins"""
    lp = torch.from_numpy(np.ascontiguousarray(logp)).float()
    N, nb, Wc = lp.shape
    j = torch.arange(nb)
    dif = j[None, :] - j[:, None]                                       # [src, dst] = dst - src
    lt = torch.from_numpy(logT).float()
    Tm = torch.where(dif.abs() <= DSTEP, lt[(dif + DSTEP).clamp(0, 2 * DSTEP)], torch.tensor(-1e9))
    u0 = torch.from_numpy(np.nan_to_num(np.asarray(u0, np.float32), nan=-1e4))
    start = torch.where(u0[:, None] > -1e3, -0.5 * ((j[None].float() - u0[:, None]) / s0) ** 2, torch.zeros(1))
    sc = alpha * lp[:, :, 0] + start
    back = torch.zeros((N, nb, Wc), dtype=torch.int16)
    for x in range(1, Wc):
        v, a = (sc[:, :, None] + Tm[None]).max(1)
        sc = v + alpha * lp[:, :, x]
        back[:, :, x] = a.to(torch.int16)
    path = torch.zeros((N, Wc), dtype=torch.long)
    path[:, -1] = sc.argmax(1)
    ar = torch.arange(N)
    for x in range(Wc - 1, 0, -1):
        path[:, x - 1] = back[ar, path[:, x], x].long()
    return path.numpy()


def decode(logp, bases, x0, x1, W, u0=None, logT=None, alpha=1.0):
    """argmax (logT None) or Viterbi bins -> local expectation +-2 bins -> full-res y per column (NaN outside [x0, x1))"""
    N, nb, Wc = logp.shape
    path = logp.argmax(1) if logT is None else viterbi(logp, logT, u0, alpha)
    idx = np.clip(path[:, None, :] + np.arange(-2, 3)[None, :, None], 0, nb - 1)
    pw = np.exp(np.take_along_axis(logp, idx, 1))
    u = (pw * idx).sum(1) / np.clip(pw.sum(1), 1e-12, None)
    yc = from_bin(u, np.asarray(bases)[:, None])
    xc = x0 + 2 * np.arange(Wc) + 0.5
    est = np.full((N, W), np.nan)
    xs = np.arange(x0, x1)
    for k in range(N):
        est[k, x0:x1] = np.interp(xs, xc, yc[k])
    return est


def load_model(tag="best"):
    ck = torch.load(OUT / f"{tag}.pt", map_location="cpu", weights_only=False)
    m = RowAnchor().to(DEV)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def robust(logT, eps):
    """heavy-tailed transition prior: (1 - eps) * empirical + eps * uniform over |step| <= DSTEP"""
    return np.log((1 - eps) * np.exp(logT) + eps / len(logT)).astype(np.float32) if eps > 0 else logT


def transition_prior(items, idx):
    """log P(bin step per half-res column) from the training GT (Laplace smoothed)"""
    h = np.zeros(2 * DSTEP + 1)
    for i in idx:
        it = items[i]
        xc = it["x0"] + 2 * np.arange(int(math.ceil((it["x1"] - it["x0"]) / 2))) + 0.5
        for k, b in enumerate(it["bases"]):
            d = np.diff(to_bin(gt_at(it["pen"][k], xc), b, xc))
            d = np.round(d[np.isfinite(d)]).astype(int)
            h += np.bincount(np.clip(d, -DSTEP, DSTEP) + DSTEP, minlength=2 * DSTEP + 1)
    return np.log((h + 1) / (h + 1).sum()).astype(np.float32)


# ---------------------------------------------------------------- evaluation on synthetic items
def eval_items(model, items, configs, logT, B_of=None, keep=None):
    """configs: {name: alpha or None (argmax)} -> per-image rows with quake / whole fail (track_joint.score)"""
    rows = []
    for it in items:
        y0s = it["pen"][:, it["x0"]]
        lp = infer(model, it["D"], it["fy"], it["fx"], it["bases"], y0s, it["x0"], it["x1"])
        u0 = [to_bin(y, b, float(it["x0"])) for y, b in zip(y0s, it["bases"])]
        row = {"name": it["name"], "gain": it["gain"], "act": int(it["act"])}
        ests = {}
        for name, a in configs.items():
            al, eps = (a, 0.0) if a is None or np.isscalar(a) else a
            ests[name] = decode(lp, it["bases"], it["x0"], it["x1"], it["W"], u0, None if a is None else robust(logT, eps), al or 1.0)
        if B_of is not None:
            ests.update(B_of(it))
        n = len(it["bases"])
        for name, est in ests.items():
            row[name] = {"quake": tj_score(est, it["pen"], [it["act"]], it["x0"], it["x1"], it["S"])[0],
                         "whole": tj_score(est, it["pen"], range(n), it["x0"], it["x1"], it["S"])[0],
                         "per_line": [tj_score(est, it["pen"], [k], it["x0"], it["x1"], it["S"])[0] for k in range(n)],
                         "quake_fail_anal": fail_anal(est[it["act"], it["x0"]:it["x1"]], it["pen"][it["act"], it["x0"]:it["x1"]], it["S"])}
        if keep is not None:
            keep[(it["name"], it["gain"])] = {k: v for k, v in ests.items()}
        rows.append(row)
    return rows


def fail_anal(e, g, S):
    """failed columns of one line split by kind: inside fail runs >= 100 columns (identity lost / following another
    trace), at steep GT (|dy/dx| > S/4 px per column: near-vertical strokes), rest"""
    ok = ~np.isnan(g)
    fail = ok & (np.isnan(e) | (np.abs(np.nan_to_num(e, nan=1e9) - g) > S / 2))
    nf = int(fail.sum())
    if nf == 0:
        return {"n_fail": 0, "n_drawn": int(ok.sum()), "long_run_frac": 0.0, "steep_frac": 0.0, "max_run": 0}
    edges = np.flatnonzero(np.diff(np.r_[0, fail.astype(int), 0]))
    runs = edges[1::2] - edges[::2]
    long_cols = np.zeros_like(fail)
    for s, L in zip(edges[::2], runs):
        if L >= 100:
            long_cols[s:s + L] = True
    steep = np.abs(np.gradient(np.nan_to_num(g))) > S / 4
    return {"n_fail": nf, "n_drawn": int(ok.sum()), "long_run_frac": round(float((fail & long_cols).sum() / nf), 3),
            "steep_frac": round(float((fail & steep & ~long_cols).sum() / nf), 3), "max_run": int(runs.max())}


def summarize(rows, methods):
    out = {}
    for g in sorted({r["gain"] for r in rows}) + ["all"]:
        rr = [r for r in rows if g == "all" or r["gain"] == g]
        out[str(g)] = {m: {"quake": round(float(np.mean([r[m]["quake"] for r in rr])), 1),
                           "whole": round(float(np.mean([r[m]["whole"] for r in rr])), 1)} for m in methods if m in rr[0]}
        out[str(g)]["n_images"] = len(rr)
    return out


# ---------------------------------------------------------------- commands
def load_set(names, gains, root):
    return [load_item(root / n, g) for n in names for g in gains]


def cmd_cacheB(which):
    from arbiter import cache_proposals, proposals
    if which == "holdout":
        cache_proposals(HOLD_DIR, HOLD_CACHE, [("film", g) for g in G_HOLD])
        return
    cdir = OUT / "cacheB_val"
    cdir.mkdir(parents=True, exist_ok=True)
    for n in split()["val"]:
        for g in G_TRAIN:
            f = cdir / f"{n}_film_g{g:g}.npz"
            if f.exists():
                continue
            d = TRAIN_DIR / n / "film"
            meta = json.loads((d / "meta.json").read_text())
            img = cv2.imread(str(d / f"heli_g{g:g}.png"), 0)
            pen = np.load(d / f"gt_g{g:g}.npz")["pen"]
            x0 = meta["margin_x"]
            A, B, _ = proposals(img, pen[:, x0], x0, img.shape[1] - x0, meta["spacing"])
            np.savez_compressed(f, A=A, B=B)
            print("cached", f.name, flush=True)


def cmd_train(minutes, bench=False):
    sp = split()
    t0 = time.time()
    items = load_set(sp["train"], G_TRAIN, TRAIN_DIR)
    vitems = load_set(sp["val"], G_TRAIN, TRAIN_DIR)
    idx_tr = list(range(len(items)))
    logT = transition_prior(items, idx_tr)
    np.save(OUT / "logT.npy", logT)
    print(f"loaded {len(items)} train / {len(vitems)} val images in {time.time() - t0:.0f}s", flush=True)
    base_plan = [(i, k) for i, it in enumerate(items) for k in range(len(it["bases"]))]
    base_plan += [(i, it["act"]) for i, it in enumerate(items)] * 2        # quake line oversampled x3
    torch.manual_seed(0)
    model = RowAnchor().to(DEV)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    budget = minutes * 60.0
    rng = np.random.default_rng(1)
    pool = ThreadPoolExecutor(2)
    log, best, step, seed = [], 1e9, 0, 100
    hist_path = OUT / "train_log.json"

    def plans():
        while True:
            p = rng.permutation(len(base_plan))
            for s in range(0, len(p) - BATCH + 1, BATCH):
                yield [base_plan[q] for q in p[s:s + BATCH]]
    gen = plans()
    fut = [pool.submit(make_batch, items, idx_tr, next(gen), seed + q) for q in range(2)]
    seed += 2
    tstart, next_val, run_loss = time.time(), 300.0, []
    while True:
        el = time.time() - tstart
        if el > budget or (bench and step >= 12):
            break
        frac = el / budget
        lr = 2e-3 * min(1.0, frac / 0.03) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
        for g_ in opt.param_groups:
            g_["lr"] = max(lr, 2e-5)
        X, U, Wt = fut.pop(0).result()
        fut.append(pool.submit(make_batch, items, idx_tr, next(gen), seed))
        seed += 1
        model.train()
        loss = soft_ce(model(X.to(DEV)), U.to(DEV), Wt.to(DEV))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        run_loss.append(float(loss.detach()))
        step += 1
        if bench and step in (2, 12):
            print(f"bench step {step} at {time.time() - tstart:.1f}s loss {run_loss[-1]:.3f}", flush=True)
        if not bench and el > next_val:
            next_val += 360.0
            model.eval()
            rows = eval_items(model, vitems, {"argmax": None}, logT)
            sm = summarize(rows, ["argmax"])["all"]["argmax"]
            crit = (sm["quake"] + sm["whole"]) / 2
            rec = {"step": step, "minutes": round(el / 60, 2), "train_loss": round(float(np.mean(run_loss[-200:])), 4),
                   "lr": lr, "val_quake_fail_argmax": sm["quake"], "val_whole_fail_argmax": sm["whole"], "crit": crit}
            run_loss = run_loss[-200:]
            log.append(rec)
            print(json.dumps(rec), flush=True)
            torch.save({"model": model.state_dict(), "step": step, "rec": rec}, OUT / "last.pt")
            if crit < best:
                best = crit
                torch.save({"model": model.state_dict(), "step": step, "rec": rec}, OUT / "best.pt")
            hist_path.write_text(json.dumps(log, indent=1))
    if not bench:
        model.eval()
        rows = eval_items(model, vitems, {"argmax": None}, logT)
        sm = summarize(rows, ["argmax"])["all"]["argmax"]
        crit = (sm["quake"] + sm["whole"]) / 2
        rec = {"step": step, "minutes": round((time.time() - tstart) / 60, 2), "train_loss": round(float(np.mean(run_loss[-200:])), 4),
               "lr": 0.0, "val_quake_fail_argmax": sm["quake"], "val_whole_fail_argmax": sm["whole"], "crit": crit, "final": True}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"model": model.state_dict(), "step": step, "rec": rec}, OUT / "last.pt")
        if crit < best:
            torch.save({"model": model.state_dict(), "step": step, "rec": rec}, OUT / "best.pt")
        hist_path.write_text(json.dumps(log, indent=1))
    pool.shutdown(wait=False, cancel_futures=True)


ALPHAS = {"argmax": None, "viterbi_a0.5": 0.5, "viterbi_a1": 1.0, "viterbi_a2": 2.0, "viterbi_a4": 4.0,
          "viterbi_a1_eps0.1": (1.0, 0.1), "viterbi_a2_eps0.1": (2.0, 0.1)}


def cmd_val():
    sp = split()
    model, ck = load_model("best")
    logT = np.load(OUT / "logT.npy")
    vitems = load_set(sp["val"], G_TRAIN, TRAIN_DIR)

    def B_of(it):
        z = np.load(OUT / "cacheB_val" / f"{it['name']}_film_g{it['gain']:g}.npz")
        return {"A": z["A"], "B": z["B"]}
    t0 = time.time()
    rows = eval_items(model, vitems, ALPHAS, logT, B_of)
    res = {"checkpoint": ck["rec"], "split": sp, "rows": rows, "summary": summarize(rows, list(ALPHAS) + ["A", "B"]),
           "runtime_s": round(time.time() - t0, 1)}
    (OUT / "val_results.json").write_text(json.dumps(res, indent=1))
    for g, v in res["summary"].items():
        print(g, json.dumps(v))


def cmd_holdout():
    """ONE run on film_holdout3 with the frozen method (config chosen on validation in frozen.json)"""
    fz = json.loads((OUT / "frozen.json").read_text())
    model, ck = load_model("best")
    logT = robust(np.load(OUT / "logT.npy"), fz.get("eps", 0.0))
    names = sorted(p.name for p in HOLD_DIR.iterdir() if p.is_dir())
    items = load_set(names, G_HOLD, HOLD_DIR)

    def B_of(it):
        z = np.load(HOLD_CACHE / f"{it['name']}_film_g{it['gain']:g}.npz")
        return {"A": z["A"], "B": z["B"]}
    cfg = {"argmax": None, fz["method"]: (fz["alpha"], 0.0)}   # logT already made robust with fz["eps"]
    rows = eval_items(model, items, cfg, logT, B_of)
    res = {"frozen": fz, "checkpoint": ck["rec"], "rows": rows, "summary": summarize(rows, list(cfg) + ["A", "B"])}
    (OUT / "holdout3_results.json").write_text(json.dumps(res, indent=1))
    for g, v in res["summary"].items():
        print(g, json.dumps(v))


def cmd_real():
    import real_paired_eval as rpe
    from obspy import UTCDateTime
    fz = json.loads((OUT / "frozen.json").read_text())
    model, ck = load_model("best")
    logT = robust(np.load(OUT / "logT.npy"), fz.get("eps", 0.0))
    allres = {}
    for name in ("KIP", "MAT"):
        cfg = rpe.SCANS[name]
        res = json.loads((ROOT / "out" / "real_paired" / f"{name}_results.json").read_text())
        img_s = cv2.imread(str(ROOT / "out" / "real_paired" / name / f"{name}_rescaled.png"), 0)
        z = np.load(ROOT / "out" / "real_paired" / name / f"{name}_paths.npz")
        bases, y0 = z["bases"], z["y0"]
        L = res["layout"]
        x0, x1, xa, S = L["x0"], L["x1"], L["xa"], rpe.TARGET_S
        W = img_s.shape[1]
        D, fy, fx = prep(img_s, x0, x1)
        lp = infer(model, D, fy, fx, bases, y0, x0, x1)
        u0 = [to_bin(y, b, float(x0)) for y, b in zip(y0, bases)]
        paths = {"RA_viterbi": decode(lp, bases, x0, x1, W, u0, logT, fz["alpha"]), "RA_argmax": decode(lp, bases, x0, x1, W, u0),
                 "B_recomputed": z["B"]}
        np.savez_compressed(OUT / f"real_{name}_paths.npz", **paths)
        al, a_glob = res["alignment"]["chosen_dp_fit"], res["amplitude"]["global_scale_px_per_m"]
        t_on, t_off = UTCDateTime(cfg["on"]), UTCDateTime(cfg["off"])
        t_ref, ref, _ = rpe.reference(cfg, t_on)
        cat = {r["line"]: r["category"] for r in res["lines"]}
        per = {m: {} for m in paths}
        for k in cat:
            idx, _, _ = rpe.sample_path(paths["B_recomputed"], k, x0, x1, xa, al, t_ref, 0.0, float(t_off - t_on))
            tk, rf = t_ref[idx], ref[idx]
            for m, P in paths.items():
                _, y, _ = rpe.sample_path(P, k, x0, x1, xa, al, t_ref, 0.0, float(t_off - t_on))
                per[m][k] = rpe.line_metrics(-(y - bases[k]) * al["sign"], rf, tk, S, a_glob)
        keys = ("r_band", "in_0.1S", "lost_0.5S", "in_0.1S_global", "lost_0.5S_global")
        summ = {}
        for c in ("quake", "neighbour", "other_active", "quiet"):
            ks = [k for k, v in cat.items() if v == c]
            if not ks:
                continue
            summ[c] = {"lines": ks}
            for m in paths:
                summ[c][m] = {key: round(float(np.mean([per[m][k][key] for k in ks])), 3) for key in keys}
            for m in ("A", "B", "judge", "DP", "DP38"):
                summ[c][m + "_published"] = {key: res["summary"][c][m][key] for key in keys}
        allres[name] = {"summary": summ, "per_line": {m: {str(k): v for k, v in d.items()} for m, d in per.items()},
                        "main_shock_line": res["main_shock_line"]}
        for c, v in summ.items():
            print(name, c, v["lines"], " | ".join(f"{m}: r {v[m]['r_band']:+.2f} in {v[m]['in_0.1S']:.2f} lost {v[m]['lost_0.5S']:.2f}"
                                                  for m in ("RA_viterbi", "RA_argmax", "B_recomputed", "B_published", "DP_published")), flush=True)
    (OUT / "real_results.json").write_text(json.dumps({"frozen": fz, "checkpoint": ck["rec"], "scans": allres}, indent=1))


def cmd_freeze():
    """pick the decoding on validation only (before any holdout3 / real run): the Viterbi variant with the lowest
    mean(quake fail, whole fail); argmax stays as the ablation"""
    v = json.loads((OUT / "val_results.json").read_text())
    sm = v["summary"]["all"]
    crit = {k: (sm[k]["quake"] + sm[k]["whole"]) / 2 for k in sm if k.startswith("viterbi")}
    best = min(crit, key=crit.get)
    a = ALPHAS[best]
    alpha, eps = (a, 0.0) if np.isscalar(a) else a
    fz = {"method": "RA_viterbi", "config": best, "alpha": float(alpha), "eps": float(eps), "checkpoint_step": v["checkpoint"]["step"],
          "rule": "Viterbi variant with the lowest mean(quake fail, whole fail) on the validation groups; argmax kept as ablation",
          "val_crit": crit, "val_argmax_crit": (sm["argmax"]["quake"] + sm["argmax"]["whole"]) / 2}
    (OUT / "frozen.json").write_text(json.dumps(fz, indent=1))
    print(json.dumps(fz))


COLORS = {"GT": (0, 170, 0), "RA_viterbi": (0, 0, 255), "RA_argmax": (200, 0, 200), "B": (255, 120, 0), "DP": (0, 140, 255)}


def overlay(img, lines_draw, ests, c0, c1, r0, r1, path, scale=2):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for name, P in ests.items():
        for k in lines_draw:
            pts = np.array([(x, P[k, x]) for x in range(c0, c1) if np.isfinite(P[k, x])], np.int32)
            if len(pts):
                cv2.polylines(vis, [pts], False, COLORS[name], 1, cv2.LINE_AA)
    tile = vis[max(0, r0):min(img.shape[0], r1), c0:c1]
    cv2.imwrite(str(path), cv2.resize(tile, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))


def cmd_figs(which):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fd = OUT / "figs"
    fd.mkdir(exist_ok=True)
    if which in ("all", "curve") and (OUT / "train_log.json").exists():
        log = json.loads((OUT / "train_log.json").read_text())
        mins = [r["minutes"] for r in log]
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        ax[0].plot(mins, [r["train_loss"] for r in log], "o-")
        ax[0].set_xlabel("minutes"); ax[0].set_ylabel("train soft-CE (last 200 steps)")
        ax[1].plot(mins, [r["val_quake_fail_argmax"] for r in log], "o-", label="val quake line")
        ax[1].plot(mins, [r["val_whole_fail_argmax"] for r in log], "s-", label="val whole image")
        ax[1].set_xlabel("minutes"); ax[1].set_ylabel("fail % (argmax decoding)"); ax[1].legend()
        fig.tight_layout(); fig.savefig(fd / "train_curve.png", dpi=110); plt.close(fig)
    fz = json.loads((OUT / "frozen.json").read_text()) if (OUT / "frozen.json").exists() else {"alpha": 1.0}
    if which in ("all", "val", "holdout"):
        model, _ = load_model("best")
        logT = robust(np.load(OUT / "logT.npy"), fz.get("eps", 0.0))
        sets = []
        if which in ("all", "val"):
            sp = split()
            sets.append(("val", TRAIN_DIR, sp["val"], (4, 8), OUT / "cacheB_val"))
        if which in ("all", "holdout") and (OUT / "holdout3_results.json").exists():
            names = sorted({r["name"] for r in json.loads((OUT / "holdout3_results.json").read_text())["rows"]})
            sets.append(("holdout3", HOLD_DIR, names, (4, 8), HOLD_CACHE))
        for tag, root, names, gains, cdir in sets:
            for g in gains:
                its = [load_item(root / n, g) for n in names]
                its = sorted(its, key=lambda it: -np.nanmax(np.abs(it["pen"][it["act"]] - it["bases"][it["act"]])))[:3]
                for it in its:
                    y0s = it["pen"][:, it["x0"]]
                    lp = infer(model, it["D"], it["fy"], it["fx"], it["bases"], y0s, it["x0"], it["x1"])
                    u0 = [to_bin(y, b, float(it["x0"])) for y, b in zip(y0s, it["bases"])]
                    est = decode(lp, it["bases"], it["x0"], it["x1"], it["W"], u0, logT, fz["alpha"])
                    B = np.load(cdir / f"{it['name']}_film_g{g:g}.npz")["B"]
                    img = cv2.imread(str(Path(it["dir"]) / f"heli_g{g:g}.png"), 0)
                    k = it["act"]
                    xc = int(np.nanargmax(np.abs(it["pen"][k] - it["bases"][k])))
                    c0 = int(np.clip(xc - 450, it["x0"], it["x1"] - 900))
                    overlay(img, [k], {"GT": it["pen"], "B": B, "RA_viterbi": est}, c0, c0 + 900, 0, img.shape[0],
                            fd / f"{tag}_g{g}_{it['name']}_quake.png")
    if which in ("all", "real") and (OUT / "real_results.json").exists():
        import real_paired_eval as rpe
        from obspy import UTCDateTime
        rr = json.loads((OUT / "real_results.json").read_text())["scans"]
        for name in ("KIP", "MAT"):
            res = json.loads((ROOT / "out" / "real_paired" / f"{name}_results.json").read_text())
            img_s = cv2.imread(str(ROOT / "out" / "real_paired" / name / f"{name}_rescaled.png"), 0)
            z = np.load(ROOT / "out" / "real_paired" / name / f"{name}_paths.npz")
            P = np.load(OUT / f"real_{name}_paths.npz")
            km = rr[name]["main_shock_line"]
            bases = z["bases"]
            L = res["layout"]
            dev = np.nan_to_num(np.abs(P["RA_viterbi"][km] - bases[km]))
            xc = int(np.argmax(dev))
            c0 = int(np.clip(xc - 450, L["x0"], L["x1"] - 900))
            for i, cc in enumerate((c0, max(L["x0"], c0 - 900))):
                overlay(img_s, range(max(0, km - 2), min(len(bases), km + 3)), {"B": z["B"], "DP": z["DP"], "RA_viterbi": P["RA_viterbi"]},
                        cc, cc + 900, int(bases[km] - 9 * 38), int(bases[km] + 9 * 38), fd / f"real_{name}_line{km}_overlay{i}.png", scale=1)
            cfg = rpe.SCANS[name]
            t_on, t_off = UTCDateTime(cfg["on"]), UTCDateTime(cfg["off"])
            t_ref, ref, _ = rpe.reference(cfg, t_on)
            al, a_glob = res["alignment"]["chosen_dp_fit"], res["amplitude"]["global_scale_px_per_m"]
            fig, axs = plt.subplots(2, 1, figsize=(16, 6.5))
            for ax, k in zip(axs, (km, km + 1)):
                idx, _, _ = rpe.sample_path(z["B"], k, L["x0"], L["x1"], L["xa"], al, t_ref, 0.0, float(t_off - t_on))
                tk = (t_ref[idx] - t_ref[idx][0]) / 60
                ax.plot(tk, a_glob * rpe.bandpass(ref[idx]) / 38, "k", lw=1.3, label="reference x global scale")
                for m, Q, col in (("B", z["B"], "tab:blue"), ("DP (alignment tool)", z["DP"], "tab:orange"), ("RA viterbi", P["RA_viterbi"], "tab:red")):
                    _, y, _ = rpe.sample_path(Q, k, L["x0"], L["x1"], L["xa"], al, t_ref, 0.0, float(t_off - t_on))
                    ax.plot(tk, rpe.bandpass(-(y - bases[k]) * al["sign"]) / 38, color=col, lw=0.7, label=m)
                ax.set_title(f"{name} line {k} ({'main shock' if k == km else 'next line'}), band {rpe.BAND[0]:.4f}-{rpe.BAND[1]} Hz")
                ax.set_xlabel("minutes"); ax.set_ylabel("line spacings"); ax.legend(loc="upper right", fontsize=7)
            fig.tight_layout(); fig.savefig(fd / f"real_{name}_timeseries.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "split":
        print(json.dumps(split(), indent=1))
    elif cmd == "cacheB":
        cmd_cacheB(sys.argv[2])
    elif cmd == "bench":
        cmd_train(10, bench=True)
    elif cmd == "train":
        cmd_train(float(sys.argv[2]) if len(sys.argv) > 2 else 50.0)
    elif cmd == "val":
        cmd_val()
    elif cmd == "freeze":
        cmd_freeze()
    elif cmd == "holdout":
        cmd_holdout()
    elif cmd == "real":
        cmd_real()
    elif cmd == "figs":
        cmd_figs(sys.argv[2] if len(sys.argv) > 2 else "all")
