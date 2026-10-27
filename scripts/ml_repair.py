"""Learned repair of digitized helicorder lines after extraction (docs/RESULTS.md).

usage: python -I scripts/ml_repair.py cache [workers]       B for out/film_train, gains 1,2,3,4,6,8 (arbiter.cache_proposals)
       python -I scripts/ml_repair.py train seq|seqp|seq1|spec   train one model (fixed schedule, MPS)
       python -I scripts/ml_repair.py dev                   validation groups: B / flat / models, per gain + bound
       python -I scripts/ml_repair.py holdout               ONE run on out/film_holdout3 with the frozen models
       python -I scripts/ml_repair.py real KIP|MAT          real paired eval (real_paired_eval's alignment + metrics)
       python -I scripts/ml_repair.py fig                   validation example figure (quake line, gains 2 and 6)

Proposer B (joint tracker) gives one path per line; on the quake line it swaps identity with neighbours and loses
segments. A repair model maps B's paths to the true series of one target line. Units: amplitude in line spacings,
y-up, relative to the layout baseline of the line (synthetic: margin_y + k S; real: real_paired_eval's baselines).
- seq  (primary): input = every line's B series within +-R lines of the target, as position relative to the TARGET's
  baseline (a_j - r for line j = k + r), plus an existence/NaN mask per slot; the target index is the centre slot.
  Conv stem -> /4 -> bidirectional Transformer (conv positional encoding) -> x4 + skip -> pointer head: at every
  column a softmax over the slots picks which B line carries the target's trace, plus a free residual
  (y = sum_j p_j pos_j + delta). Initialised to the identity (select own line, delta 0). Swaps can be undone exactly by
  pointing at the neighbour; content that no line traced can only be guessed through delta.
- seqp (added after seq/seq1/spec plateaued at the flat baseline): seq + a baseline candidate slot (value 0), and the
  pointer is supervised with cross-entropy to the GT-oracle slot (nearest B line within S/2, else the baseline).
- seq1 (ablation): the same network with R = 0 (own series only) -> what neighbours add.
- spec (the user's question): STFT of the target's own B series (complex, real+imag channels) -> 2D U-Net (residual)
  -> iSTFT; loss on the waveform and the STFT magnitude.
Target = GT `sig` (signal-only y) in the same units; synthetic fail is scored against GT `pen` as everywhere else.
Train/val split by event group (first 8 chars), ~20% of groups held out; fixed schedule, no early stopping.
References in every table: flat (the layout baseline itself, amplitude 0) and oracle (GT-chosen nearest B line within
S/2, else the baseline; a bound for any re-assignment model, not a method). Result (docs/RESULTS.md): every model ends at
the flat baseline's fail %, quake-line r ~0.1 vs oracle ~0.8; the fail metric rewards flattening, so r is primary.
"""
import json
import os
import random
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from arbiter import cache_proposals          # noqa: E402
from track_joint import score                # noqa: E402

OUT = ROOT / "out" / "ml" / "repair"
TRAIN_BENCH, TRAIN_CACHE = ROOT / "out" / "film_train", ROOT / "out" / "cache_film_train"
HOLD_BENCH = ROOT / "out" / "film_holdout3"
HOLD_CACHE_SHARED, HOLD_CACHE_OWN = ROOT / "out" / "cache_film_holdout3", OUT / "cache_b"
TRAIN_GAINS, HOLD_GAINS = (1, 2, 3, 4, 6, 8), (1, 2, 4, 6, 8)
R_SEQ = 7                 # neighbour slots each side (8-line training records: every line is in reach)
CROP, BATCH = 2048, 16
STEPS = {"seq": 12000, "seq1": 6000, "spec": 6000, "seqp": 5000}
CE_W = 0.3                # seqp: weight of the pointer cross-entropy (oracle slot) next to the L1
N_FFT, HOP = 126, 16      # spectrogram: 64 bins (0-0.5 Hz at 1 sample/s), 16 s hop
VAL_FRAC, SPLIT_SEED = 0.2, 0


# ---------------------------------------------------------------- B cache
def _cache_one(args):
    """B (and A, ink) for one record via arbiter.cache_proposals on a one-record bench (symlink): workers never touch
    the same file and every file is written once"""
    rec, cache, gains = args
    with tempfile.TemporaryDirectory(dir=OUT) as tmp:
        (Path(tmp) / rec.name).symlink_to(rec, target_is_directory=True)
        cache_proposals(Path(tmp), cache, [("film", g) for g in gains])
    return rec.name


def build_cache(bench, cache, gains, workers=6):
    cache.mkdir(parents=True, exist_ok=True)
    recs = sorted(p for p in bench.iterdir() if p.is_dir())
    todo = [r for r in recs if not all((cache / f"{r.name}_film_g{g:g}.npz").exists() for g in gains)]
    print(f"cache {cache}: {len(recs) - len(todo)} records done, {len(todo)} to go ({workers} workers)", flush=True)
    with Pool(workers) as pool:
        for i, name in enumerate(pool.imap_unordered(_cache_one, [(r, cache, gains) for r in todo])):
            print(f"  [{i + 1}/{len(todo)}] {name}", flush=True)


# ---------------------------------------------------------------- data
def load_items(bench, cache, gains):
    """one item per record x gain: amplitudes (spacings, y-up, rel. to the layout baseline) over [x0, x1)"""
    items = []
    for rec in sorted(p for p in bench.iterdir() if p.is_dir()):
        d = rec / "film"
        meta = json.loads((d / "meta.json").read_text())
        S, x0, n = float(meta["spacing"]), meta["margin_x"], meta["n_lines"]
        base = meta["margin_y"] + S * np.arange(n)
        for g in gains:
            gt = np.load(d / f"gt_g{g:g}.npz")
            B = np.load(cache / f"{rec.name}_film_g{g:g}.npz")["B"]
            x1 = gt["pen"].shape[1] - x0
            items.append({"record": rec.name, "group": rec.name[:8], "gain": g, "S": S, "x0": x0, "x1": x1,
                          "active": meta["active_line"], "base": base, "B": B, "pen": gt["pen"],
                          "aB": ((base[:, None] - B[:, x0:x1]) / S).astype(np.float32),
                          "apen": ((base[:, None] - gt["pen"][:, x0:x1]) / S).astype(np.float32),
                          "asig": ((base[:, None] - gt["sig"][:, x0:x1]) / S).astype(np.float32)})
    return items


def split_groups(items):
    groups = sorted({it["group"] for it in items})
    random.Random(SPLIT_SEED).shuffle(groups)
    n_rec = len({it["record"] for it in items})
    val, cnt = [], 0
    for g in groups:
        if cnt >= VAL_FRAC * n_rec:
            break
        val.append(g)
        cnt += len({it["record"] for it in items if it["group"] == g})
    return sorted(val)


def seq_input(aB, k, R):
    """slot r = line k + r: position relative to line k's baseline (a_j - r), 0 where missing; mask 1 where present"""
    n, T = aB.shape
    V, M = np.zeros((2 * R + 1, T), np.float32), np.zeros((2 * R + 1, T), np.float32)
    for i, r in enumerate(range(-R, R + 1)):
        j = k + r
        if 0 <= j < n:
            ok = np.isfinite(aB[j])
            V[i, ok], M[i, ok] = aB[j, ok] - r, 1.0
    return V, M


# ---------------------------------------------------------------- models
def build_model(kind):
    import torch
    import torch.nn as nn

    class SeqRepair(nn.Module):
        def __init__(self, R, d=128, layers=4, heads=4, c=64, base_slot=False):
            super().__init__()
            ns = 2 * R + 1
            self.base_slot = base_slot                                     # seqp: extra candidate = the baseline (0)
            self.stem = nn.Sequential(nn.Conv1d(2 * ns, c, 7, padding=3), nn.GELU(), nn.Conv1d(c, c, 7, padding=3), nn.GELU())
            self.down = nn.Sequential(nn.Conv1d(c, d, 4, stride=2, padding=1), nn.GELU(), nn.Conv1d(d, d, 4, stride=2, padding=1))
            self.pos = nn.Conv1d(d, d, 31, padding=15, groups=8)          # relative position (wav2vec2-style)
            self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, 0.1, activation="gelu",
                                                                       batch_first=True, norm_first=True), layers)
            self.norm = nn.LayerNorm(d)
            self.up = nn.ConvTranspose1d(d, c, 4, stride=4)
            self.dec = nn.Sequential(nn.Conv1d(2 * c, c, 7, padding=3), nn.GELU(), nn.Conv1d(c, c, 7, padding=3), nn.GELU())
            self.sel, self.res = nn.Conv1d(c, ns + base_slot, 1), nn.Conv1d(c, 1, 1)
            for m in (self.sel, self.res):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            with torch.no_grad():
                self.sel.bias[R] = 6.0                                     # start near the identity (own line, p ~ 0.97)

        def forward(self, v, m):                                           # (batch, slots, T)
            T = v.shape[-1]
            pad = (-T) % 4
            if pad:
                v, m = nn.functional.pad(v, (0, pad)), nn.functional.pad(m, (0, pad))
            h0 = self.stem(torch.cat([0.25 * v, m], 1))
            h = self.down(h0)
            h = h + self.pos(h)
            h = self.norm(self.tf(h.transpose(1, 2))).transpose(1, 2)
            h = self.dec(torch.cat([self.up(h), h0], 1))
            if self.base_slot:
                v, m = torch.cat([v, torch.zeros_like(v[:, :1])], 1), torch.cat([m, torch.ones_like(m[:, :1])], 1)
            p = torch.softmax(self.sel(h).masked_fill(m < 0.5, -1e4), 1)
            return ((p * v).sum(1) + self.res(h)[:, 0])[..., :T], p[..., :T]

    def block(ci, co):
        return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                             nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())

    class SpecUNet(nn.Module):
        def __init__(self, ch=(32, 64, 128, 256)):
            super().__init__()
            self.enc = nn.ModuleList([block(2 if i == 0 else ch[i - 1], c) for i, c in enumerate(ch)])
            self.upc = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(len(ch) - 1, 0, -1)])
            self.dec = nn.ModuleList([block(2 * ch[i - 1], ch[i - 1]) for i in range(len(ch) - 1, 0, -1)])
            self.out = nn.Conv2d(ch[0], 2, 1)
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)  # start as the identity

        def spec(self, x):
            w = torch.hann_window(N_FFT, device=x.device)
            return torch.stft(x, N_FFT, HOP, window=w, return_complex=True, center=True)

        def forward(self, x):                                              # (batch, T) own amplitude series
            T = x.shape[-1]
            Z = self.spec(x)                                               # (batch, 64, frames)
            z = 0.25 * torch.stack([Z.real, Z.imag], 1)
            nf = z.shape[-1]
            z = nn.functional.pad(z, (0, (-nf) % 8))
            hs, h = [], z
            for i, e in enumerate(self.enc):
                h = e(h)
                if i < len(self.enc) - 1:
                    hs.append(h)
                    h = nn.functional.max_pool2d(h, 2)
            for u, dcd in zip(self.upc, self.dec):
                h = dcd(torch.cat([u(h), hs.pop()], 1))
            o = (z + self.out(h))[..., :nf] / 0.25
            w = torch.hann_window(N_FFT, device=x.device)
            return torch.istft(torch.complex(o[:, 0], o[:, 1]), N_FFT, HOP, window=w, center=True, length=T), Z

    return SpecUNet() if kind == "spec" else SeqRepair(0 if kind == "seq1" else R_SEQ, base_slot=kind == "seqp")


def device():
    import torch
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def run_model(model, kind, aB, dev, lines=None):
    """repaired amplitude for every (or the given) line of one record: (n, T)"""
    import torch
    n, T = aB.shape
    lines = range(n) if lines is None else lines
    out = np.full((n, T), np.nan, np.float32)
    with torch.no_grad():
        if kind == "spec":
            x = torch.tensor(np.nan_to_num(aB[list(lines)]), device=dev)
            y, _ = model(x)
        else:
            R = 0 if kind == "seq1" else R_SEQ
            VM = [seq_input(aB, k, R) for k in lines]
            y, _ = model(torch.tensor(np.stack([v for v, _ in VM]), device=dev), torch.tensor(np.stack([m for _, m in VM]), device=dev))
        out[list(lines)] = y.float().cpu().numpy()
    return out


# ---------------------------------------------------------------- training
def train(kind):
    import torch
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    items = load_items(TRAIN_BENCH, TRAIN_CACHE, TRAIN_GAINS)
    val_groups = split_groups(items)
    tr = [it for it in items if it["group"] not in val_groups]
    va = [it for it in items if it["group"] in val_groups]
    # sample (item, line) with weight 0.25 + B's fail fraction on that line: the errors live on few lines
    pairs, wts = [], []
    for i, it in enumerate(tr):
        for k in range(it["aB"].shape[0]):
            pairs.append((i, k))
            wts.append(0.25 + float(np.mean(np.abs(it["aB"][k] - it["apen"][k]) > 0.5)))
    wts = np.array(wts) / np.sum(wts)
    dev = device()
    model = build_model(kind).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    steps = STEPS[kind]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4 if kind != "spec" else 1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps, pct_start=0.05)
    R = 0 if kind == "seq1" else R_SEQ
    log = open(OUT / f"train_{kind}.log", "w")
    print(f"[{kind}] {nparam / 1e6:.2f} M params; train {len(tr)} items ({len({it['record'] for it in tr})} records), "
          f"val groups {val_groups} ({len(va)} items); {steps} steps x batch {BATCH} x crop {CROP} on {dev}", file=log, flush=True)
    t0, run = time.time(), []
    for step in range(steps):
        sel = rng.choice(len(pairs), BATCH, p=wts)
        V, M, Y, X1, C = [], [], [], [], []
        for s in sel:
            i, k = pairs[s]
            it = tr[i]
            T = it["aB"].shape[1]
            c0 = int(rng.integers(0, T - CROP + 1))
            flip = rng.random() < 0.5                       # vertical flip: y -> -y, line order reversed
            y = it["asig"][k, c0:c0 + CROP]
            if kind == "spec":
                x = np.nan_to_num(it["aB"][k, c0:c0 + CROP])
                X1.append(-x if flip else x)
            else:
                v, m = seq_input(it["aB"][:, c0:c0 + CROP], k, R)
                if flip:
                    v, m = -v[::-1], m[::-1]
                V.append(v); M.append(m)
                if kind == "seqp":                          # oracle slot: nearest B line within S/2, else baseline
                    yp = it["apen"][k, c0:c0 + CROP] * (-1 if flip else 1)
                    d = np.where(m > 0.5, np.abs(v - yp[None]), np.inf)
                    j = np.argmin(d, 0)
                    C.append(np.where(~np.isfinite(yp), -100, np.where(d[j, np.arange(CROP)] <= 0.5, j, v.shape[0])))
            Y.append(-y if flip else y)
        Y = torch.tensor(np.stack(Y), device=dev)
        ok = torch.isfinite(Y)
        Yz = torch.nan_to_num(Y)
        if kind == "spec":
            pred, _ = model(torch.tensor(np.stack(X1), device=dev))
            w = torch.hann_window(N_FFT, device=dev)
            mag = lambda z: torch.stft(z, N_FFT, HOP, window=w, return_complex=True, center=True).abs() / w.sum()
            loss = (pred - Yz).abs()[ok].mean() + 0.5 * (mag(pred) - mag(Yz)).abs().mean()
        else:
            pred, pp = model(torch.tensor(np.ascontiguousarray(np.stack(V)), device=dev), torch.tensor(np.ascontiguousarray(np.stack(M)), device=dev))
            loss = (pred - Yz).abs()[ok].mean()
            if kind == "seqp":
                loss = loss + CE_W * torch.nn.functional.nll_loss(torch.log(pp + 1e-6), torch.tensor(np.stack(C), device=dev, dtype=torch.long), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run.append(float(loss.detach()))
        if (step + 1) % 250 == 0 or step + 1 == steps:
            model.eval()
            vl = []
            for it in va[::3]:                              # monitoring only (no model selection)
                P = run_model(model, kind, it["aB"], dev)
                vl.append(np.nanmean(np.abs(P - it["asig"])))
            model.train()
            print(f"step {step + 1} loss {np.mean(run[-250:]):.4f} val_L1 {np.mean(vl):.4f} ({time.time() - t0:.0f}s)", file=log, flush=True)
    torch.save({"kind": kind, "state": model.state_dict(), "val_groups": val_groups, "steps": steps,
                "train_s": time.time() - t0}, OUT / f"model_{kind}.pt")
    print(f"saved model_{kind}.pt ({time.time() - t0:.0f}s)", file=log, flush=True)


def load_model(kind, dev):
    import torch
    ck = torch.load(OUT / f"model_{kind}.pt", map_location="cpu")
    m = build_model(kind)
    m.load_state_dict(ck["state"])
    return m.to(dev).eval(), ck


# ---------------------------------------------------------------- metrics
def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def line_scores(a, it, k):
    """fail % vs GT pen (missing or |err| > S/2, as track_joint.score); r, in_0.1S, lost_0.5S vs GT sig (no fit)"""
    apen, asig = it["apen"][k], it["asig"][k]
    ok = np.isfinite(apen) & np.isfinite(asig)
    fail = np.isnan(a) | (np.abs(np.nan_to_num(a, nan=1e9) - apen) > 0.5)
    e = np.abs(np.nan_to_num(a, nan=1e9) - asig)
    an = np.nan_to_num(a)
    return {"fail": float(fail[ok].mean() * 100), "r": pearson(an[ok], asig[ok]),
            "in01": float(np.mean(e[ok] < 0.1)), "lost05": float(np.mean(e[ok] > 0.5))}


def decompose(it, k):
    """quake-line columns by where its true trace is in B: own line / another line that left its own trace (swap) /
    another line on its own trace (crossing) / no B line within S/2 (lost)"""
    aB, apen = it["aB"], it["apen"]
    n = aB.shape[0]
    pos_true = apen[k] - 0.0                                 # y-up rel. to line k's baseline
    pos_B = aB - (np.arange(n) - k)[:, None]                  # every B line in the same frame
    near = np.abs(pos_B - pos_true[None]) <= 0.5
    off_own = np.abs(aB - apen) > 0.5
    own = near[k]
    oth = np.delete(np.arange(n), k)
    swap = ~own & (near[oth] & off_own[oth]).any(0)
    cross = ~own & ~swap & near[oth].any(0)
    lost, near0 = ~(own | swap | cross), np.abs(apen[k]) <= 0.5
    return {"own": own, "swap": swap, "cross": cross, "lost_near0": lost & near0, "lost_far": lost & ~near0}


CLASSES = ("own", "swap", "cross", "lost_near0", "lost_far")


def oracle_select(it, k):
    """GT oracle (a bound, not a method): the B line nearest to the true trace if within S/2, else the baseline"""
    n = it["aB"].shape[0]
    pos_B = np.nan_to_num(it["aB"] - (np.arange(n) - k)[:, None], nan=1e9)
    d = np.abs(pos_B - it["apen"][k][None])
    j = np.argmin(d, 0)
    cols = np.arange(pos_B.shape[1])
    return np.where(d[j, cols] <= 0.5, pos_B[j, cols], 0.0).astype(np.float32)


def evaluate(items, preds, tag):
    """per item x line scores for B, flat (layout baseline) and every model; quake-line failure by decomposition class"""
    rows = []
    for it in items:
        n = it["aB"].shape[0]
        q = it["active"]
        dq = decompose(it, q)
        for k in range(n):
            cat = "quake" if k == q else "neighbour" if abs(k - q) == 1 else "other"
            row = {"set": tag, "record": it["record"], "gain": it["gain"], "line": k, "cat": cat}
            meth = {"B": it["aB"][k], "flat": np.zeros_like(it["aB"][k]), "oracle": oracle_select(it, k)}
            meth.update({m: P[k] for m, P in preds[it["record"], it["gain"]].items()})
            for m, a in meth.items():
                row[m] = line_scores(a, it, k)
                if k == q:
                    fail = np.abs(a - it["apen"][k]) > 0.5
                    row[m]["fail_by_class"] = {c: (float(fail[msk].mean() * 100) if msk.any() else None) for c, msk in dq.items()}
            if k == q:
                row["class_frac"] = {c: float(msk.mean() * 100) for c, msk in dq.items()}
            rows.append(row)
    # check: B's fail here equals track_joint.score on the pixel paths
    it = items[0]
    ref = score(it["B"], it["pen"], range(it["aB"].shape[0]), it["x0"], it["x1"], it["S"])[0]
    mine = round(float(np.mean([r["B"]["fail"] for r in rows if r["record"] == it["record"] and r["gain"] == it["gain"]])), 1)
    assert abs(ref - mine) < 0.11, (ref, mine)
    return rows


def summarize(rows, methods, gains):
    out = {}
    for g in list(gains) + ["all"]:
        rr = [r for r in rows if g == "all" or r["gain"] == g]
        s = {}
        for cat in ("quake", "neighbour", "other", "whole"):
            cr = [r for r in rr if cat == "whole" or r["cat"] == cat]
            s[cat] = {m: {key: round(float(np.mean([r[m][key] for r in cr])), 3) for key in ("fail", "r", "in01", "lost05")} for m in methods}
        qr = [r for r in rr if r["cat"] == "quake"]
        s["quake_class_frac"] = {c: round(float(np.mean([r["class_frac"][c] for r in qr])), 1) for c in CLASSES}
        s["quake_any_line_bound"] = round(float(np.mean([100 - r["class_frac"]["lost_near0"] - r["class_frac"]["lost_far"] for r in qr])), 1)
        s["quake_any_line_or_baseline_bound"] = round(float(np.mean([100 - r["class_frac"]["lost_far"] for r in qr])), 1)
        s["quake_fail_by_class"] = {m: {c: round(float(np.mean([v for v in [r[m]["fail_by_class"][c] for r in qr] if v is not None])), 1)
                                        for c in CLASSES} for m in methods}
        s["n_items"] = len(qr)
        out[str(g)] = s
    return out


def print_summary(summ, methods, title, fh=None):
    lines = [f"== {title}"]
    for g, s in summ.items():
        lines.append(f"gain {g} (n={s['n_items']}): quake columns {s['quake_class_frac']} -> any-line bound {s['quake_any_line_bound']}%, any-line-or-baseline {s['quake_any_line_or_baseline_bound']}%")
        for cat in ("quake", "neighbour", "whole"):
            lines.append(f"  {cat:9s} " + " | ".join(f"{m}: fail {s[cat][m]['fail']:5.1f} r {s[cat][m]['r']:+.2f} in {s[cat][m]['in01']:.2f} lost {s[cat][m]['lost05']:.2f}" for m in methods))
        lines.append("  quake fail by class: " + " | ".join(f"{m}: " + "/".join(f"{s['quake_fail_by_class'][m][c]:.0f}" for c in CLASSES) for m in methods))
    txt = "\n".join(lines)
    print(txt, flush=True)
    if fh:
        fh.write(txt + "\n")


def predict_all(items, kinds, dev):
    preds = {(it["record"], it["gain"]): {} for it in items}
    for kind in kinds:
        model, _ = load_model(kind, dev)
        for it in items:
            preds[it["record"], it["gain"]][kind] = run_model(model, kind, it["aB"], dev)
    return preds


def available_models():
    return [k for k in ("seq", "seqp", "seq1", "spec") if (OUT / f"model_{k}.pt").exists()]


def dev_eval():
    items = load_items(TRAIN_BENCH, TRAIN_CACHE, TRAIN_GAINS)
    kinds = available_models()
    _, ck = load_model(kinds[0], "cpu")
    va = [it for it in items if it["group"] in ck["val_groups"]]
    rows = evaluate(va, predict_all(va, kinds, device()), "val")
    methods = ["B", "flat", "oracle"] + kinds
    summ = summarize(rows, methods, TRAIN_GAINS)
    with open(OUT / "dev_summary.txt", "w") as fh:
        print_summary(summ, methods, f"validation groups {ck['val_groups']} ({len({it['record'] for it in va})} records)", fh)
    (OUT / "dev_results.json").write_text(json.dumps({"val_groups": ck["val_groups"], "summary": summ, "rows": rows}, indent=1))
    # training-set bound (all 43 records, B only): how much of the quake line exists anywhere in B
    rows_all = evaluate(items, {(it["record"], it["gain"]): {} for it in items}, "all")
    summ_all = summarize(rows_all, ["B", "flat", "oracle"], TRAIN_GAINS)
    with open(OUT / "bound_all_train.txt", "w") as fh:
        print_summary(summ_all, ["B", "flat", "oracle"], "all 43 film_train records (B only, bound)", fh)
    (OUT / "bound_all_train.json").write_text(json.dumps(summ_all, indent=1))


def holdout_eval():
    """the one final run: frozen models on out/film_holdout3 (B from the shared cache if another experiment made it)"""
    if HOLD_CACHE_SHARED.exists() and all((HOLD_CACHE_SHARED / f"{r.name}_film_g{g:g}.npz").exists()
                                          for r in HOLD_BENCH.iterdir() if r.is_dir() for g in HOLD_GAINS):
        cache = HOLD_CACHE_SHARED
    else:
        cache = HOLD_CACHE_OWN
        build_cache(HOLD_BENCH, cache, HOLD_GAINS)
    items = load_items(HOLD_BENCH, cache, HOLD_GAINS)
    kinds = available_models()
    rows = evaluate(items, predict_all(items, kinds, device()), "holdout3")
    methods = ["B", "flat", "oracle"] + kinds
    summ = summarize(rows, methods, HOLD_GAINS)
    with open(OUT / "holdout3_summary.txt", "w") as fh:
        print_summary(summ, methods, f"film_holdout3 ({len({it['record'] for it in items})} records, B cache {cache})", fh)
    (OUT / "holdout3_results.json").write_text(json.dumps({"cache": str(cache), "summary": summ, "rows": rows}, indent=1))


# ---------------------------------------------------------------- real paired eval
def real_eval(name):
    """repair B's lines on the rescaled scan (paths saved by real_paired_eval) and score them with its alignment
    (chosen_dp_fit), global amplitude, line categories and line_metrics"""
    from obspy import UTCDateTime
    from real_paired_eval import SCANS, TARGET_S, line_metrics, lowpass, reference, sample_path
    cfg = SCANS[name]
    res = json.loads((ROOT / "out" / "real_paired" / f"{name}_results.json").read_text())
    z = np.load(ROOT / "out" / "real_paired" / name / f"{name}_paths.npz")
    B, bases = z["B"], z["bases"]
    lay, al, a_glob = res["layout"], res["alignment"]["chosen_dp_fit"], res["amplitude"]["global_scale_px_per_m"]
    x0, x1, xa, S = lay["x0"], lay["x1"], lay["xa"], TARGET_S
    # B's amplitude series, resampled from px to 1 sample/s (the synthetic geometry the models were trained on)
    cols = np.arange(x0, x1, dtype=float)
    pps = lay["pps_nominal"]
    ts = np.arange(0.0, (x1 - 1 - x0) / pps + 1e-9, 1.0)
    xs = x0 + ts * pps
    aB = np.stack([np.interp(xs, cols, (b - B[k, x0:x1]) / S) for k, b in enumerate(bases)]).astype(np.float32)
    dev = device()
    paths = {"B": B, "flat": np.where(np.isnan(B), np.nan, bases[:, None] + 0 * B)}
    for kind in available_models():
        model, _ = load_model(kind, dev)
        a = run_model(model, kind, aB, dev)
        P = np.full_like(B, np.nan)
        P[:, x0:x1] = np.stack([b - S * np.interp(cols, xs, a[k]) for k, b in enumerate(bases)])
        paths[kind] = P
    t_on, t_off = UTCDateTime(cfg["on"]), UTCDateTime(cfg["off"])
    t_ref, ref, _ = reference(cfg, t_on)
    t_lo, t_hi = 0.0, float(t_off - t_on)
    rows = []
    for ln in res["lines"]:
        k = ln["line"]
        idx, yB, xk = sample_path(B, k, x0, x1, xa, al, t_ref, t_lo, t_hi)
        tk, rf = t_ref[idx], ref[idx]
        row = {"line": k, "category": ln["category"], "methods": {}}
        for m, P in paths.items():
            _, y, _ = sample_path(P, k, x0, x1, xa, al, t_ref, t_lo, t_hi)
            row["methods"][m] = line_metrics(-(y - bases[k]) * al["sign"], rf, tk, S, a_glob)
        # bound analogue: the reference mapped onto the scan exactly as real_paired_eval's overlay `pred` (global
        # amplitude, per-line offset + trend fitted to B), then is ANY B line within S/2 of it at that column?
        tt = tk - tk.mean()
        cg = np.linalg.lstsq(np.c_[np.ones_like(tt), tt], lowpass(-(yB - bases[k]) * al["sign"]) - a_glob * lowpass(rf), rcond=None)[0]
        yq = bases[k] - al["sign"] * (a_glob * lowpass(rf) + cg[0] + cg[1] * tt)
        near = np.abs(np.stack([np.interp(xk[idx], cols, B[j, x0:x1]) for j in range(len(bases))]) - yq[None]) <= S / 2
        row["bound"] = {"own": round(float(near[k].mean()), 3), "any_line": round(float(near.any(0).mean()), 3)}
        rows.append(row)
    # reproduce real_paired_eval's B numbers first (same paths, alignment, metric code)
    for r in rows:
        stored = next(v for v in res["lines"] if v["line"] == r["line"])["methods"]["B"]
        assert all(abs(stored[key] - r["methods"]["B"][key]) < 1e-3 for key in ("r_band", "in_0.1S_global", "lost_0.5S_global")), (r["line"], stored, r["methods"]["B"])
    keys = ("r_band", "in_0.1S", "lost_0.5S", "in_0.1S_global", "lost_0.5S_global", "rms_err_global_S")
    summ = {}
    for cat in ("quake", "neighbour", "other_active", "quiet"):
        rr = [r for r in rows if r["category"] == cat]
        summ[cat] = {"lines": [r["line"] for r in rr],
                     "bound_own": round(float(np.mean([r["bound"]["own"] for r in rr])), 3),
                     "bound_any_line": round(float(np.mean([r["bound"]["any_line"] for r in rr])), 3)}
        for m in paths:
            summ[cat][m] = {key: round(float(np.mean([r["methods"][m][key] for r in rr])), 3) for key in keys}
    (OUT / f"real_{name}.json").write_text(json.dumps({"scan": name, "alignment": al, "global_scale_px_per_m": a_glob,
                                                         "resampled_to_1sps_from_pps": pps, "summary": summ, "lines": rows}, indent=1))
    with open(OUT / f"real_{name}.txt", "w") as fh:
        for cat, v in summ.items():
            t = f"[{name}] {cat:12s} {v['lines']} bound own/any {v['bound_own']:.2f}/{v['bound_any_line']:.2f} | " + " | ".join(
                f"{m}: r {v[m]['r_band']:+.3f} in {v[m]['in_0.1S_global']:.2f} lost {v[m]['lost_0.5S_global']:.2f} (LS in {v[m]['in_0.1S']:.2f} lost {v[m]['lost_0.5S']:.2f})"
                for m in paths)
            print(t, flush=True)
            fh.write(t + "\n")
    np.savez_compressed(OUT / f"real_{name}_paths.npz", **{m: P for m, P in paths.items()}, bases=bases)


def figure(gains=(2, 6)):
    """validation examples: the quake line of the val record with the largest swing; truth, B, flat-free models"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = load_items(TRAIN_BENCH, TRAIN_CACHE, gains)
    kinds = available_models()
    _, ck = load_model(kinds[0], "cpu")
    va = [it for it in items if it["group"] in ck["val_groups"]]
    preds = predict_all(va, kinds, device())
    fig, axs = plt.subplots(len(gains), 1, figsize=(16, 3.4 * len(gains)), squeeze=False)
    for ax, g in zip(axs[:, 0], gains):
        it = max((it for it in va if it["gain"] == g), key=lambda it: np.nanmax(np.abs(it["asig"][it["active"]])))
        k = it["active"]
        c = int(np.nanargmax(np.abs(it["asig"][k])))
        lo, hi = max(0, c - 700), min(it["asig"].shape[1], c + 1300)
        t = np.arange(lo, hi)
        ax.plot(t, it["asig"][k, lo:hi], "k", lw=1.6, label="truth (GT sig)")
        ax.plot(t, it["aB"][k, lo:hi], color="tab:red", lw=0.9, label="B")
        for kind, col in (("seq", "tab:blue"), ("seqp", "tab:purple"), ("seq1", "tab:green"), ("spec", "tab:orange")):
            if kind in preds[it["record"], g]:
                ax.plot(t, preds[it["record"], g][kind][k, lo:hi], color=col, lw=0.9, label=kind)
        ax.set_title(f"{it['record']} gain {g}, quake line {k} (validation group)")
        ax.set_xlabel("s from line start"); ax.set_ylabel("line spacings (up +)"); ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig_val_quake.png", dpi=90); plt.close(fig)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "fig":
        figure()
    if cmd == "cache":
        build_cache(TRAIN_BENCH, TRAIN_CACHE, TRAIN_GAINS, int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif cmd == "train":
        train(sys.argv[2])
    elif cmd == "dev":
        dev_eval()
    elif cmd == "holdout":
        holdout_eval()
    elif cmd == "real":
        real_eval(sys.argv[2].upper())
