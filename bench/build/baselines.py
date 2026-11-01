"""Turn the stored paths into SeismoBench submissions under <out>/<split>/<baseline>/. The paths under out/ are not in this repository
(scripts/ml_fb.py dump makes them); the submissions are in the GitHub Release.

usage: python -I bench/build/baselines.py <out>
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTROOT = Path(sys.argv[1])
SPLITS = {"holdout7": ROOT / "out/film2_holdout7", "holdout6": ROOT / "out/film2_holdout6", "holdout4": ROOT / "out/film2_holdout4"}
RULE = json.loads((ROOT / "out/ml/fb/frozen_rule.json").read_text())
TAU, L = RULE["tau_S"], RULE["min_run_cols"]


def runs_ge(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    out = mask.copy()
    for s, t in zip(e[::2], e[1::2]):
        if t - s < L:
            out[s:t] = False
    return out


for split, root in SPLITS.items():
    fb = ROOT / "out/ml/fb/paths" / split
    for name in ("flat_line", "sam3_forward", "sam3_both_directions"):
        (OUTROOT / split / name).mkdir(parents=True, exist_ok=True)
    n = 0
    for rec in sorted(p for p in root.iterdir() if p.is_dir()):
        meta = json.loads((rec / "film" / "meta.json").read_text())
        S, x0, nl = float(meta["spacing"]), int(meta["margin_x"]), int(meta["n_lines"])
        base = meta["margin_y"] + S * np.arange(nl)
        for g in meta["gains"]:
            gain = f"{g:g}"
            z = np.load(fb / f"{rec.name}_g{gain}.npz")
            F, B = z["F"].astype(np.float32), z["B"].astype(np.float32)
            flat = np.repeat(base[:, None], F.shape[1], 1).astype(np.float32)
            np.savez_compressed(OUTROOT / split / "flat_line" / f"{rec.name}_g{gain}.npz", path=flat)
            np.savez_compressed(OUTROOT / split / "sam3_forward" / f"{rec.name}_g{gain}.npz", path=F)
            d = np.where(np.isfinite(F) & np.isfinite(B), np.abs(F - B) / S, np.inf)
            acc = np.zeros(F.shape, bool)
            sl = slice(x0, F.shape[1] - x0)
            acc[:, sl] = np.stack([runs_ge(a) for a in d[:, sl] < TAU])
            np.savez_compressed(OUTROOT / split / "sam3_both_directions" / f"{rec.name}_g{gain}.npz", path=F, accept=acc)
            n += 1
    print(split, n, "images", flush=True)
