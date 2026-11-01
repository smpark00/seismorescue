"""The classical tracker on all splits (proposal B of scripts/arbiter.py, started at the true pen at the first column) and the retired
SAM 3 + RA2 rule on holdout6/4, as SeismoBench submissions. Reads stored paths under out/ that are not in this repository; the submissions
are in the GitHub Release.

usage: python -I bench/build/extra.py <out>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
OUTROOT = Path(sys.argv[1])
SPLITS = {"holdout7": ROOT / "out/film2_holdout7", "holdout6": ROOT / "out/film2_holdout6", "holdout4": ROOT / "out/film2_holdout4"}
OLD = json.loads((ROOT / "out/ml/consensus/frozen_rule.json").read_text())
print("retired rule", {k: OLD[k] for k in OLD if k in ("tau_S", "min_run_cols", "tau", "L")}, flush=True)
TAU, L = OLD.get("tau_S", OLD.get("tau")), OLD.get("min_run_cols", OLD.get("L", 1))


def runs_ge(mask, L):
    if L <= 1:
        return mask
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    out = mask.copy()
    for s, t in zip(e[::2], e[1::2]):
        if t - s < L:
            out[s:t] = False
    return out


for split, root in SPLITS.items():
    (OUTROOT / split / "classical_tracker").mkdir(parents=True, exist_ok=True)
    if split != "holdout7":
        (OUTROOT / split / "sam3_plus_offline_network").mkdir(parents=True, exist_ok=True)
    cache = ROOT / "out/ml/sim2real" / f"cache_{split}"
    made = 0
    for rec in sorted(p for p in root.iterdir() if p.is_dir()):
        meta = json.loads((rec / "film" / "meta.json").read_text())
        S, x0 = float(meta["spacing"]), int(meta["margin_x"])
        for g in meta["gains"]:
            gain = f"{g:g}"
            c = cache / f"{rec.name}_film_g{gain}.npz"
            if c.exists():
                B = np.load(c)["B"]
            else:
                from arbiter import proposals
                img = cv2.imread(str(rec / "film" / f"heli_g{gain}.png"), 0)
                pen = np.load(rec / "film" / f"gt_g{gain}.npz")["pen"]
                _, B, _ = proposals(img, pen[:, x0], x0, img.shape[1] - x0, S)
                made += 1
            np.savez_compressed(OUTROOT / split / "classical_tracker" / f"{rec.name}_g{gain}.npz", path=np.asarray(B, np.float32))
            if split != "holdout7":
                F = np.load(ROOT / "out/ml/sam3_film2/paths" / split / f"{rec.name}_g{gain}.npz")["SAM"].astype(np.float32)
                R = np.load(ROOT / "out/ml/consensus/paths" / f"{split}_RA2" / f"{rec.name}_g{gain}.npz")["RA2"].astype(np.float32)
                d = np.where(np.isfinite(F) & np.isfinite(R), np.abs(F - R) / S, np.inf)
                acc = np.zeros(F.shape, bool)
                sl = slice(x0, F.shape[1] - x0)
                acc[:, sl] = np.stack([runs_ge(a, L) for a in d[:, sl] < TAU])
                np.savez_compressed(OUTROOT / split / "sam3_plus_offline_network" / f"{rec.name}_g{gain}.npz", path=F, accept=acc)
    print(split, "tracker runs", made, flush=True)
