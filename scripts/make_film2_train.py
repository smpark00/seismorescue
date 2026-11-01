"""film2 renders (research/31) in the film_train layout that ml_sam / ml_sam3 read. Files are hard links; the source
renders are not modified.
  (no arg)  training: out/ml/sim2real/render/train/<rec>/film/heli_v<j>.png -> out/film2_train/<rec>/film/heli_g<k>.png,
            gt_g<k>.npz with k = j + 1 (k is only an image id, not a gain: each variant has its own random amplitude P,
            kept in meta.json "variant_of_k").
  holdout4  test split (research/37 §6): out/ml/sim2real/render/holdout4/<rec>/film/heli_g<g>.png (fixed gains
            1, 2, 4, 6, 8 with P = g) -> out/film2_holdout4/<rec>/film/, same file names; meta.json gets "gains".
  holdout6  the same for the E3 test split (research/41).
usage: .venv/bin/python -I scripts/make_film2_train.py [holdout4|holdout6]"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RENDER = ROOT / "out" / "ml" / "sim2real" / "render"
HOLD = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("holdout4", "holdout6") else None
SRC, DST = (RENDER / HOLD, ROOT / "out" / f"film2_{HOLD}") if HOLD else (RENDER / "train", ROOT / "out" / "film2_train")

n = 0
for rec in sorted(p for p in SRC.iterdir() if p.is_dir()):
    meta = json.loads((rec / "film" / "meta.json").read_text())
    tags = sorted(meta["variants"])                      # train: v0 .. v5; holdout4 / holdout6: g1 g2 g4 g6 g8
    out = DST / rec.name / "film"
    out.mkdir(parents=True, exist_ok=True)
    if HOLD:
        gains = sorted(float(t[1:]) for t in tags)
        assert gains == [1, 2, 4, 6, 8] and all(meta["variants"][f"g{g:g}"]["P"] == g for g in gains), (rec.name, tags)
        names = [(f"heli_g{g:g}.png", f"heli_g{g:g}.png") for g in gains] + [(f"gt_g{g:g}.npz", f"gt_g{g:g}.npz") for g in gains]
        meta = {**meta, "gains": gains}
    else:
        names = [(f"{p}_{t}.{e}", f"{p}_g{k}.{e}") for k, t in enumerate(tags, 1) for p, e in (("heli", "png"), ("gt", "npz"))]
        meta = {**meta, "gains": [float(k) for k in range(1, len(tags) + 1)],
                "variant_of_k": {str(k): {"variant": t, "P": meta["variants"][t]["P"]} for k, t in enumerate(tags, 1)}}
    for a, b in names:
        if not (out / b).exists():
            os.link(rec / "film" / a, out / b)
    n += len(meta["gains"])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
print(f"{n} images from {len(list(DST.iterdir()))} records -> {DST}")
