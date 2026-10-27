# Checkpoints

| Folder | Model | Frozen decode setting | Synthetic holdout (quake line) |
|---|---|---|---|
| `rowanchor/` | Row-anchor sequence model, `best.pt` (step 3368, 0.88 M parameters), with `logT.npy` (learned step prior) | `frozen.json`: Viterbi, α = 4 | r 0.96, failure 2.0% |
| `sam21_tiny_ft/` | SAM 2.1 hiera-tiny fine-tuned (LoRA + mask decoder), `ckpt_tiny_best.pt` | `frozen_method.json`: λ = 1e-5, window 256 columns, step 192 | r 0.89, failure 12.0% |

**Row-anchor.** The scripts read from `out/`, so copy the files there first:

```bash
mkdir -p out/ml/rowanchor && cp checkpoints/rowanchor/* out/ml/rowanchor/
.venv/bin/python -I scripts/ml_rowanchor.py holdout      # needs the rendered out/film_holdout3
```

**SAM 2.1.** Pass the checkpoint explicitly:

```bash
.venv/bin/python -I scripts/ml_sam.py eval holdout3 --ckpt checkpoints/sam21_tiny_ft/ckpt_tiny_best.pt --lam 1e-5
```

`ml_sam.py` downloads the base SAM 2.1 weights from `dl.fbaipublicfiles.com` on first use. The fine-tuned checkpoint is a derivative of Meta's SAM 2.1 (Apache-2.0); see [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
