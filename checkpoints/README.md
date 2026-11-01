# Checkpoints

| Folder | Model | Frozen decode setting | Synthetic holdout (quake line) |
|---|---|---|---|
| `sam3_film2/` | **SAM 3, fine-tuned by us** (adapter, needs the gated base weights). Separate licence: [its README](sam3_film2/README.md) | `frozen_method.json`: λ = 1e-5, window 406 × 256, step 192 | See the main README |
| `ra2/` | RA2, our row-anchor model on film2 renders, 0.88 M parameters, trained by us during the event with no pretrained weights. **Not a Meta model.** Only checks SAM 3; its path is never exported. Training code: `scripts/ml_sim2real_train.py` | `frozen.json`: soft-mean | See the main README |
| `consensus/` | The frozen L3 rule: accept where the two paths agree within 0.15 line spacing | `frozen_rule.json` | See the main README |
| `rowanchor/` | Row-anchor sequence model, `best.pt` (step 3368, 0.88 M parameters), with `logT.npy` (learned step prior) | `frozen.json`: Viterbi, α = 4 | r 0.96, failure 2.0% |
| `sam21_tiny_ft/` | SAM 2.1 hiera-tiny fine-tuned (LoRA + mask decoder), `ckpt_tiny_best.pt` | `frozen_method.json`: λ = 1e-5, window 256 columns, step 192 | r 0.89, failure 12.0% |

`rowanchor/` and `sam21_tiny_ft/` are the earlier models. `ra2/best.pt` was fine-tuned from `rowanchor/best.pt`.

The scripts read from `out/`, so run `./bootstrap.sh` first. It copies these files there.

**Row-anchor.**

```bash
.venv/bin/python -I scripts/ml_rowanchor.py holdout      # needs the rendered out/film_holdout3
```

**SAM 2.1.** Pass the checkpoint explicitly:

```bash
.venv/bin/python -I scripts/ml_sam.py eval holdout3 --ckpt checkpoints/sam21_tiny_ft/ckpt_tiny_best.pt --lam 1e-5
```

`ml_sam.py` downloads the base SAM 2.1 weights from `dl.fbaipublicfiles.com` on first use. The fine-tuned checkpoint is a derivative of Meta's SAM 2.1 (Apache-2.0); see [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
