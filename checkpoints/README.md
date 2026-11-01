# Checkpoints

| Folder | Model | Frozen setting | Results |
|---|---|---|---|
| `sam3_film2/` | **SAM 3, fine-tuned by us** (adapter, needs the gated base weights). Separate licence: [its README](sam3_film2/README.md) | `frozen_method.json`: λ = 1e-5, window 406 × 256, step 192 | See the main README |
| `both_directions/` | The frozen L3 rule: accept where the two SAM 3 passes (left to right, right to left) agree. A rule, not a model | `frozen_rule.json` | See the main README |
| `sam21_tiny_ft/` | SAM 2.1 hiera-tiny fine-tuned (LoRA + mask decoder), `ckpt_tiny_best.pt`. Our first fine-tune, no longer used by the pipeline | `frozen_method.json`: λ = 1e-5, window 256 columns, step 192 | Earlier renderer: quake-line r 0.89, failure 12.0% |

The scripts read from `out/`, so run `./bootstrap.sh` first. It copies these files there.

The small row-anchor networks we trained as offline baselines are not here. Why: [archive/offline_baselines](../archive/offline_baselines/README.md).

**SAM 2.1.** Pass the checkpoint explicitly:

```bash
.venv/bin/python -I scripts/ml_sam.py eval holdout3 --ckpt checkpoints/sam21_tiny_ft/ckpt_tiny_best.pt --lam 1e-5
```

`ml_sam.py` downloads the base SAM 2.1 weights from `dl.fbaipublicfiles.com` on first use. The fine-tuned checkpoint is a derivative of Meta's SAM 2.1 (Apache-2.0); see [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
