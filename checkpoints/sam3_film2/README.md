# SAM 3 film2 adapter

Fine-tuned from Meta's SAM 3 during the hackathon, for tracing WWSSN film lines. **This folder is not under the repository's MIT licence.** It is a derivative of SAM 3 and is distributed only under the SAM License ([LICENSE](LICENSE), an exact copy of the upstream file).

| Item | Value |
|---|---|
| File | `ckpt_sam3_film2_best.pt`, 48,957,717 bytes |
| sha256 | `b66e3d5159fb8d01b946c9291427f3e5544d5d1ef62ebdfd40a1066aaa93defe` |
| Base model | `facebook/sam3`, revision `3c879f39826c281e95690f02c7821c4de09afae7` (gated, not included) |
| What it holds | LoRA weights (rank 8, last 16 of 32 backbone blocks), plus the fine-tuned FPN neck and mask decoder. It does not hold the base weights of the backbone blocks. |
| Training | Synthetic film from our renderer (`scripts/ml_sim2real_render.py`, `scripts/make_film2_train.py`), 60 min on one Modal H100 (`scripts/ml_sam3_modal.py`) |
| Frozen settings | `frozen_method.json`: λ 1e-5, window 406 × 256, step 192, fp32 |

**To run it** you need your own approved access to `facebook/sam3`:

```bash
.venv/bin/hf auth login
HF_HUB_OFFLINE=0 .venv/bin/hf download facebook/sam3 --revision 3c879f39826c281e95690f02c7821c4de09afae7 --include config.json model.safetensors
```

**Use restrictions.** The SAM License section 1.b.v applies to this adapter and to anything made from it, including the limits on military use and on nuclear applications. Read the licence before use. Provided as is, without warranty.
