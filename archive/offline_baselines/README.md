# Offline baselines (not part of what runs)

Earlier in the event we trained small row-anchor networks of our own (row-anchor v1 and RA2, 0.88 M parameters, no pretrained weights). RA2 was used for a while as a second opinion next to SAM 3.

On 2026-11-01 the organizers ruled that a custom neural network trained from scratch cannot run in a project at runtime, however small, because every model a project calls at runtime must be a Meta model. So we removed it:

- Nothing in `scripts/` imports from this folder, and the pipeline no longer loads these networks. The second opinion is now SAM 3 itself, tracing each line from both ends (see the [main README](../../README.md)).
- Their weights are no longer in this repository. Without them these scripts cannot be rerun as they stand.

What is left, so the earlier comparison can still be read:

| Path | Contents |
|---|---|
| `ml_rowanchor.py`, `ml_sim2real_train.py` | Training and evaluation code of the two networks |
| `results/consensus/` | Stored results of the earlier SAM 3 + RA2 rule: rule choice on holdout4 (E2), test on holdout6 (E3), the real chips |
| `results/click/` | Stored results of the click experiment on that earlier rule |
