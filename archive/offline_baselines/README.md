# Offline baselines (not part of what runs)

Earlier in the event we trained small row-anchor networks of our own (row-anchor v1 and RA2, 0.88 M parameters, no pretrained weights). RA2 was used for a while as a second opinion next to SAM 3.

We removed it so that every model called at runtime is a Meta model:

- The pipeline does not import from this folder or load these networks. Only code for the earlier evaluations does: the `dump` and `e3` commands of `scripts/ml_consensus.py`, and `results/ml/rescoring/verify_ra.py`. The second opinion is now SAM 3 itself, tracing each line from both ends (see the [main README](../../README.md)).
- Their weights are no longer in the current files; older commits still hold them. Without them these scripts cannot be rerun as they stand.

## How the earlier rule compared

- **It was more precise, on easier columns.** On the images of the earlier test (10 other quakes, 90 images, so this is a comparison, not a test) SAM 3 with RA2 gave 75.7% / 97.5% on the earthquake line and 92.6% / 99.2% on all lines (accepted columns / of those, within ¼ line spacing), against 78.4% / 95.0% and 94.0% / 98.0% for the two SAM 3 passes. A flat line on the columns each rule accepted scores 80.5% and 95.4% for the earlier rule, 76.7% and 93.5% for the two passes, so on the earthquake line their lead over a flat line is about the same (17.0 vs 18.3 points). Where the pen moved it reached 88.1% against 79.9%, on fewer of those columns (43.9% vs 54.5%). Its SAM 3 paths also started from the true pen position; the two passes start from L0's baselines. [LEADERBOARD.md](../../LEADERBOARD.md) lists this rule as `sam3_plus_offline_network`, with intervals.
- **RA2 did better than SAM 3 on two real earthquake lines:** r 0.15 vs 0.02 on MAT and 0.48 vs 0.21 on KIP (each with its own decoder). On ALQ it is the reverse (+0.50 vs +0.75). We removed it to keep every runtime model a Meta model, not because it lost.

## What is left

What is left, so the earlier comparison can still be read:

| Path | Contents |
|---|---|
| `ml_rowanchor.py`, `ml_sim2real_train.py` | Training and evaluation code of the two networks |
| `results/consensus/` | Stored results of the earlier SAM 3 + RA2 rule: rule choice on holdout4 (E2), test on holdout6 (E3), the real chips |
| `results/click/` | Stored results of the click experiment on that earlier rule |
