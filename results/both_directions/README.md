# Results of the both-directions rule

SAM 3, fine-tuned by us, traces every line twice: forward from the left edge, and backward on the mirrored image from the right edge. A column is accepted where the two passes stay within τ line spacings for at least L columns in a row. The value exported is the forward pass.

| File | What it is |
|---|---|
| [`../../checkpoints/both_directions/frozen_rule.json`](../../checkpoints/both_directions/frozen_rule.json) | The rule in use: τ 0.25, L 20 |
| `select_holdout4.json` | The 24 candidates on the development set (10 earthquakes, 100 images), with and without the minute marks, and the choice |
| `select_holdout4_before_amendment.json`, `frozen_rule_fallback.json` | The first selection, kept as it came out: no candidate met the targets, so the fallback written down in advance (τ 0.05, L 60) was chosen |
| `holdout7_started.json` | Start marker of the test, written before its first image: time, number of images, hashes of the code, the rule and the checkpoint |
| `score_holdout7.json` | **The test:** 10 earthquakes on days not used before, 95 images, run once |
| `score_holdout6.json` | The same rule on the images of an earlier test. These images had been seen before, so this is only a comparison with the earlier rule |
| `score_holdout4.json` | The same rule on the development set |
| `real_results.json` | The three real chips with a digital record. ALQ is by line type only |
| `click/` | The click experiment on the development set |

Logs of these runs are in [`../logs/`](../logs).

Three definitions the files do not state. `true_motion_S_median` over several lines is the mean of the per-line medians. `clicks_per_image` divides by the 72 images that have review windows. The circular-shift control for MAT line 6 skips shifts under 600 s, which would come close to the true alignment (with them the largest |r| is 0.85).

## The change to the scoring

We changed how the targets are scored after seeing the development set and before running the test. This is the record.

1. **Written down first:** on the development set, take the rule with the most accepted earthquake-line columns among those with precision ≥ 95% on the earthquake line and ≥ 99% on all lines. If none qualifies, use τ 0.05, L 60.
2. **What happened:** none qualified. All-lines precision never passed 98.5%. The fallback accepted 25.5% of earthquake-line columns and 71.1% of all columns, which would send most quiet lines to a person.
3. **Why:** of the 33,562 wrongly accepted columns at τ 0.05, L 60, 33,483 (99.76%) were minute marks, where the pen jumps up for a few seconds. Neither pass follows these ticks. Only 12 of 2,048,557 accepted columns were off by more than S/2. So the limit was the ticks, not slips onto other lines.
4. **The change:** the targets are measured without the columns where the pen is more than 0.1 S from the signal (3.69% of the columns). The grid, the targets, the order of choice and the fallback stayed the same. Thresholds of 0.05 S and 0.02 S pick the same rule.
5. **Why that is fair to the output:** the export bridges these seconds with a straight line anyway. On the development set 99.87% of the columns left out fall in bridged seconds (seconds :58–:02 of each minute and :58–:06 at the hour; 106,088 of 106,224). On real film this depends on L0's clock.
6. **What it picked:** τ 0.25, L 20.
7. **What did not change:** the test numbers and the judgement written down in advance are scored with the minute marks, as first registered. The numbers without them are reported beside them.

The test set played no part in choosing the rule, so its numbers have no selection bias. But the rule that was tested is not the one the first scoring would have picked.
