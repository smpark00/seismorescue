# SeismoBench: tracing lines on seismogram film

A benchmark for one task: follow every pen line on a scanned analog seismogram, including where the lines cross. We know of no shared benchmark for this task (two searches, not exhaustive).

**Read this first.**

- **The truth of every split is public, so scores rest on trust.**
- **There are no outside participants. Every row is one of our own baselines.**
- The film is synthetic: real broadband records rendered through our calibrated film renderer, so the true pen position is exact. A high score here does not mean the same score on real film, which is why the real chips have their own table below.

How to score your own tracer: [bench/README.md](bench/README.md). Definitions are at the top of [bench/score.py](bench/score.py). Rows are in a fixed order, not a ranking. "Earthquake line r" scores the whole line and ignores `accept`, so it is the same for the forward and the both-directions rows; `r_accepted` is in the result files. S is one line spacing (38 px). "Right" means within S/4. A flat line gets 0% where the pen moved by definition.

## holdout7

**Main table.** 10 earthquakes (M7.5–8.8), 95 images. It was our test split: rule and code were frozen before it ran, once. Its truth is public from now on.

| Baseline | Earthquake line r | Accepted, earthquake line | Right, accepted | Flat line, same columns | Pen moved: right, all columns | Pen moved: right, accepted | Accepted, all lines | Right, accepted | Flat line, same columns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sam3_both_directions` | 0.865 | 87.9% | 94.2% [92.8, 95.3] | 72.5% | 67.2% | 80.6% | 93.9% | 97.9% | 93.6% |
| `sam3_forward` | 0.865 | 100.0% | 87.5% [79.6, 91.9] | 65.8% | 67.2% | 67.2% | 100.0% | 94.1% | 89.5% |
| `classical_tracker` | 0.010 | 100.0% | 23.0% [17.5, 28.0] | 65.8% | 9.0% | 9.0% | 100.0% | 50.6% | 89.5% |
| `flat_line` | 0.000 | 100.0% | 65.8% [58.9, 71.1] | 65.8% | 0.0% | 0.0% | 100.0% | 89.5% | 89.5% |

- `sam3_both_directions`: SAM 3 fine-tuned by us, traced from both ends, accepted where the passes agree (τ 0.25 S, at least 20 columns)
- `sam3_forward`: The same SAM 3, left to right only, every column accepted
- `classical_tracker`: Our classical tracker, no learning, started at the true pen position. Built for an earlier film geometry and not tuned again
- `flat_line`: No skill: the baseline as the answer

## holdout6

10 other earthquakes, 90 images. An earlier test split, so the both-directions rows here are a comparison, not a test.

| Baseline | Earthquake line r | Accepted, earthquake line | Right, accepted | Flat line, same columns | Pen moved: right, all columns | Pen moved: right, accepted | Accepted, all lines | Right, accepted | Flat line, same columns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sam3_both_directions` | 0.821 | 78.4% | 95.0% [93.9, 96.2] | 76.7% | 61.0% | 79.9% | 94.0% | 98.0% | 93.5% |
| `sam3_plus_offline_network` | 0.819 | 75.7% | 97.5% [96.5, 98.2] | 80.5% | 60.4% | 88.1% | 92.6% | 99.2% | 95.4% |
| `sam3_forward` | 0.821 | 100.0% | 83.8% [77.2, 89.5] | 66.4% | 61.0% | 61.0% | 100.0% | 94.1% | 91.0% |
| `classical_tracker` | 0.022 | 100.0% | 31.8% [22.6, 40.2] | 66.4% | 8.7% | 8.7% | 100.0% | 57.4% | 91.0% |
| `flat_line` | 0.000 | 100.0% | 66.4% [57.9, 73.3] | 66.4% | 0.0% | 0.0% | 100.0% | 91.0% | 91.0% |

- `sam3_both_directions`: SAM 3 fine-tuned by us, traced from both ends, accepted where the passes agree (τ 0.25 S, at least 20 columns)
- `sam3_plus_offline_network`: The earlier rule, SAM 3 checked by a small network of ours, with the SAM 3 paths started at the true pen position. Retired: every model that runs in our product must be a Meta model
- `sam3_forward`: The same SAM 3, left to right only, every column accepted
- `classical_tracker`: Our classical tracker, no learning, started at the true pen position. Built for an earlier film geometry and not tuned again
- `flat_line`: No skill: the baseline as the answer

## holdout4

Development set: 10 other earthquakes, 100 images. The rule was chosen here, so these rows are development numbers.

| Baseline | Earthquake line r | Accepted, earthquake line | Right, accepted | Flat line, same columns | Pen moved: right, all columns | Pen moved: right, accepted | Accepted, all lines | Right, accepted | Flat line, same columns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sam3_both_directions` | 0.843 | 84.9% | 94.2% [93.6, 94.8] | 72.7% | 65.3% | 80.0% | 96.9% | 97.8% | 93.2% |
| `sam3_plus_offline_network` | 0.825 | 77.2% | 97.8% [97.3, 98.1] | 79.1% | 63.1% | 90.1% | 93.9% | 99.3% | 95.5% |
| `sam3_forward` | 0.843 | 100.0% | 85.7% [81.4, 89.4] | 65.8% | 65.3% | 65.3% | 100.0% | 96.1% | 91.4% |
| `classical_tracker` | 0.017 | 100.0% | 26.9% [20.3, 34.3] | 65.8% | 10.6% | 10.6% | 100.0% | 54.6% | 91.4% |
| `flat_line` | 0.000 | 100.0% | 65.8% [63.0, 68.6] | 65.8% | 0.0% | 0.0% | 100.0% | 91.4% | 91.4% |

- `sam3_both_directions`: SAM 3 fine-tuned by us, traced from both ends, accepted where the passes agree (τ 0.25 S, at least 20 columns)
- `sam3_plus_offline_network`: The earlier rule, SAM 3 checked by a small network of ours, with the SAM 3 paths started at the true pen position. Retired: every model that runs in our product must be a Meta model
- `sam3_forward`: The same SAM 3, left to right only, every column accepted
- `classical_tracker`: Our classical tracker, no learning, started at the true pen position. Built for an earlier film geometry and not tuned again
- `flat_line`: No skill: the baseline as the answer

## Real film (no ranking)

Three chips with a digital record at the same site, frozen both-directions rule, against the digital record. Far too few chips to rank anything. Scores come after a low-pass filter, one global amplitude fit and a per-line offset and trend fit. Time alignment and sign were fitted to the same digital record with a classical tracker. A path on the wrong line can still score well in the 0.1 S column. KIP and ALQ images and the digital records are not in this repository, so this table cannot be recomputed from it alone. ALQ is by line type only. Accepted and flagged r are means over lines.

| Chip, lines | Accepted share | r, accepted | r, flagged | Within 0.1 S, accepted: SAM 3 | Flat line |
|---|---:|---:|---:|---:|---:|
| MAT, earthquake line | 0.0% | – | +0.02 | – | – |
| MAT, lines next to it | 52.8% | +0.08 | -0.00 | 100% | 100% |
| MAT, other active lines | 70.3% | +0.69 | +0.43 | 63% | 62% |
| MAT, quiet lines | 99.8% | +0.28 | – | 95% | 99% |
| KIP, earthquake line | 50.4% | -0.22 | +0.23 | 29% | 89% |
| KIP, lines next to it | 49.9% | +0.17 | +0.11 | 79% | 79% |
| KIP, other active lines | 59.5% | +0.35 | +0.01 | 62% | 61% |
| KIP, quiet lines | 94.2% | +0.05 | -0.07 | 100% | 100% |
| ALQ, earthquake line | 90.9% | +0.91 | +0.68 | 70% | 43% |
| ALQ, lines next to it | 98.9% | +0.79 | – | 91% | 78% |
| ALQ, other active lines | 66.3% | +0.66 | +0.04 | 91% | 83% |
| ALQ, quiet lines | 92.6% | +0.11 | -0.07 | 100% | 100% |

Source: [results/both_directions/real_results.json](results/both_directions/real_results.json), [results/flat/flat_real.json](results/flat/flat_real.json).
