# SeismoBench

Score your own tracer on synthetic seismogram film with exact ground truth. Results: [LEADERBOARD.md](../LEADERBOARD.md).

The truth is public, so scores rest on trust. There are no outside participants so far; every row is one of our own baselines.

## Quick start

Needs NumPy and Python 3.12, 3.11.4+ or 3.10.12+ (the fetcher's safe unpacking needs them). The data is about 240 MB for the three splits, plus the baselines, in this repository's GitHub Release, with the same files on Hugging Face ([seungminpark/seismobench](https://huggingface.co/datasets/seungminpark/seismobench)). `SEISMOBENCH_URL` points the fetcher at a mirror; the hashes are checked either way.

```bash
python -I bench/fetch_bench.py holdout7 baselines
python -I bench/score.py bench_data/holdout7 bench_data/baselines/holdout7/sam3_both_directions --name sam3_both_directions
```

From the Hugging Face mirror:

```bash
SEISMOBENCH_URL=https://huggingface.co/datasets/seungminpark/seismobench/resolve/main python -I bench/fetch_bench.py holdout7 baselines
```

## Data

| Split | Images | Use |
|---|---:|---|
| `holdout7` | 95 | Main table. 10 earthquakes (M7.5–8.8) |
| `holdout6` | 90 | 10 other earthquakes |
| `holdout4` | 100 | Development set; 10 other earthquakes |

The same earthquake never appears in two splits. A split holds, per record, `film/heli_g<gain>.png` at five amplitudes (1×, 2×, 4×, 6×, 8×), `film/gt_g<gain>.npz` (`pen`: the pen's row per line and column; `sig`: the signal without the pen's minute marks; `mark`: where the minute marks are) and `film/meta.json` (`spacing`, `margin_x`, `margin_y`, `n_lines`, `gains`, `active_line`: the line with the earthquake, and the renderer's settings).

The films are rendered from public broadband records: IRIS/EarthScope FDSN, networks IU ([doi:10.7914/SN/IU](https://doi.org/10.7914/SN/IU)) and II ([doi:10.7914/SN/II](https://doi.org/10.7914/SN/II)).

## Your submission

One file per image, `<record>_g<gain>.npz`:

| Array | Type, shape | Meaning |
|---|---|---|
| `path` | float, lines × columns | The trace's row per column. NaN where you give no answer |
| `accept` | bool, same shape, optional | The columns you stand behind. Without it, every column with a value counts as accepted |

```bash
python -I bench/score.py bench_data/holdout7 my_tracer/ --name my_tracer --out bench/results/holdout7/my_tracer.json
python -I bench/make_leaderboard.py
```

A row takes its name from the result file. Definitions are at the top of [score.py](score.py). A rule that rejects columns reports how many it kept and how right those are, next to a flat line on the same columns. Please report both.

## How the baselines were made

[build/](build) holds the scripts that turned our stored paths into submissions and packed the bundles. They need those paths under `out/`, which are not in this repository: `scripts/ml_fb.py dump` makes the two SAM 3 passes (gated weights and a GPU). The submissions themselves are in the release, so you can score them without any of this.

Rebuilding on an Apple laptop gave the same arrays for 1,294 of the 1,330 submission files. The other 36 are all `holdout7/classical_tracker`: `build/extra.py` runs that tracker again instead of reading stored paths, and it is not bit-identical across machines (earthquake line right 23.25% against the release's 22.95%). The release files are the reference.

## Licence

The code is MIT. The film images and truth files are our renderings, released under the same licence.

We found no licence declared by either source network (DataCite and the FDSN network pages, checked 2026-11-03). EarthScope states that it distributes such data under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([registry entry](https://registry.opendata.aws/earthscope-geophysical-data/)). The CC BY 4.0 attribution still applies: if you use or share the films, cite the two networks above and say that the films were rendered from their records. The same EarthScope statement asks users to cite the facility as well.

The SAM 3 baseline paths are outputs of a model under Meta's SAM License.
