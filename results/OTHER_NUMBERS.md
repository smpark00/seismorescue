# Numbers without a result file

Most numbers in the README can be traced to a file under `results/`, `examples/` or `bench/results/`. These cannot. They were measured during the work and written down in our notes, which are not published. KIP and ALQ long-period records stay local, so some of their numbers can only be listed here.

| Number | What it is | How it was measured |
|---|---|---|
| ALQ 1975 amplitude slope 0.55 | Earthquake line, SAM 3 against the digital record | Real-film evaluation, forward pass |
| ALQ 1975 r +0.76 | The same line with the earthquake line and the lines next to it left out of the time fit | Refit of the stored alignment, separate review |
| ALQ 1975 at most 0.30, side peak 0.47 | Controls for r +0.75: 1,500 circular shifts at most 0.262, 2,000 phase scrambles at most 0.299; outside the peak of a ±600 s lag scan at most 0.47 | Same review |
| RA2 r 0.15, 0.48, +0.50 | The retired network on the MAT, KIP and ALQ earthquake lines. Given in [archive/offline_baselines](../archive/offline_baselines/README.md), not in the README | Real-film evaluations of the earlier rule |
| ALQ 1973 within 3.03 s, median −1.6 s, +0.05%, 3588.05 s vs 3588.0 s | L0 timing on the held-out chip | L0 test against the co-located digital record |
| MAT within 4 s | L0 timing on the development chip (3.99 s) | Same check |
| ALQ 1975 within 4.0 s, median −1.4 s, 25 lines | L0 automatic mode after the polarity fix (4.00 s, −1.38 s) | Regression check |
| 0.848 vs 0.802, +0.045 [+0.015, +0.079] | SAM 3 vs SAM 2.1 on unseen earthquakes, earlier renderer | Paired by earthquake |
| +1 s r 0.52; +2 s r 0.74, +1 s r 0.63 | Timing check of the MAT export against the 1973 digital record, whole and in the two segments with signal | `export_mseed.py paircheck` (the digital record is not in this repository) |
| 0.049 S | Median step at 18 line seams | Separate review of the export |
| 10 of 30, 3 of 30 | Public long-period chips in the 30-minute format, and chips where automatic layout failed | L0 on a random sample of 30 chips |
| 62.7% → 20.5%, flat line 28.7% | Earthquake-line failure rate before and after fine-tuning on film2 validation | SAM 3 evaluation; `scripts/flat_baseline.py` |
| 14.8 GB; 16.8 GB and 1.4 GB | Glimmer layers on the GPU, seen when the server started; file sizes of the two GGUF files | Server log; file sizes |
| 0.56 → 0.98, 0.985 without ink | Glimmer QLoRA pilot on 200 validation crops | Modal run |
| About $15, about $0.60 | Modal cost of all SAM 3 runs, and of the QLoRA pilot | `modal billing summary` before and after |
| 143 s, 9.5 min | An earlier version of step 1, one SAM 3 pass, on the RTX 5070 Ti and on an Apple-silicon laptop (567 s) | Timed runs |
| About 4–5 s | One re-track on the laptop's Apple GPU, on the review page | Timed clicks, not an evaluation |
| About 189,000 | Film chips in the scanned selection: 189,180 on a [2019 USGS page](https://www.usgs.gov/programs/earthquake-hazards/science/preserving-historic-earthquake-records-and-photos-instruments) and in [ASL's notes](https://lsd-sphinx.readthedocs.io/en/latest/find_data/organizations/wwssn.html). That USGS page gives 3.7 million chips for the whole collection and calls the selection 5% of it. The [release page](https://www.usgs.gov/data/world-wide-standardized-seismographic-network-film-chip-scans) words it differently and gives no chip count | Not counted by us |
| All 4 are positives | The 4 of 300 new chips that L0 read as negatives | Looked at by eye, not scripted |

Two stored medians differ from the README. `seconds_median` in `results/e7/e7_results.json` is 118.5 and `secs_median` in the Glimmer results is 134.5, because those fields take the upper of the two middle rows. The README gives the true medians, 117 s (116.7) and 134 s (133.85).
