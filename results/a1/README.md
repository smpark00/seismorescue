# A1: the frozen L0 and L1 on 300 new public chips

One run, unattended, on chips we had not seen. There were no pass or fail thresholds. These chips have no ground-truth times, so this run says nothing about time accuracy.

**Sample.** `scripts/fetch_e4_chips.py data/a1_chips --seed 20261101 --n 300 --per-station 4 --max-gb 14`, excluding the E4 and E7 chips, the MAT 1973-12-28 chip and the KIP station. It walks the public EarthScope LHZ chips in a seeded random order and takes at most 4 per station, without looking at the images. So this is a random sample with a cap per station, not a simple random sample of the archive. 300 chips, 105 stations. The list is in `index.json`.

**Run.** `scripts/a1_batch.py run`, 4 at a time, on a CPU. Muse Spark 1.3 (contributor tier) saw public CC0 chips only. `started.json` holds the hashes of the code that ran. An API or network failure may be retried and reported; no other result was changed. The one retry was `BOG_63_04_07_1314`, the damaged file described below, and it failed the same way.

| Result | Value | Interval (Wilson 95%) |
|---|---|---|
| Agent finished with a valid answer | 297/300 (99.0%) | [97.1, 99.7] |
| Within the call budget | 299/300 | |
| Retries | 1 | |
| Agent's value differs from the file name | 45/297 (15.2%) | [11.5, 19.7] |
| Agent gave no value for a field, nothing contradicts the name (not counted above) | 24/297 (8.1%) | [5.5, 11.7] |
| **Checked on the chip: the file name differs from the prints** | **at least 32/297 (10.8%)** | [7.7, 14.8] |
| Line format supported by L0 (one line per hour) | 167/297 (56.2%) | [50.5, 61.8] |
| L0 time model | 164/297 (55.2%) | [49.5, 60.8] |
| L0 layout failed | 31/297 (10.4%) | [7.5, 14.4] |
| L0 found no minute marks | 27/297 (9.1%) | [6.3, 12.9] |
| L0 flagged an unsupported format | 72/297 (24.2%) | [19.7, 29.4] |
| Line period within 3588 ± 1 s, supported chips | 152/167 (91.0%) | [85.7, 94.5] |
| Line-count check disagrees, chips with a time model | 78/164 (47.6%) | [40.1, 55.2] |
| Line-count check disagrees, chips where the OFF time could be read | 78/113 (69.0%) | [60.0, 76.8] |
| Widest-swing line within ±1 of the predicted surface-wave line (a weak check) | 40/123 (32.5%) | [24.9, 41.2] |
| Cost, median time per chip | $1.06, 113.5 s | |

The L0 rows count the chips where L0 ran, which are not quite the agent's 297: `DUG_76_03_09_0001` (no valid answer) is among them and `ADE_64_02_11_0017` (valid answer, L0 not run) is not. On chips with both, the counts are 166 and 163 of 296, 151/166 and 78/163.

## Checking the 45 mismatches

Spark read each of the 45 chips a second time (`a1_reread.json`). None of the second reads matched the file name in all four fields; 35 repeated the first read exactly.

Then two Claude sessions read all 45 chips, each without seeing the file name, Spark's reads or the other session's answers. They copied the border digits, the label and the date stamps on the film (`reader_A.json`, `reader_B.json`; free-text notes removed). `scripts/a1_name_checks.py` compares those readings with the name, using only values both readers agree on. **The judges are Claude, not independent people.**

| Verdict | All 45 | Pre-registered sample of 30 |
|---|---|---|
| The file name differs from the prints | **32** | **23** |
| The chip supports the name on every field Spark flagged: Spark misread it | 8 | 6 |
| Unclear | 5 | 1 |

- The pre-registered plan was a seeded sample of 30 (`check_sample.json`), which gives an estimate of 34.5 chips of 297 (11.6%). We decided before seeing any reading to read the other 15 as well. The 32 is a count of chips seen, not an estimate.
- **We fixed one bug after seeing the results.** The first version of the script sent chips with an ambiguous day-month border to "unclear" even when a date stamp settled it, against the rule we had written down. Fixing it moved 4 chips to "differs" (first result 28 / 8 / 9). Three of the 4 were in the sample, which went from 20 / 6 / 4 to 23 / 6 / 1 and from an estimate of 30.0 chips to 34.5. The readings did not change.
- Of the 32: date 22, time 10, station 2, component 2 (some chips have two). Four look like a swapped month and day: chips filed as `63_01_05` are stamped May 1 1963, and chips filed as `63_02_04` are stamped Apr 2 1963.
- "Differs" means the name differs from what the chip prints. We do not say which side is right. The list is in `name_checks.json`, whose `name_wrong` and `estimate_name_wrong` mean this and nothing more.
- **The agent is wrong too:** at least 8 of the 45 flags are Spark's own misreads, mostly handwritten station codes. On one chip with an underexposed border (`NUR_64_02_11_0548`) it reported digits that are not visible.

## What else this run showed

- **All 4 chips L0 read as negatives are positives.** We looked at them: dark traces on a lighter ground. Our polarity rule assumes traces are rare, and dense or stained chips break that. None of the four produced times.
- **Three chips failed.** `BOG_63_04_07_1314` is truncated in the archive: its last image strip ends at byte 28,226,274 and the file has 28,222,914, as the archive's own metadata says. On `ADE_63_02_04_0056` Spark's digits did not decode. On `DUG_76_03_09_0001` it named a main event while the catalogue was empty.
- **"Unsupported format" is L0's flag, not a checked fact.** Most of the 72 read as about 30 or 120 minutes per line, and 120 is how L0 misreads 30-minute-line film. But 20 of the 50 that read about 120 have only 24 to 30 lines, so some may be hour-per-line chips it mismeasured.

## Changes after this run

The run was not repeated after these. The catalogue query now asks USGS for earthquakes only, and an unreadable image now gives a clear error. The results here come from the code hashed in `started.json`. The scripts in this repository differ from it by those changes, by path handling, by a `SPARK_MODEL` environment override in `metaapi.py` and by a named constant in `real_paired_eval.py`. `index.json` differs from its hash in one field: `script`, shortened to remove a home folder. `doc_sha256` is the hash of our internal plan, which is not published. The raw per-chip records stay out of this repository.
