# Data

## Shipped in this repository (7 MB)

| Path | Contents | Source |
|---|---|---|
| `film_dev/` | 43 eight-hour LHZ records (miniSEED) of 22 earthquakes (M7.5–8.8), plus a JSON sidecar per record (event, station, P arrival, window start). Rendered into the training/validation films. | IRIS/EarthScope FDSN, networks IU and II |
| `film_holdout3/` | 19 records of 10 further earthquakes: the frozen-method test set | same |
| `bench/`, `bench_holdout/`, `bench_holdout2/`, `bench_holdout3/` | Shorter records for the first-geometry benchmark (development and the pre-registered holdouts) | same |
| `bench_catalog/` | Event catalogue per record (USGS ComCat) for the seismologist agent | USGS |
| `tohoku_IU_ANMO_LHZ.mseed` | 2011 Tohoku M9.1 recorded at IU.ANMO, used by the first experiments | IRIS/EarthScope |

Rendered films (`out/film_train`, `out/film_holdout3`) are not shipped. They are generated deterministically by `scripts/render_helicorder.py`; see the README's Reproduce section.

Waveform citations:
- IU: Albuquerque Seismological Laboratory/USGS (1988), Global Seismograph Network (GSN – IRIS/USGS), [doi:10.7914/SN/IU](https://doi.org/10.7914/SN/IU).
- II: Scripps Institution of Oceanography (1986), Global Seismograph Network – IRIS/IDA, [doi:10.7914/SN/II](https://doi.org/10.7914/SN/II).

## Public film chips

| Path | Contents |
|---|---|
| `real_scans/MAT_73_12_28_0020_LHZ.TIFF` + `.json` | The example chip (26 MB) and its CC0 provenance sidecar. Shipped so the demo works if the archive moves. Source: `https://data.earthscope.org/archive/seismology/products/film_chip/data/MAT/MAT_73_12_28_0020_LHZ.TIFF` |
| `real_scans/ALQ_75_10_27_1551_SHZ.json` | Sidecar of the second development chip. Fetch the image from `.../film_chip/data/ALQ/ALQ_75_10_27_1551_SHZ.TIFF` into `real_scans/` |
| `wwssn_stations/wwssn_stations.json` | Station-number table used by the L1 agent. See `THIRD_PARTY_NOTICES.md` |

USGS WWSSN film-chip scans are CC0 1.0 ([doi:10.5066/P13PHZMA](https://doi.org/10.5066/P13PHZMA)); the EarthScope film-chip product is [doi:10.17611/DP/FC.1](https://doi.org/10.17611/DP/FC.1).

**Other chips are not shipped.** Fetch them when you need them:

```bash
# the 30 chips of E4 (about 1 GB), from the repository root
.venv/bin/python -I scripts/fetch_e4_chips.py data/e4_chips
# the 20 new chips of E7
.venv/bin/python -I scripts/fetch_e4_chips.py data/e7_chips --seed 20261030 --n 20 --per-station 1 --exclude-index data/e4_chips/index.json
```

The ALQ 1975 long-period chip of the real-film table is at `.../film_chip/data/ALQ/ALQ_75_10_27_1551_LHZ.TIFF`. Its images are not published here: this repository gives numbers only.

The scripts send a chip to a Meta API only if it has a sidecar that records a CC0 licence and an EarthScope source (`l1_context.guard`).

## Paired digital records and local scans (not shipped)

- The digital records used for comparison (networks HG and SR) are fetched by the user from the EarthScope FDSN services. Put them in `data_local/real_scans/pairs/` as `<scan>__<NET.STA>.mseed` plus `.stationxml`.
- The KIP scan and the calibration scans are local copies. They are not in this repository, and we publish numbers only. `scripts/real_paired_eval.py` and `scripts/ml_sim2real_calib.py` hold only their settings.
- Data acknowledgement: "Data were accessed from the NSF NGF data archive operated by EarthScope Consortium (NSF award 2435260)."
