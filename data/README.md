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

## Real paired scans (not shipped; how to get them)

| Scan | Where | Licence | Digital pair |
|---|---|---|---|
| MAT LP-Z, 1973-12-28 (M7.5 Vanuatu) | EarthScope film-chip product: `https://data.earthscope.org/archive/seismology/products/film_chip/data/MAT/MAT_73_12_28_0020_LHZ.TIFF` | USGS WWSSN film-chip scans are CC0 1.0 ([doi:10.5066/P13PHZMA](https://doi.org/10.5066/P13PHZMA)); the EarthScope pilot set ([doi:10.17611/DP/FC.1](https://doi.org/10.17611/DP/FC.1)) is documented as part of that release | HG.MAT, LH? at 1 sps, from EarthScope FDSN dataselect |
| KIP LP-Z, 1975-02-02 (M7.6 Attu) | USGS WWSSN film-chip scans (ScienceBase/Globus). We used a mirror copy, which we do not redistribute. | CC0 1.0 for the USGS release | HG.KIP, LH? at 1 sps |

- Put the scans in `data/real_scans/` (MAT) and `data_local/real_scans/` (KIP).
- Put the digital pairs in `data_local/real_scans/pairs/` as `<scan>__<NET.STA>.mseed` plus `.stationxml`.
- Then run `scripts/real_paired_eval.py MAT` or `KIP`.
- This repository contains metrics for KIP but no KIP image.
