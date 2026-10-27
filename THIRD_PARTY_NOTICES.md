# Third-party notices

## SAM 2.1 (Meta)
- `checkpoints/sam21_tiny_ft/ckpt_tiny_best.pt` contains weights fine-tuned from Meta's SAM 2.1 hiera-tiny checkpoint.
  - Source: https://github.com/facebookresearch/sam2
  - Licence: Apache License 2.0
  - The base checkpoint is downloaded at run time from `dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt`.
- `scripts/ml_sam.py` re-implements the SAM 2.1 single-image path under the official parameter names so that the checkpoint loads.
  - It is an independent re-implementation, not a copy of the SAM 2 source.
- Apache License 2.0: https://www.apache.org/licenses/LICENSE-2.0

## Meta Model API
- Muse Spark 1.3 and SAM 3.1 are called through the Meta Model API (`https://api.meta.ai/v1`) under Meta's terms.
- No API key is stored in this repository. Set `MODEL_API_KEY` in your environment.

## Data
- Seismic waveforms: IRIS/EarthScope FDSN web services.
  - Networks IU ([doi:10.7914/SN/IU](https://doi.org/10.7914/SN/IU)) and II ([doi:10.7914/SN/II](https://doi.org/10.7914/SN/II)).
  - The 1970s HGLP pairs (network HG) are fetched by the user and not redistributed.
- WWSSN film-chip scans: U.S. Geological Survey, "World-Wide Standardized Seismographic Network Film Chip Scans".
  - USGS data release, [doi:10.5066/P13PHZMA](https://doi.org/10.5066/P13PHZMA), CC0 1.0.
  - The EarthScope film-chip product is [doi:10.17611/DP/FC.1](https://doi.org/10.17611/DP/FC.1).
- Event catalogue: USGS ComCat.

## Python packages
- See `requirements.txt`: numpy, scipy, matplotlib, opencv-python-headless, scikit-image, ObsPy, requests, Pillow, PyTorch, meta-sam-parser.
- Each is used under its own licence.
