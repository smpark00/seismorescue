#!/usr/bin/env bash
# Create the experiment virtualenv inside this repository (.venv).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
command -v uv >/dev/null || { echo "uv not found: brew install uv"; exit 1; }
[ -d .venv ] || uv venv -q -p 3.12 .venv
uv pip install -q -p .venv/bin/python -r requirements.txt
.venv/bin/python -c "import numpy, cv2, skimage, obspy, torch, meta_sam_parser; print('env ok:', numpy.__version__, cv2.__version__, obspy.__version__)"
