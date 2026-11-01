#!/usr/bin/env bash
# Put the shipped checkpoints, results and example outputs where the scripts look for them (out/ is not tracked by git).
# Safe to run again: existing files in out/ are kept, so a run of your own is never overwritten.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

copy() {  # copy <source file or folder> <destination folder>
  mkdir -p "$2"
  cp -Rn "$1" "$2"/ 2>/dev/null || true
}

copy checkpoints/sam3_film2/ckpt_sam3_film2_best.pt out/ml/sam3_film2_modal
copy checkpoints/sam3_film2/frozen_method.json      out/ml/sam3_film2
copy checkpoints/both_directions/frozen_rule.json out/ml/fb
copy results/e4/e4_results.json             out/e4
copy results/a1/a1_results.json             out/a1
copy results/a1/check_sample.json           out/a1
copy examples/MAT_73_12_28_0020_LHZ         out/rescue

for f in out/ml/sam3_film2_modal/ckpt_sam3_film2_best.pt out/ml/fb/frozen_rule.json out/e4/e4_results.json \
         out/rescue/MAT_73_12_28_0020_LHZ/paths.npz; do
  [ -f "$f" ] || { echo "missing after copy: $f"; exit 1; }
done
echo "bootstrap ok: checkpoints, results and the MAT example are in out/"
