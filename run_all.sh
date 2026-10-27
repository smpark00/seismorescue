#!/usr/bin/env bash
# Reproduce the 2026-10-27 SeismoRescue experiments end to end.
#   ./run_all.sh                 # full run (SAM 3.1 + Spark contributor calls, ~10-12 min, < $0.1)
#   SKIP_API=1 ./run_all.sh      # offline part only (download/render/classical baselines)
#   REAL_SCAN=/path/scan.png ./run_all.sh   # use another real scan for the real-scan test
#   OUT=/tmp/x ./run_all.sh      # change output folder (default: ./out, gitignored)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
OUT="${OUT:-$HERE/out}"
DATA="$HERE/data/tohoku_IU_ANMO_LHZ.mseed"
[ -x "$PY" ] || { echo "virtualenv missing: run ./setup_env.sh first"; exit 1; }

echo "== 1) waveform"; [ -f "$DATA" ] && echo "using $DATA" || "$PY" -I "$HERE/scripts/fetch_waveform.py" "$DATA"
echo "== 2) render synthetic helicorders (gains 0.25/0.8/2/5)"; "$PY" -I "$HERE/scripts/render_helicorder.py" "$DATA" "$OUT/synth" 2>&1 | grep -v Warning
echo "== 2b) single-active-line helicorders (gains 1/1.5/2)"; "$PY" -I "$HERE/scripts/render_helicorder.py" "$DATA" "$OUT/synth_active" active 2>&1 | grep -v Warning
echo "== 3) classical baselines"; "$PY" -I "$HERE/scripts/digitize_baseline.py" "$OUT/synth"
echo "== 3b) classical baselines, single active line"; "$PY" -I "$HERE/scripts/digitize_baseline.py" "$OUT/synth_active"
if [ -f "$HERE/results/sam_crossings_results.json" ]; then echo "== 3c) joint tracker spike (S1)"; "$PY" -I "$HERE/scripts/track_joint.py" "$OUT/synth" "$OUT/synth_active" "$HERE/results/sam_crossings_results.json" 2>&1 | grep -v -i warn; fi

if [ "${SKIP_API:-0}" = "1" ]; then echo "SKIP_API=1 -> skipping SAM 3.1 / Spark steps"; exit 0; fi
if [ -z "${MODEL_API_KEY:-}" ]; then
  MODEL_API_KEY="$(security find-generic-password -a "$USER" -s meta-model-api -w 2>/dev/null || true)"; export MODEL_API_KEY
fi
[ -n "${MODEL_API_KEY:-}" ] || { echo "MODEL_API_KEY not available (Keychain item 'meta-model-api')"; exit 1; }

echo "== 4) SAM 3.1 concept/box/point tests"; "$PY" -I "$HERE/scripts/exp_sam.py" "$HERE/scripts" "$OUT/synth"
echo "== 5) SAM vs classical centreline (same metrics)"; "$PY" -I "$HERE/scripts/exp_sam_centerline.py" "$HERE/scripts" "$OUT/synth"
echo "== 5b) SAM where lines actually touch"; "$PY" -I "$HERE/scripts/exp_sam_crossings.py" "$HERE/scripts" "$OUT/synth" "$OUT/synth_active"
echo "== 6) Spark contributor tests"; "$PY" -I "$HERE/scripts/exp_spark.py" "$HERE/scripts" "$OUT/synth"
echo "== 6b) Spark crossing judge (S2)"; "$PY" -I "$HERE/scripts/exp_spark_crossing_judge.py" "$HERE/scripts" "$OUT/synth" "$OUT/synth_active" "$OUT/spark_judge" 40 2>&1 | grep -v -i warn
REAL="${REAL_SCAN:-$HERE/data_local/dummy-seismo-small.png}"
if [ -f "$REAL" ]; then
  echo "== 7) real scan test: $REAL"; "$PY" -I "$HERE/scripts/exp_real.py" "$HERE/scripts" "$REAL" "$OUT/real"
  echo "== 7b) real quake section: zoom/median SAM retest (S3)"; "$PY" -I "$HERE/scripts/exp_real_sam_zoom.py" "$HERE/scripts" "$REAL" "$OUT/real_zoom"
else
  echo "== 7) real scan not found ($REAL) -> skipped (see data_local/README.md)"
fi
echo "done -> $OUT"
