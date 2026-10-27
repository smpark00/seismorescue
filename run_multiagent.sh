#!/usr/bin/env bash
# Reproduce the 10-27 multi-agent pipeline and its grouped cross-validation (research/16, 17, 18).
#   ./run_multiagent.sh            # everything (network for data, Meta API for Spark steps; ~2 h, ~$6)
#   SKIP_API=1 ./run_multiagent.sh # data + renders + offline judges only
# Order: benchmark data -> renders -> proposer cache + stage-1 CV -> learned judge CV -> Spark judge (stage 2)
#        -> catalogue + digitized series -> seismologist agent (2 runs) -> evaluation.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"; S="$HERE/scripts"; O="$HERE/out"
[ -x "$PY" ] || { echo "virtualenv missing: run ./setup_env.sh first"; exit 1; }

echo "== 1) benchmark waveforms (12 events x 2 stations) + renders"
[ -d "$HERE/data/bench" ] || "$PY" -I "$S/fetch_benchmark.py" "$HERE/data/bench" 12 2
for f in "$HERE"/data/bench/*.mseed; do t=$(basename "$f" .mseed)
  [ -d "$O/bench/$t/all" ] || "$PY" -I "$S/render_helicorder.py" "$f" "$O/bench/$t/all" > /dev/null 2>&1
  [ -d "$O/bench/$t/active" ] || "$PY" -I "$S/render_helicorder.py" "$f" "$O/bench/$t/active" active > /dev/null 2>&1
done
echo "== 2) proposers A/B (cached) + stage-1 grouped CV (rule arbiter, oracle)"
"$PY" -I "$S/cv_arbiter.py" "$O/bench" "$O/cache_props" "$O/cv_arbiter_stage1.json" 2>&1 | grep -v -i warn | tail -8
echo "== 3) learned judge, grouped CV (writes CV-held-out event predictions)"
"$PY" -I "$S/cv_judges.py" "$O/bench" "$O/cache_props" "$O/judges" 2>&1 | grep -v -i warn | grep -v 'yi = '
echo "== 4) catalogue per record + digitized series from the held-out judge"
[ -d "$HERE/data/bench_catalog" ] || "$PY" -I "$S/seismo_tools.py" prefetch "$HERE/data/bench" "$HERE/data/bench_catalog"
"$PY" -I "$S/seismo_tools.py" series "$O/bench" "$O/cache_props" "$O/judges/event_predictions.json" "$O/series" > /dev/null 2>&1

if [ "${SKIP_API:-0}" = "1" ]; then echo "SKIP_API=1 -> skipping Spark steps"; exit 0; fi
if [ -z "${MODEL_API_KEY:-}" ]; then
  MODEL_API_KEY="$(security find-generic-password -a "$USER" -s meta-model-api -w 2>/dev/null || true)"; export MODEL_API_KEY
fi
echo "== 5) Spark A/B judge on the 15% least-confident events (stage 2) + scoring"
"$PY" -I "$S/exp_spark_event_judge.py" "$S" "$O/bench" "$O/cache_props" "$O/judges" 0.15 40 2>&1 | grep -v -i warn | tail -2
"$PY" -I "$S/eval_spark_stage.py" "$O/bench" "$O/cache_props" "$O/judges" 2>&1 | grep -v -i warn | grep -v 'yi = '
echo "== 6) seismologist agent (Spark + tools), two independent runs, then scoring"
for run in run1 run2; do "$PY" -I "$S/seismologist_agent.py" "$S" "$O/series" "$HERE/data/bench_catalog" "$O/agent" $run 16 2>&1 | tail -1; done
"$PY" -I "$S/eval_seismologist.py" "$O/series" "$HERE/data/bench_catalog" "$O/agent" run1 run2 2>&1 | grep -v -i warn
