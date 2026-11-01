#!/bin/bash
# Build the SeismoBench submissions, score them and pack the release bundles. CPU only. Run from the repository root,
# with the stored paths and the rendered holdouts under out/ (bench/README.md).
set -eu
export OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
P=.venv/bin/python
W=bench_build
mkdir -p "$W/results"
$P -I bench/build/baselines.py "$W/baselines"
$P -I bench/build/extra.py "$W/baselines"
for s in holdout7 holdout6 holdout4; do
  for b in "$W/baselines/$s"/*; do
    n=$(basename "$b")
    $P -I bench/score.py "out/film2_$s" "$b" --name "$n" --out "$W/results/$s/$n.json"
  done
done
mkdir -p "$W/release"
for s in holdout7 holdout6 holdout4; do
  tar -cf "$W/release/$s.tar" --owner=0 --group=0 --numeric-owner --transform "s#^out/film2_$s#$s#" out/film2_$s
done
tar -cf "$W/release/baselines.tar" --owner=0 --group=0 --numeric-owner -C "$W" baselines
(cd "$W/release"; sha256sum *.tar | sed 's/  / /' > ../SHA256SUMS)
