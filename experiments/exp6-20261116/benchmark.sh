#!/usr/bin/env bash
set -euo pipefail
: "${1:?Usage: benchmark.sh RESULT_DIR}"
out=$(realpath -m "$1")
mkdir -p "$out"
echo "SNR accumulation results: $out"
for size in 1000000 494032768; do
  uv --no-config run --no-project --active python benchmarks/snr/run.py "$out/parameters-$size.json" --parameters "$size" --tensor-numel 1000000 --rollouts 32 --pool-size 2 --dtype bfloat16 --threads 4 --warmup 1 --repeats 3
done
