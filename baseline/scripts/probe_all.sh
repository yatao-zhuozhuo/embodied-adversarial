#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$BASELINE_REPO"
status=0
for bench in "${@:-libero metaworld maniskill3 robocasa}"; do
  # No arguments means all four; explicit arguments select individual suites.
  for name in $bench; do
    worker="$name"
    [[ "$name" != maniskill3 ]] || worker=maniskill
    for pass in environment-preflight environment-repeat; do
      mkdir -p "baseline/results/$pass"
      for index in {0..9}; do
        if ! CUDA_VISIBLE_DEVICES=3 timeout 300 "sim/venvs/$worker/bin/python" \
          baseline/scripts/environment_probe.py \
          --manifest "baseline/manifests/${name}_10.json" --index "$index" \
          --output "baseline/results/$pass/${name}-${index}.json"; then
          status=1
        fi
      done
    done
  done
done
exit "$status"
