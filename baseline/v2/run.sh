#!/usr/bin/env bash
# Existing inference API must be ready. This entry point never starts another machine.
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$BASELINE_REPO"
if [[ $# -lt 1 ]]; then
  echo 'Usage: bash baseline/v2/run.sh MODEL_ID [--benchmark BENCH] [--limit N]'
  exit 2
fi
v2_model="$1"
shift
exec "$BASELINE_REPO/baseline/.runtime/coordinator/bin/python" -m baseline.v2.runtime \
  --model "$v2_model" --root "$V2_DIR/results/formal" \
  --sim-url "${BASELINE_SIM_URL:-http://127.0.0.1:8220/mcp}" "$@"
