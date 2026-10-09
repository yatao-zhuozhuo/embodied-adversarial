#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/environment.sh"
cd "$BASELINE_REPO"
exec "$BASELINE_REPO/baseline/.runtime/coordinator/bin/python" -m sim.mcp_server \
  --host 127.0.0.1 --port "${BASELINE_SIM_PORT:-8220}"
