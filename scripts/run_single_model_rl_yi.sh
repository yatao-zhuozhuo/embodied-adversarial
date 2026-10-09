#!/usr/bin/env bash
# Run on the designated 8-H200 Yicloud workspace; retain original output layout.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/single_model_rl_yi_env.sh"
cd "$REPO_ROOT"
export RUN_ROOT="${RUN_ROOT:-$OPENETA_TENSORBOARD_SERIES_ROOT}"
export OPENETA_HOST_TOKEN_FILE="$RUN_ROOT/service/.host_token"
mkdir -p "$RUN_ROOT/service"
"$SWIFT_VENV/bin/python" scripts/submit_single_model_rl.py \
  --kind mcp --run-root "$RUN_ROOT/service"
for attempt in $(seq 1 120); do
  if curl --noproxy '*' -fsS "$OPENETA_EMBODIED_MCP_URL/session/readiness/envs" >/dev/null; then
    exec "$SWIFT_VENV/bin/python" scripts/run_single_model_curriculum.py \
      --config "$OPENETA_RL_CONFIG" --run-root "$RUN_ROOT" "$@"
  fi
  sleep 1
done
echo "MCP startup timed out; inspect $RUN_ROOT/service/mcp.log" >&2
exit 1
