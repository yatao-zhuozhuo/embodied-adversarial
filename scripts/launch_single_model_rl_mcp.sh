#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/single_model_rl_env.sh"
: "${RUN_ROOT:?RUN_ROOT required}"
cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT/logs"
export OPENETA_WORKER_LOG_DIR="$RUN_ROOT/logs/workers"
if [[ ! -f "$RUN_ROOT/.host_token" ]]; then
  (umask 077; "${SWIFT_VENV:-/opt/openeta-swift}/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))' > "$RUN_ROOT/.host_token")
fi
export OPENETA_HOST_STATE_TOKEN="$(cat "$RUN_ROOT/.host_token")"
exec "${OPENETA_MCP_PYTHON:-$REPO_ROOT/.venv/bin/python}" -m sim.mcp_server --host 127.0.0.1 --port "${OPENETA_RL_MCP_PORT:-19765}"
