#!/usr/bin/env bash
set -euo pipefail
: "${RUN_ROOT:?}" "${NODE_RANK:?}" "${NNODES:?}" "${ORIGINAL_ACCEPTANCE:?}"
mkdir -p "$RUN_ROOT/nodes/$NODE_RANK" "$RUN_ROOT/coordination"
exec > >(tee -a "$RUN_ROOT/nodes/$NODE_RANK/job.log") 2>&1
service_pid=''
artifact_owner_pid=''
finish() {
  status=$?
  if [[ "$status" != 0 ]]; then
    printf '{"node_rank":%s,"error":"job shell exited %s; inspect node job.log"}\n' "$NODE_RANK" "$status" > "$RUN_ROOT/coordination/failed-$NODE_RANK.json"
  fi
  if [[ -n "$service_pid" ]]; then kill -- -"$service_pid" 2>/dev/null || true; fi
  if [[ -n "$artifact_owner_pid" ]]; then
    kill "$artifact_owner_pid" 2>/dev/null || true
    "$OPENETA_MCP_PYTHON" scripts/single_model_rl_artifact_owner.py --run-root "$RUN_ROOT" || true
  fi
}
trap finish EXIT
source scripts/single_model_rl_yi_env.sh
cd "$REPO_ROOT"
bash scripts/bootstrap_single_model_rl_yi_gpu.sh
if "$OPENETA_MCP_PYTHON" -c 'import json,os,sys; from pathlib import Path; sys.exit(not json.loads((Path(os.environ["RUN_ROOT"])/"curriculum_plan.json").read_text()).get("startup_maniskill_gpu_probe",False))'; then
  "$OPENETA_MCP_PYTHON" scripts/probe_single_model_rl_maniskill_gpus.py \
    --output "$RUN_ROOT/nodes/$NODE_RANK/maniskill_gpu_probe"
fi
if [[ "$NODE_RANK" == 0 ]]; then
  "$OPENETA_MCP_PYTHON" scripts/single_model_rl_artifact_owner.py --run-root "$RUN_ROOT" --watch &
  artifact_owner_pid=$!
fi
export OPENETA_TENSORBOARD_SERIES_ROOT="$RUN_ROOT"
export OPENETA_HOST_TOKEN_FILE="$RUN_ROOT/nodes/$NODE_RANK/service/.host_token"
export OPENETA_EMBODIED_MCP_URL="http://127.0.0.1:${OPENETA_RL_MCP_PORT:-19785}"
export OPENETA_RL_MCP_PORT="${OPENETA_RL_MCP_PORT:-19785}"
mkdir -p "$RUN_ROOT/nodes/$NODE_RANK/service"
RUN_ROOT="$RUN_ROOT/nodes/$NODE_RANK/service" setsid bash scripts/launch_single_model_rl_mcp.sh > "$RUN_ROOT/nodes/$NODE_RANK/service/mcp.log" 2>&1 &
service_pid=$!
for attempt in $(seq 1 120); do
  if curl --noproxy '*' -fsS "$OPENETA_EMBODIED_MCP_URL/session/readiness/envs" >/dev/null; then break; fi
  kill -0 "$service_pid"
  sleep 1
done
export OPENETA_HOST_STATE_TOKEN="$(cat "$OPENETA_HOST_TOKEN_FILE")"
"$SWIFT_VENV/bin/python" scripts/run_single_model_rl_yi_multinode.py --run-root "$RUN_ROOT" --original-acceptance "$ORIGINAL_ACCEPTANCE"
