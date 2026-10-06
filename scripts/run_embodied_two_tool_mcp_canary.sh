#!/usr/bin/env bash
# Real simulator MCP + deterministic Swift scheduler canary for both roles.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/staged_two_tool_mcp_canary_remote}"
MCP_PORT="${OPENETA_MCP_PORT:-18765}"
SWIFT_PYTHON="${SWIFT_VENV:-/opt/openeta-swift}/bin/python"
export OPENETA_EMBODIED_MCP_URL="http://127.0.0.1:${MCP_PORT}"
export OPENETA_HOST_STATE_TOKEN="$("${SWIFT_PYTHON}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export OPENETA_MANISKILL_PYTHON="${OPENETA_MANISKILL_PYTHON:-${REPO_ROOT}/sim/venvs/maniskill312/bin/python}"
export OPENETA_WORKER_GPUS="${OPENETA_WORKER_GPUS:-0}"
export OPENETA_WORKER_POOL_MAX="${OPENETA_WORKER_POOL_MAX:-1}"
export OPENETA_STAGED_COLOCATE=true
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
mkdir -p "${RUN_ROOT}/logs"
if curl --noproxy '*' --connect-timeout 1 -fsS \
    "${OPENETA_EMBODIED_MCP_URL}/session/readiness/envs" >/dev/null 2>&1; then
  echo "MCP port ${MCP_PORT} is already in use" >&2
  exit 2
fi
uv run python -m sim.mcp_server --host 127.0.0.1 --port "${MCP_PORT}" \
  >"${RUN_ROOT}/logs/simulator-mcp.log" 2>&1 &
MCP_PID=$!
cleanup() {
  kill "${MCP_PID}" 2>/dev/null || true
  wait "${MCP_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
ready=false
for _ in $(seq 1 60); do
  if curl --noproxy '*' -fsS \
      "${OPENETA_EMBODIED_MCP_URL}/session/readiness/envs" >/dev/null 2>&1; then
    ready=true
    break
  fi
  if ! kill -0 "${MCP_PID}" 2>/dev/null; then break; fi
  sleep 1
done
if [[ "${ready}" != "true" ]]; then
  echo "MCP server failed; see ${RUN_ROOT}/logs/simulator-mcp.log" >&2
  exit 3
fi

"${SWIFT_PYTHON}" scripts/validate_embodied_two_tool_mcp.py \
  --output "${RUN_ROOT}/mcp_preflight.json"
if [[ "${OPENETA_CONCURRENCY_DIAG:-false}" == true ]]; then
  sleep "${OPENETA_DIAG_IDLE_S:-0}"
  PYTHONPATH="${REPO_ROOT}" "${SWIFT_PYTHON}" scripts/diagnose_embodied_mcp_concurrency.py "${RUN_ROOT}" \
    >"${RUN_ROOT}/concurrency.jsonl"
  cat "${RUN_ROOT}/concurrency.jsonl"
  exit 0
fi
PYTHONPATH="${REPO_ROOT}" "${SWIFT_PYTHON}" -c '
from agent.runtime.two_tool_xml import serialize_decision
print(serialize_decision("observe_env"))
print(serialize_decision("step_env", {"action":[0,0,0,0,0,0,1], "num_steps":2}))
print(serialize_decision("task_complete"))
' >"${RUN_ROOT}/commands.jsonl"
OPENETA_SWIFT_ARTIFACT_ROOT="${RUN_ROOT}/alice" \
"${SWIFT_PYTHON}" scripts/validate_embodied_swift_scheduler.py \
  --dataset runs/staged_colocate_5round/round-0000/alice_tasks.jsonl \
  --commands-file "${RUN_ROOT}/commands.jsonl" \
  >"${RUN_ROOT}/alice_result.json"
OPENETA_SWIFT_ARTIFACT_ROOT="${RUN_ROOT}/bob" \
"${SWIFT_PYTHON}" scripts/validate_embodied_swift_scheduler.py \
  --dataset runs/staged_colocate_5round/datasets/bob_bootstrap_train.jsonl \
  --commands-file "${RUN_ROOT}/commands.jsonl" \
  >"${RUN_ROOT}/bob_result.json"
"${SWIFT_PYTHON}" -c '
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
for role in ("alice","bob"):
    result=json.loads((root/f"{role}_result.json").read_text())
    assert result["assistant_turns"]==3
    assert result["trained_tokens"]==result["logprob_tokens"]
    assert result["termination_reason"]=="model_finish"
    artifact=json.loads((root/role/role/"scheduler-canary.json").read_text())
    assert artifact["protocol_version"]=="openeta.two_tool_xml.v1"
    assert len(artifact["restore_receipts"])>=1
print("Alice and Bob real-MCP scheduler canaries passed")
' "${RUN_ROOT}"
