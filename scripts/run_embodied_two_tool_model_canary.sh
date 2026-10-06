#!/usr/bin/env bash
# One real Qwen rollout for Alice and Bob via the two-tool simulator MCP path.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/staged_two_tool_model_canary_remote}"
SWIFT_PYTHON="${SWIFT_VENV:-/opt/openeta-swift}/bin/python"
MCP_PORT="${OPENETA_MCP_PORT:-18765}"
ALICE_PORT="${OPENETA_MODEL_ALICE_PORT:-18822}"
BOB_PORT="${OPENETA_MODEL_BOB_PORT:-18821}"
mkdir -p "${RUN_ROOT}/logs"
export OPENETA_EMBODIED_PROTOCOL=two_tool_xml_v1
export OPENETA_EMBODIED_MCP_URL="http://127.0.0.1:${MCP_PORT}"
export OPENETA_HOST_STATE_TOKEN="$("${SWIFT_PYTHON}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
export OPENETA_MANISKILL_PYTHON="${OPENETA_MANISKILL_PYTHON:-${REPO_ROOT}/sim/venvs/maniskill312/bin/python}"
export OPENETA_WORKER_GPUS=0 OPENETA_WORKER_POOL_MAX=1
export OPENETA_STAGED_COLOCATE=true
export OPENETA_ENABLE_THINKING="${OPENETA_ENABLE_THINKING:-true}"
export OPENETA_THINKING_MAX_TOKENS="${OPENETA_THINKING_MAX_TOKENS:-4096}"
export OPENETA_INITIAL_POLICY_ORIGIN="${OPENETA_INITIAL_POLICY_ORIGIN:-base_model_untrained}"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

for port in "${MCP_PORT}" "${ALICE_PORT}" "${BOB_PORT}"; do
  if curl --noproxy '*' --connect-timeout 1 -fsS "http://127.0.0.1:${port}/health/" >/dev/null 2>&1 ||
     curl --noproxy '*' --connect-timeout 1 -fsS "http://127.0.0.1:${port}/session/readiness/envs" >/dev/null 2>&1; then
    echo "port ${port} is already in use" >&2
    exit 2
  fi
done

uv run python -m sim.mcp_server --host 127.0.0.1 --port "${MCP_PORT}" \
  >"${RUN_ROOT}/logs/simulator-mcp.log" 2>&1 &
MCP_PID=$!
ALICE_PID="" BOB_PID=""
cleanup() {
  for pid in "${ALICE_PID}" "${BOB_PID}"; do
    if [[ -n "${pid}" ]]; then
      kill -- "-${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  done
  kill "${MCP_PID}" 2>/dev/null || true
  wait "${MCP_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_ready() {
  local url="$1" pid="$2" label="$3"
  for _ in $(seq 1 180); do
    if curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS "${url}" >/dev/null 2>&1; then
      echo "${label} ready"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then break; fi
    sleep 2
  done
  echo "${label} failed to become ready" >&2
  return 1
}
wait_ready "${OPENETA_EMBODIED_MCP_URL}/session/readiness/envs" "${MCP_PID}" MCP

setsid env ROLE=alice ROLLOUT_ADAPTER_PATH="${INITIAL_ALICE_ADAPTER:-}" ROLLOUT_GPU=1 ROLLOUT_PORT="${ALICE_PORT}" \
  MAX_TURNS=3 VLLM_MAX_MODEL_LEN=16384 VLLM_MAX_NUM_SEQS=1 \
  VLLM_GPU_MEMORY_UTILIZATION=0.30 VLLM_KV_CACHE_MEMORY_BYTES=2147483648 \
  MAX_PIXELS=4096 OPENETA_SWIFT_ARTIFACT_ROOT="${RUN_ROOT}/alice" \
  ./scripts/run_embodied_swift_grpo.sh rollout \
  >"${RUN_ROOT}/logs/alice-swift.log" 2>&1 &
ALICE_PID=$!
setsid env ROLE=bob ROLLOUT_ADAPTER_PATH="${INITIAL_BOB_ADAPTER:-}" ROLLOUT_GPU=2 ROLLOUT_PORT="${BOB_PORT}" \
  MAX_TURNS=3 VLLM_MAX_MODEL_LEN=16384 VLLM_MAX_NUM_SEQS=1 \
  VLLM_GPU_MEMORY_UTILIZATION=0.30 VLLM_KV_CACHE_MEMORY_BYTES=2147483648 \
  MAX_PIXELS=4096 OPENETA_SWIFT_ARTIFACT_ROOT="${RUN_ROOT}/bob" \
  ./scripts/run_embodied_swift_grpo.sh rollout \
  >"${RUN_ROOT}/logs/bob-swift.log" 2>&1 &
BOB_PID=$!
wait_ready "http://127.0.0.1:${ALICE_PORT}/health/" "${ALICE_PID}" Alice
wait_ready "http://127.0.0.1:${BOB_PORT}/health/" "${BOB_PID}" Bob

"${SWIFT_PYTHON}" scripts/validate_embodied_two_tool_model.py \
  --role alice --port "${ALICE_PORT}" \
  --dataset runs/staged_colocate_5round/round-0000/alice_tasks.jsonl \
  --artifact-root "${RUN_ROOT}/alice" --output "${RUN_ROOT}/alice_result.json"
"${SWIFT_PYTHON}" scripts/validate_embodied_two_tool_model.py \
  --role bob --port "${BOB_PORT}" \
  --dataset runs/staged_colocate_5round/datasets/bob_bootstrap_train.jsonl \
  --artifact-root "${RUN_ROOT}/bob" --output "${RUN_ROOT}/bob_result.json"
