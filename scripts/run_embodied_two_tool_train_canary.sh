#!/usr/bin/env bash
# Bounded eight-H200 GRPO integration probe. Run detached on the target notebook.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
RUN_NAME="${RUN_NAME:-staged_two_tool_xml_train_canary}"
RUN_ROOT="${RUN_ROOT:-/tmp/openeta-two-tool-train-canary}"
mkdir -p "${RUN_ROOT}"
printf '%s\n' "$$" >"${RUN_ROOT}/runner.pid"
export RUN_NAME RUN_ROOT
export ROUNDS=1 TASKS_PER_ROUND=1 TRAIN_STEPS=1 TRAIN_EPOCHS=1
export GRADIENT_ACCUMULATION_STEPS=1
export MAX_TURNS="${MAX_TURNS:-2}" OPENETA_MAX_CONTROL_STEPS="${OPENETA_MAX_CONTROL_STEPS:-4}"
export OPENETA_ENABLE_THINKING="${OPENETA_ENABLE_THINKING:-true}"
export OPENETA_THINKING_MAX_TOKENS="${OPENETA_THINKING_MAX_TOKENS:-4096}"
export MAX_COMPLETION_LENGTH=8192 MAX_LENGTH=12288 VLLM_MAX_MODEL_LEN=12288
export MAX_PIXELS="${MAX_PIXELS:-4096}"
export OPENETA_WORKER_GPUS="${OPENETA_WORKER_GPUS:-0}"
export OPENETA_WORKER_POOL_MAX="${OPENETA_WORKER_POOL_MAX:-1}"
if [[ "${OPENETA_DIAGNOSTIC_GPU_MONITOR:-false}" == true ]]; then
  (
    while true; do
      date -u '+%Y-%m-%dT%H:%M:%SZ'
      nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader
      sleep 2
    done
  ) >"${RUN_ROOT}/gpu_memory.log" 2>&1 &
  MONITOR_PID=$!
  trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT
fi
./scripts/run_embodied_swift_staged_formal5.sh
status=$?
printf '%s\n' "${status}" >"${RUN_ROOT}/runner.exit"
exit "${status}"
