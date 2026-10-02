#!/usr/bin/env bash
# Fill one teacher's 500-success ManiSkill quota using the local GPU simulator.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
TEACHER="${1:-}"
if [[ "${TEACHER}" != "glm" && "${TEACHER}" != "qwen" ]]; then
  echo "usage: $0 glm|qwen" >&2
  exit 2
fi
if [[ -z "${INF_API_KEY:-}" ]]; then
  echo "INF_API_KEY must be exported in the environment" >&2
  exit 2
fi

export OPENETA_LLM_PROVIDER=openai-compatible
export OPENETA_LLM_API_KEY="${INF_API_KEY}"
export OPENETA_LLM_TIMEOUT_S="${OPENETA_LLM_TIMEOUT_S:-180}"
export OPENETA_LLM_MAX_ATTEMPTS="${OPENETA_LLM_MAX_ATTEMPTS:-3}"
export OPENETA_LLM_RETRY_BACKOFF_S="${OPENETA_LLM_RETRY_BACKOFF_S:-1.0}"
export OPENETA_LLM_CONTEXT_WINDOW_TOKENS="${OPENETA_LLM_CONTEXT_WINDOW_TOKENS:-1000000}"
export OPENETA_LLM_MAX_TOKENS="${OPENETA_LLM_MAX_TOKENS:-4096}"
export OPENETA_LLM_ENABLE_THINKING="${OPENETA_LLM_ENABLE_THINKING:-false}"
export OPENETA_WEB_SEARCH_ENABLED=false
export OPENETA_WEB_FETCH_ENABLED=false
export OPENETA_VDM_CAMERA_ROLE=base_camera
export OPENETA_VISUAL_VDM_ENABLED=false
export OPENETA_MANISKILL_PYTHON="${OPENETA_MANISKILL_PYTHON:-${REPO_ROOT}/sim/venvs/maniskill312/bin/python}"
export OPENETA_WORKER_GPUS="${OPENETA_WORKER_GPUS:-0}"
export OPENETA_WORKER_POOL_MAX="${OPENETA_WORKER_POOL_MAX:-1}"

if [[ "${TEACHER}" == "glm" ]]; then
  export OPENETA_LLM_MODEL=GLM-5.3-w8a8c8
  export OPENETA_LLM_API_BASE=https://hhapbb8eb5cocm8kmma8okkogmea5pde.openapi-sj.sii.edu.cn/v1
  export OPENETA_LLM_ENABLE_VISION=false
  API_HOST=hhapbb8eb5cocm8kmma8okkogmea5pde.openapi-sj.sii.edu.cn
else
  export OPENETA_LLM_MODEL=Qwen3.8-27B
  export OPENETA_LLM_API_BASE=https://cqhbod8bjjjbcoakk8pmeebgkaq9akcq.openapi-sj.sii.edu.cn/v1
  export OPENETA_LLM_ENABLE_VISION=true
  API_HOST=cqhbod8bjjjbcoakk8pmeebgkaq9akcq.openapi-sj.sii.edu.cn
fi
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${API_HOST}"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,${API_HOST}"

GPU_UTIL="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 | head -n 1 | tr -d ' ')"
if [[ "${GPU_UTIL}" =~ ^[0-9]+$ ]] && (( GPU_UTIL > 25 )) && [[ "${OPENETA_ALLOW_BUSY_GPU:-false}" != "true" ]]; then
  echo "GPU 0 is currently ${GPU_UTIL}% utilized; release the existing task or set OPENETA_ALLOW_BUSY_GPU=true" >&2
  exit 3
fi

MCP_PORT="${OPENETA_MCP_PORT:-8765}"
MCP_LOG="SFT_data/reports/${TEACHER}.collection.mcp.log"
mkdir -p SFT_data/reports
uv run python -m sim.mcp_server --host 127.0.0.1 --port "${MCP_PORT}" >"${MCP_LOG}" 2>&1 &
MCP_PID=$!
cleanup() {
  kill "${MCP_PID}" 2>/dev/null || true
  wait "${MCP_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

READY=false
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${MCP_PORT}/session/readiness/envs" >/dev/null 2>&1; then
    READY=true
    break
  fi
  sleep 1
done
if [[ "${READY}" != "true" ]]; then
  echo "simulator MCP failed to become ready; see ${MCP_LOG}" >&2
  exit 4
fi

uv run python scripts/collect_maniskill_sft.py \
  --teacher "${TEACHER}" \
  --sim-url "http://127.0.0.1:${MCP_PORT}/sse" \
  --batch-size "${OPENETA_COLLECTION_BATCH_SIZE:-5}" \
  --concurrency "${OPENETA_COLLECTION_CONCURRENCY:-1}" \
  --provider-concurrency "${OPENETA_PROVIDER_CONCURRENCY:-1}" \
  --max-attempts "${OPENETA_COLLECTION_MAX_ATTEMPTS:-2500}"
