#!/usr/bin/env bash
# Launch the isolated simulator MCP and the resumable two-tool SFT collector.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${CODINGROUTER_API_KEY:-${DEEPSEEK_API_KEY:-${INF_API_KEY:-${OPENETA_LLM_API_KEY:-}}}}" ]]; then
  echo "Set the API key for the selected provider" >&2
  exit 2
fi

MANIFEST="${OPENETA_TWO_TOOL_MANIFEST:-${REPO_ROOT}/SFT_data/manifests/two_tool_80.v1.json}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "Manifest missing: ${MANIFEST}" >&2
  exit 2
fi

export OPENETA_MANISKILL_PYTHON="${OPENETA_MANISKILL_PYTHON:-${REPO_ROOT}/sim/venvs/maniskill312/bin/python}"
export OPENETA_LIBERO_PYTHON="${OPENETA_LIBERO_PYTHON:-${REPO_ROOT}/sim/venvs/libero/bin/python}"
export OPENETA_METAWORLD_PYTHON="${OPENETA_METAWORLD_PYTHON:-${REPO_ROOT}/sim/venvs/metaworld/bin/python}"
export OPENETA_ROBOCASA_PYTHON="${OPENETA_ROBOCASA_PYTHON:-${REPO_ROOT}/sim/venvs/robocasa/bin/python}"
export LIBERO_DIR="${LIBERO_DIR:-${REPO_ROOT}/sim/venvs/_sources/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/SFT_data/configs/libero}"
export OPENETA_WORKER_GPUS="${OPENETA_WORKER_GPUS:-0}"
export OPENETA_WORKER_POOL_MAX="${OPENETA_WORKER_POOL_MAX:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
if [[ -f /etc/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
fi

API_HOST=cqhbod8bjjjbcoakk8pmeebgkaq9akcq.openapi-sj.sii.edu.cn,api.deepseek.com,api.codingrouter.ai
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost,${API_HOST}"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,${API_HOST}"

PORT="${OPENETA_TWO_TOOL_MCP_PORT:-8766}"
LOG="${OPENETA_TWO_TOOL_MCP_LOG_DIR:-${REPO_ROOT}/SFT_data/reports/qwen_xml_v1}/mcp.${PORT}.log"
mkdir -p "$(dirname "${LOG}")"
uv run python -m sim.mcp_server --host 127.0.0.1 --port "${PORT}" >"${LOG}" 2>&1 &
MCP_PID=$!
cleanup() {
  kill "${MCP_PID}" 2>/dev/null || true
  wait "${MCP_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

READY=false
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:${PORT}/session/readiness/envs" >/dev/null 2>&1; then
    READY=true
    break
  fi
  sleep 1
done
if [[ "${READY}" != true ]]; then
  echo "MCP server did not become ready; see ${LOG}" >&2
  exit 4
fi

uv run python scripts/collect_two_tool_sft.py \
  --manifest "${MANIFEST}" \
  --sim-url "http://127.0.0.1:${PORT}/sse" \
  "$@"
