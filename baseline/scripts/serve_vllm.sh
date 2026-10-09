#!/usr/bin/env bash
# Run in the same SWIFT_VENV as run_embodied_swift_staged_formal5.sh.
set -euo pipefail
ulimit -c 0 2>/dev/null || true
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SWIFT_VENV="${SWIFT_VENV:-${REPO_ROOT}/baseline/.runtime/swift}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/baseline/.runtime/models}"
case "${1:-}" in
  qwen38_27b) MODEL_NAME=Qwen3.8-27B; DEFAULT_PORT=8101 ;;
  qwen35_9b) MODEL_NAME=Qwen3.5-9B; DEFAULT_PORT=8102 ;;
  qwen35_4b) MODEL_NAME=Qwen3.5-4B; DEFAULT_PORT=8103 ;;
  qwen35_4b_embodied) MODEL_NAME=Qwen3.5-4B-embodied; DEFAULT_PORT=8104 ;;
  *) echo "Usage: CUDA_VISIBLE_DEVICES=0 bash $0 {qwen38_27b|qwen35_9b|qwen35_4b|qwen35_4b_embodied}" >&2; exit 2 ;;
esac
PYTHON="${SWIFT_VENV}/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing formal5 Python: $PYTHON (check the machine-local symlink target)." >&2
  exit 2
fi
if [[ ! -s "${MODEL_ROOT}/${MODEL_NAME}/config.json" ]]; then
  echo "Download Qwen/${MODEL_NAME} with ModelScope to ${MODEL_ROOT}/${MODEL_NAME} first." >&2
  exit 2
fi
"$PYTHON" - "${MODEL_ROOT}/${MODEL_NAME}" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
index = json.loads((root / "model.safetensors.index.json").read_text())
missing = [name for name in set(index["weight_map"].values())
           if not (root / name).is_file() or (root / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"Incomplete model download: {missing}")
PY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MAX_JOBS="${MAX_JOBS:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
if [[ -d /usr/local/nvidia/lib64 ]]; then
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
BASELINE_API_OVERLAY="${BASELINE_API_OVERLAY:-${REPO_ROOT}/baseline/.runtime/api-compat}"
if [[ ! -f "${BASELINE_API_OVERLAY}/fastapi/__init__.py" ]]; then
  echo "Install the API compatibility overlay first (see baseline/README.md)." >&2
  exit 2
fi
export PYTHONPATH="${BASELINE_API_OVERLAY}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# TileLang is a training backward-kernel overlay, not required by this inference server.
cd "$REPO_ROOT"
exec "$PYTHON" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_ROOT}/${MODEL_NAME}" \
  --served-model-name "$MODEL_NAME" \
  --host "${BASELINE_HOST:-127.0.0.1}" --port "${BASELINE_PORT:-$DEFAULT_PORT}" \
  --dtype bfloat16 --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --max-model-len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${VLLM_MAX_NUM_SEQS:-1}" \
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.65}" \
  --kv-cache-memory-bytes "${VLLM_KV_CACHE_MEMORY_BYTES:-4294967296}" \
  --mm-processor-cache-gb 0 --limit-mm-per-prompt '{"image":5,"video":0}' \
  --mm-processor-kwargs '{"size":{"shortest_edge":4096,"longest_edge":262144}}' \
  --gdn-prefill-backend triton \
  --reasoning-parser qwen3 --generation-config vllm --enforce-eager
