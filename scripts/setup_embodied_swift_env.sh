#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_VENV="${SWIFT_VENV:-/opt/openeta-swift}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.org/simple}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create the isolated runtime" >&2
  exit 2
fi

if [[ ! -x "${SWIFT_VENV}/bin/python" ]]; then
  uv venv --python "${PYTHON_BIN}" "${SWIFT_VENV}"
fi

UV_INDEX_URL="${PYPI_INDEX}" uv pip install \
  --python "${SWIFT_VENV}/bin/python" \
  -r "${REPO_ROOT}/configs/requirements-embodied-swift.txt"

"${SWIFT_VENV}/bin/python" - <<'PY'
import torch
import transformers
import vllm
import swift
import causal_conv1d
import fla

print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "vllm": vllm.__version__,
    "swift": swift.__version__,
    "flash_linear_attention": fla.__version__,
    "causal_conv1d": causal_conv1d.__version__,
    "cuda_available": torch.cuda.is_available(),
})
PY
