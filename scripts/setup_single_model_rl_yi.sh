#!/usr/bin/env bash
# Run on the network-enabled host sharing this checkout, before GPU launch.
# Existing baseline simulator environments and the shared Swift environment are reused.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/single_model_rl_yi_env.sh"
cd "$REPO_ROOT"
UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${PYTHON_BIN:-$SWIFT_VENV/bin/python}"
mkdir -p "$REPO_ROOT/.runtime/yi"
# Persist the base interpreter for GPU workspaces whose home directory differs.
if [[ ! -x "$REPO_ROOT/.runtime/yi/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12" ]]; then
  mkdir -p "$REPO_ROOT/.runtime/yi/python"
  cp -a "$(readlink -f "$PYTHON_BIN" | xargs dirname | xargs dirname)" \
    "$REPO_ROOT/.runtime/yi/python/cpython-3.12.11-linux-x86_64-gnu"
fi
"$UV_BIN" pip install --python "$PYTHON_BIN" --target "$OPENETA_TILELANG_OVERLAY" \
  --no-deps 'tilelang==0.1.14' 'apache-tvm-ffi==0.1.12' 'ml-dtypes==0.6.0' 'z3-solver==4.15.4.0'
"$UV_BIN" pip install --python "$PYTHON_BIN" --target "$OPENETA_MANISKILL_EXTRA_PATH" \
  --no-deps 'svgpathtools==1.8.0' 'svgwrite==1.4.3' 'pytest==8.4.2' 'pluggy==1.6.0' 'iniconfig==2.3.0' 'liger-kernel==0.8.4'
"$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path
Path(os.environ['VK_ICD_FILENAMES']).write_text(json.dumps({
    'file_format_version': '1.0.0',
    'ICD': {'library_path': '/usr/local/nvidia/lib64/libGLX_nvidia.so.0', 'api_version': '1.3.194'},
}, indent=2) + '\n')
PY
hf download YangYongzhuo/Qwen3.5-4B-embodied-mixed \
  --revision 8fd40ae02ce8fae67c90b85dc0a8d3c24069a16d --local-dir "$MODEL_PATH"
hf cache verify YangYongzhuo/Qwen3.5-4B-embodied-mixed \
  --revision 8fd40ae02ce8fae67c90b85dc0a8d3c24069a16d --local-dir "$MODEL_PATH"
hf download haosulab/ManiSkill2 data/mani_skill2_ycb.zip --repo-type dataset \
  --local-dir "$REPO_ROOT/.runtime/yi/maniskill-downloads"
"$PYTHON_BIN" - <<'PY'
import zipfile
from pathlib import Path
root = Path('.runtime/yi')
with zipfile.ZipFile(root / 'maniskill-downloads/data/mani_skill2_ycb.zip') as archive:
    archive.extractall(root / 'maniskill-assets')
PY
