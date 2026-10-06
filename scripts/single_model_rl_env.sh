#!/usr/bin/env bash
# Source from a launch script on embodied-selfplay-copy.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export OPENETA_MANISKILL_EXTRA_PATH="/inspire/qb-ilm/project/exploration-topic/ky26060/openeta-runtime/rl-maniskill-extra"
export OPENETA_MANISKILL_PYTHON="$REPO_ROOT/sim/venvs/maniskill312/bin/python"
export OPENETA_LIBERO_PYTHON="$REPO_ROOT/sim/venvs/libero/bin/python"
export OPENETA_METAWORLD_PYTHON="$REPO_ROOT/sim/venvs/metaworld/bin/python"
export OPENETA_ROBOCASA_PYTHON="$REPO_ROOT/sim/venvs/robocasa/bin/python"
export LIBERO_DIR="$REPO_ROOT/sim/venvs/_sources/LIBERO"
export LIBERO_CONFIG_PATH="$REPO_ROOT/SFT_data/configs/libero"
export OPENETA_WORKER_GPUS="${OPENETA_WORKER_GPUS:-0,1,2,3,4,5,6,7}"
export OPENETA_WORKER_POOL_MAX="${OPENETA_WORKER_POOL_MAX:-8}"
export MUJOCO_GL=egl
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
if [[ -s /etc/vulkan/icd.d/nvidia_icd.json ]]; then export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json; fi
export OPENETA_EMBODIED_MCP_URL="http://127.0.0.1:${OPENETA_RL_MCP_PORT:-19765}"
