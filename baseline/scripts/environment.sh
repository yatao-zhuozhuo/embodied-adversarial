#!/usr/bin/env bash
# Source on the evaluation GPU host. Does not alter the training environment.
BASELINE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LIBERO_DIR="$BASELINE_REPO/baseline/.runtime/sources/LIBERO"
export LIBERO_CONFIG_PATH="$BASELINE_REPO/baseline/.runtime/libero-config"
export MUJOCO_GL=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="$BASELINE_REPO/baseline/.runtime/nvidia/egl_vendor.json"
export OPENETA_LIBERO_CONTROLLER_PROFILE=osc_pose
export MS_ASSET_DIR="$BASELINE_REPO/baseline/.runtime/maniskill-assets"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
# The host NVIDIA Vulkan path loses the device on the first camera capture.
# Freeze ManiSkill rendering to Mesa lavapipe; MuJoCo continues to use EGL.
export VK_ICD_FILENAMES="$BASELINE_REPO/baseline/.runtime/mesa/lvp_icd.json"
export OPENETA_MANISKILL_RENDER_BACKEND=sapien_cpu
export LP_NUM_THREADS=4
export LD_LIBRARY_PATH="$BASELINE_REPO/baseline/.runtime/mesa:$BASELINE_REPO/baseline/.runtime/nvidia/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libvulkan.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
export PATH="/usr/local/nvidia/bin:$PATH"
export OPENETA_WORKER_GPUS="${OPENETA_WORKER_GPUS:-3}"
export OPENETA_WORKER_POOL_MAX=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20261002
