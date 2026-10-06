#!/usr/bin/env bash
# Source on the evaluation host. All defaults belong to this checkout.
BASELINE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export BASELINE_REPO
export LIBERO_DIR="${LIBERO_DIR:-$BASELINE_REPO/baseline/.runtime/sources/LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$BASELINE_REPO/baseline/.runtime/libero-config}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OPENETA_LIBERO_CONTROLLER_PROFILE=osc_pose
export MS_ASSET_DIR="${MS_ASSET_DIR:-$BASELINE_REPO/baseline/.runtime/maniskill-assets}"
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
# This H200 deployment passed the native NVIDIA camera probe. Select mesa
# explicitly when deploying to a host that requires software rendering.
export BASELINE_VULKAN_BACKEND="${BASELINE_VULKAN_BACKEND:-nvidia}"
export LD_LIBRARY_PATH="$BASELINE_REPO/baseline/.runtime/mesa/rootfs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_PRELOAD="$BASELINE_REPO/baseline/.runtime/mesa/rootfs/usr/lib/x86_64-linux-gnu/libvulkan.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
if [[ "$BASELINE_VULKAN_BACKEND" == mesa ]]; then
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$BASELINE_REPO/baseline/.runtime/mesa/lvp_icd.json}"
  export OPENETA_MANISKILL_RENDER_BACKEND="${OPENETA_MANISKILL_RENDER_BACKEND:-sapien_cpu}"
else
  export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$BASELINE_REPO/baseline/.runtime/nvidia/vulkan_icd.json}"
  export OPENETA_MANISKILL_RENDER_BACKEND="${OPENETA_MANISKILL_RENDER_BACKEND:-sapien_cuda}"
fi
export LP_NUM_THREADS="${LP_NUM_THREADS:-4}"
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/usr/local/nvidia/bin:$PATH"
export OPENETA_WORKER_GPUS="${OPENETA_WORKER_GPUS:-0}"
export OPENETA_WORKER_POOL_MAX="${OPENETA_WORKER_POOL_MAX:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=20261002
