#!/usr/bin/env bash
# GPU Job entry: check the two failure-prone native backends before resuming.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
PYTHON="${SWIFT_VENV:-/opt/openeta-swift}/bin/python"
OPENETA_TILELANG_OVERLAY="${OPENETA_TILELANG_OVERLAY:-/inspire/qb-ilm/project/exploration-topic/ky26060/openeta-runtime/tilelang-0.1.14-py312}"
test -x "${PYTHON}"
test -d "${OPENETA_TILELANG_OVERLAY}"
export PYTHONPATH="${REPO_ROOT}:${OPENETA_TILELANG_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Checking FLA backward with the TileLang overlay"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/probe_fla_tilelang.py
echo "Checking isolated ManiSkill render and snapshot on GPU 0"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/probe_maniskill_isolated.py \
  --render-gpu 0 --render-backend sapien_cuda:0

exec bash scripts/run_embodied_swift_staged_formal5.sh
