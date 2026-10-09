#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/embodied_bob_grpo_smoke}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

exec python3 scripts/train_embodied_grpo.py \
  --model "${MODEL_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --role "${ROLE:-bob}" \
  --iterations "${ITERATIONS:-1}" \
  --group-size "${GROUP_SIZE:-2}" \
  --max-steps "${MAX_STEPS:-1}" \
  --update-epochs "${UPDATE_EPOCHS:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-40}" \
  --save-every "${SAVE_EVERY:-1}"
