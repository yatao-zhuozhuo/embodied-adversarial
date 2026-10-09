#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/single_model_rl_32tasks_env.sh"
export OPENETA_RL_CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/single_model_rl_curriculum_mixed.json"
export MODEL_PATH=/inspire/qb-ilm/project/exploration-topic/ky26060/model/Qwen3.5-4B-embodied-mixed
export MAX_TURNS=48 OPENETA_RL_TURN_TOKENS=1536 OPENETA_RL_EPISODE_TOKENS=73728 VLLM_MAX_MODEL_LEN=32768
export DATASET_SHUFFLE=false OPENETA_CURRICULUM_ZERO_LIMIT=4
unset RESUME_FROM_CHECKPOINT TRAIN_ADAPTER_PATH

# One MuJoCo graphics process per GPU; it can host several independent envs.
export OPENETA_WORKER_POOL_MAX_METAWORLD=8 OPENETA_WORKER_POOL_MAX_LIBERO=8 OPENETA_WORKER_POOL_MAX_ROBOCASA=8

# Keep the same timeline across all resumes of this experiment.
export OPENETA_TENSORBOARD_SERIES_ROOT="${OPENETA_TENSORBOARD_SERIES_ROOT:-/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_curriculum_mixed_fast3_20261008}"
