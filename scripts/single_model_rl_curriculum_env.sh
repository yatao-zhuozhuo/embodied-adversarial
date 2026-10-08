#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/single_model_rl_32tasks_env.sh"
export OPENETA_RL_CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/single_model_rl_curriculum_mixed.json"
export MODEL_PATH=/inspire/qb-ilm/project/exploration-topic/ky26060/model/Qwen3.5-4B-embodied-mixed
export MAX_TURNS=48 OPENETA_RL_TURN_TOKENS=1536 OPENETA_RL_EPISODE_TOKENS=73728 VLLM_MAX_MODEL_LEN=32768
export DATASET_SHUFFLE=false OPENETA_CURRICULUM_ZERO_LIMIT=4
unset RESUME_FROM_CHECKPOINT TRAIN_ADAPTER_PATH
