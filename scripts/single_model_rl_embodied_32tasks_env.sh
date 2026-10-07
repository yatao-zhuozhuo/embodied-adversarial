#!/usr/bin/env bash
# Fresh RL from the user's embodied model, with the existing 32-task settings.
source "$(dirname "${BASH_SOURCE[0]}")/single_model_rl_32tasks_env.sh"
export OPENETA_RL_CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/single_model_rl_embodied_32tasks.json"
export MODEL_PATH=/inspire/qb-ilm/project/exploration-topic/ky26060/model/Qwen3.5-4B-embodied
unset RESUME_FROM_CHECKPOINT TRAIN_ADAPTER_PATH
