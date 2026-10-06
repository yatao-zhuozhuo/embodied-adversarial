#!/usr/bin/env bash
# Source on embodied-selfplay-copy before submit_single_model_rl.py.
export OPENETA_RL_CONFIG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/configs/single_model_rl_32tasks.json"
export NUM_GENERATIONS=32 PER_DEVICE_TRAIN_BATCH_SIZE=4 GRADIENT_ACCUMULATION_STEPS=1
export TRAIN_EPOCHS=5 REPORT_TO=tensorboard TRAIN_GPU=0,1,2,3,4,5,6,7
export OPENETA_WORKER_POOL_MAX=32
