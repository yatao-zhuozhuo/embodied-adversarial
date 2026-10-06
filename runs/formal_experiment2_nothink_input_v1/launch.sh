#!/usr/bin/env bash
set -euo pipefail
cd /inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA
export RUN_NAME=formal_experiment2_nothink_input_v1
export RUN_ROOT="$PWD/runs/formal_experiment2_nothink_input_v1"
export MODEL_PATH=/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B
export TRAIN_GPU=0,1,2,3,4,5,6,7
export ROUNDS=5 TASKS_PER_ROUND=4 TRAIN_EPOCHS=3 TRAIN_STEPS=-1
export OPENETA_ENABLE_THINKING=false OPENETA_THINKING_MAX_TOKENS=4096
export MAX_TURNS=32 OPENETA_MAX_CONTROL_STEPS=96
export NUM_GENERATIONS=8 GENERATION_BATCH_SIZE=8
export PER_DEVICE_TRAIN_BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=8 STEPS_PER_GENERATION=1
export TEMPERATURE=1.0 TOP_P=0.9 LR=5e-6 LORA_RANK=8 LORA_ALPHA=16
export MAX_LENGTH=28672 MAX_COMPLETION_LENGTH=21504 VLLM_MAX_MODEL_LEN=28672
export OPENETA_BOB_EVALUATIONS=2
export BOOTSTRAP_ALICE_RUN="$PWD/runs/embodied_selfplay_grpo16"
export OPENETA_INITIALIZE_FROM_BASE=true
export OPENETA_EMBODIED_PROTOCOL=two_tool_xml_v1
export DRY_RUN="${DRY_RUN:-false}"
export OPENETA_INPUT_PROTOCOL=openeta.two_tool_input.v1
export OPENETA_CAMERA_RESOLUTION=256 MAX_PIXELS=65536
export OPENETA_COMPLETION_SCOPE=per_round
unset RESUME_FROM_CHECKPOINT
exec bash scripts/run_embodied_swift_staged_formal5.sh
