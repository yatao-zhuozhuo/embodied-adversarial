#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export WANDB_MODE=disabled
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
export OPENETA_INPUT_PROTOCOL=openeta.two_tool_input.v1
export OPENETA_EMBODIED_PROTOCOL=two_tool_xml_v1
export OPENETA_STAGED_COLOCATE=false
export OPENETA_ENABLE_THINKING=false
export OPENETA_COMPLETION_SCOPE=per_round
export OPENETA_RL_TURN_TOKENS="${OPENETA_RL_TURN_TOKENS:-768}"
export MAX_TURNS="${MAX_TURNS:-60}"
export OPENETA_RL_EPISODE_TOKENS="${OPENETA_RL_EPISODE_TOKENS:-$((MAX_TURNS * OPENETA_RL_TURN_TOKENS))}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
export OPENETA_TILELANG_OVERLAY="${OPENETA_TILELANG_OVERLAY:-/inspire/qb-ilm/project/exploration-topic/ky26060/openeta-runtime/tilelang-0.1.14-py312}"
export PYTHONPATH="${REPO_ROOT}:${OPENETA_TILELANG_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
export SWIFT_SINGLE_DEVICE_MODE=1
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export NPROC_PER_NODE="${#GPUS[@]}"
export MASTER_PORT="${MASTER_PORT:-29617}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
SWIFT_VENV="${SWIFT_VENV:-/opt/openeta-swift}"
: "${DATASET_PATH:?DATASET_PATH is required}"
: "${OUTPUT_DIR:?OUTPUT_DIR is required}"
: "${OPENETA_EMBODIED_MCP_URL:?MCP URL is required}"
export OPENETA_SWIFT_ARTIFACT_ROOT="${OPENETA_SWIFT_ARTIFACT_ROOT:-${OUTPUT_DIR}/rollouts}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GENERATION_BATCH_SIZE=$((${NNODES:-1} * NPROC_PER_NODE * PER_DEVICE_TRAIN_BATCH_SIZE))
if (( GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0 || NUM_GENERATIONS < 2 )); then
  echo 'generation batch must contain complete groups with G >= 2' >&2; exit 2
fi
INIT_ARGS=()
if [[ -n "${MODEL_TYPE:-}" ]]; then
  INIT_ARGS+=(--model_type "$MODEL_TYPE")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  INIT_ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT")
elif [[ -n "${TRAIN_ADAPTER_PATH:-}" ]]; then
  INIT_ARGS+=(--adapters "$TRAIN_ADAPTER_PATH")
fi
export NUM_GENERATIONS PER_DEVICE_TRAIN_BATCH_SIZE GENERATION_BATCH_SIZE
if [[ "${NODE_RANK:-0}" == 0 && -n "${OPENETA_TENSORBOARD_SERIES_ROOT:-}" ]]; then
  "$SWIFT_VENV/bin/python" scripts/single_model_tensorboard_series.py --source-root "$OPENETA_TENSORBOARD_SERIES_ROOT" --start
fi
if [[ "${NODE_RANK:-0}" == 0 ]]; then
  "$SWIFT_VENV/bin/python" scripts/record_single_model_rl_run.py --output "$OUTPUT_DIR/run_manifest.json" --dataset "$DATASET_PATH"
fi
"$SWIFT_VENV/bin/python" -c 'import tilelang'
exec "$SWIFT_VENV/bin/python" "$SWIFT_VENV/bin/swift" rlhf \
 --rlhf_type grpo --check_model false --model "${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}" \
 "${INIT_ARGS[@]}" --external_plugins "$REPO_ROOT/plugins/single_model_swift_grpo.py" \
 --reward_funcs single_model_success --dataset "$DATASET_PATH" --load_from_cache_file false \
 --dataset_num_proc 1 --split_dataset_ratio 0 --use_vllm true --vllm_mode colocate \
 --multi_turn_scheduler single_model_native_scheduler --max_turns "$MAX_TURNS" \
 --completion_length_limit_scope per_round --vllm_tensor_parallel_size 1 \
 --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" --vllm_max_num_seqs "$PER_DEVICE_TRAIN_BATCH_SIZE" \
 --vllm_gpu_memory_utilization 0.50 --vllm_mm_processor_cache_gb 0 \
 --vllm_engine_kwargs '{"kv_cache_memory_bytes":8589934592}' \
 --vllm_enforce_eager true --vllm_enable_prefix_caching true --vllm_enable_lora true \
 --vllm_limit_mm_per_prompt '{"image":5}' --sleep_level 2 \
 --max_length "$VLLM_MAX_MODEL_LEN" --max_completion_length "$OPENETA_RL_TURN_TOKENS" \
 --max_pixels 65536 --num_train_epochs "${TRAIN_EPOCHS:-1}" --max_steps "${TRAIN_STEPS:--1}" \
 --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
 --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
 --num_generations "$NUM_GENERATIONS" --generation_batch_size "$GENERATION_BATCH_SIZE" \
 --dataset_shuffle "${DATASET_SHUFFLE:-true}" --train_dataloader_shuffle "${DATASET_SHUFFLE:-true}" \
 --ignore_data_skip "${IGNORE_DATA_SKIP:-false}" \
 --steps_per_generation 1 --learning_rate "${LR:-5e-6}" --lr_scheduler_type constant \
 --tuner_type lora --lora_rank 8 --lora_alpha 16 --target_modules all-linear \
 --gradient_checkpointing true --use_liger_kernel true --bf16 true --beta "${BETA:-0.01}" \
 --epsilon 0.2 --loss_type grpo --loss_scale last_round --importance_sampling_level token \
 --temperature "${TEMPERATURE:-1.0}" --top_p 0.9 --logging_steps 1 --save_steps 1 \
 --save_total_limit 2 --output_dir "$OUTPUT_DIR" --log_completions true --report_to "${REPORT_TO:-tensorboard}"
