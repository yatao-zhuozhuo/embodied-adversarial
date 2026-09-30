#!/usr/bin/env bash
set -euo pipefail

# Eight-rank colocate entry for staged Alice training and ordinary Bob GRPO.
# Alice uses one base model per GPU and routes adapter ID 1 (current Alice),
# then adapter ID 2 (the frozen Bob) through the same local vLLM engine.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_VENV="${SWIFT_VENV:-/opt/openeta-swift}"
PYTHON="${SWIFT_VENV}/bin/python"
SWIFT_BIN="${SWIFT_VENV}/bin/swift"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}"
ROLE="${ROLE:-alice}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MAX_TURNS="${MAX_TURNS:-32}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
DATASET_PATH="${DATASET_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/embodied_staged_colocate/${ROLE}_train}"
TRAIN_ADAPTER_PATH="${TRAIN_ADAPTER_PATH:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ ! -x "${PYTHON}" || ! -x "${SWIFT_BIN}" ]]; then
  echo "Swift runtime not found under ${SWIFT_VENV}" >&2
  exit 2
fi
if [[ -z "${DATASET_PATH}" || ! -f "${DATASET_PATH}" ]]; then
  echo "DATASET_PATH must name an existing JSONL dataset" >&2
  exit 2
fi
IFS=',' read -r -a OPENETA_VISIBLE_GPU_LIST <<<"${CUDA_VISIBLE_DEVICES}"
if [[ "${#OPENETA_VISIBLE_GPU_LIST[@]}" -ne 8 || "${NPROC_PER_NODE}" -ne 8 ]]; then
  echo "staged entry requires exactly 8 visible GPUs and NPROC_PER_NODE=8" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES NPROC_PER_NODE
export SWIFT_SINGLE_DEVICE_MODE=1
if [[ -n "${OPENETA_TILELANG_OVERLAY:-}" ]]; then
  if [[ ! -d "${OPENETA_TILELANG_OVERLAY}" ]]; then
    echo "TileLang overlay is missing: ${OPENETA_TILELANG_OVERLAY}" >&2
    exit 2
  fi
  export PYTHONPATH="${OPENETA_TILELANG_OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ "${OPENETA_REQUIRE_TILELANG:-false}" =~ ^(1|true|yes|on)$ ]]; then
  if ! "${PYTHON}" -c 'import tilelang' >/dev/null 2>&1; then
    echo "TileLang is required but cannot be imported from ${OPENETA_TILELANG_OVERLAY:-PYTHONPATH}" >&2
    exit 2
  fi
fi
export OPENETA_SWIFT_ARTIFACT_ROOT="${OPENETA_SWIFT_ARTIFACT_ROOT:-${OUTPUT_DIR}/rollouts}"
export OPENETA_STAGED_ARTIFACT_ROOT="${OPENETA_STAGED_ARTIFACT_ROOT:-$(dirname "${OUTPUT_DIR}")}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

INIT_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  INIT_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
elif [[ -n "${TRAIN_ADAPTER_PATH}" ]]; then
  INIT_ARGS+=(--adapters "${TRAIN_ADAPTER_PATH}")
fi

if [[ "${ROLE}" == "alice" ]]; then
  if [[ -z "${OPENETA_FROZEN_BOB_ADAPTER:-}" || ! -d "${OPENETA_FROZEN_BOB_ADAPTER}" ]]; then
    echo "Alice staged training requires OPENETA_FROZEN_BOB_ADAPTER" >&2
    exit 2
  fi
  export OPENETA_STAGED_COLOCATE=true
  export OPENETA_BOB_EVALUATIONS="${OPENETA_BOB_EVALUATIONS:-2}"
  export OPENETA_BOB_POLICY_VERSION="${OPENETA_BOB_POLICY_VERSION:-${OPENETA_FROZEN_BOB_ADAPTER}}"
  SCHEDULER="embodied_alice_scheduler"
  REWARD_FUNC="embodied_alice_reward"
  ENTRY=(
    "${PYTHON}" -m torch.distributed.run
    --nproc_per_node "${NPROC_PER_NODE}"
    "${REPO_ROOT}/scripts/run_embodied_staged_colocate.py"
  )
elif [[ "${ROLE}" == "bob" ]]; then
  export OPENETA_STAGED_COLOCATE=false
  SCHEDULER="embodied_bob_scheduler"
  REWARD_FUNC="embodied_bob_reward"
  ENTRY=("${SWIFT_BIN}" rlhf)
else
  echo "ROLE must be alice or bob" >&2
  exit 2
fi

exec "${ENTRY[@]}" \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  "${INIT_ARGS[@]}" \
  --external_plugins "${REPO_ROOT}/plugins/embodied_swift_grpo.py" \
  --reward_funcs "${REWARD_FUNC}" \
  --dataset "${DATASET_PATH}" \
  --load_from_cache_file false \
  --dataset_num_proc 1 \
  --use_vllm true \
  --vllm_mode colocate \
  --multi_turn_scheduler "${SCHEDULER}" \
  --max_turns "${MAX_TURNS}" \
  --completion_length_limit_scope total \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-32768}" \
  --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS:-1}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.50}" \
  --vllm_mm_processor_cache_gb "${VLLM_MM_PROCESSOR_CACHE_GB:-0}" \
  --vllm_engine_kwargs "{\"kv_cache_memory_bytes\": ${VLLM_KV_CACHE_MEMORY_BYTES:-8589934592}}" \
  --vllm_enforce_eager "${VLLM_ENFORCE_EAGER:-true}" \
  --vllm_enable_prefix_caching true \
  --vllm_enable_lora true \
  --vllm_limit_mm_per_prompt "{\"image\": ${MAX_TURNS}}" \
  --sleep_level "${VLLM_SLEEP_LEVEL:-2}" \
  --max_length "${MAX_LENGTH:-32768}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-24576}" \
  --max_pixels "${MAX_PIXELS:-4096}" \
  --num_train_epochs "${TRAIN_EPOCHS:-3}" \
  --max_steps "${TRAIN_STEPS:--1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --num_generations "${NUM_GENERATIONS}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE}" \
  --steps_per_generation "${STEPS_PER_GENERATION:-1}" \
  --learning_rate "${LR:-5e-6}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --target_modules all-linear \
  --gradient_checkpointing true \
  --use_liger_kernel "${USE_LIGER_KERNEL:-true}" \
  --bf16 true \
  --beta "${BETA:-0.01}" \
  --epsilon "${EPSILON:-0.2}" \
  --loss_type grpo \
  --importance_sampling_level token \
  --temperature "${TEMPERATURE:-1.0}" \
  --top_p "${TOP_P:-0.9}" \
  --logging_steps 1 \
  --save_steps 1 \
  --save_total_limit 2 \
  --output_dir "${OUTPUT_DIR}" \
  --log_completions true \
  --report_to none
