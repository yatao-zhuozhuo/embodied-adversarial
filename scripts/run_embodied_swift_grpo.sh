#!/usr/bin/env bash
set -euo pipefail

# Swift 的 rollout server/client 在本机端口通信，必须绕过环境中的代理。
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_VENV="${SWIFT_VENV:-/opt/openeta-swift}"
SWIFT_BIN="${SWIFT_VENV}/bin/swift"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}"
ROLE="${ROLE:-bob}"
MODE="${1:-rollout}"
ROLLOUT_GPU="${ROLLOUT_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-1}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8111}"
MAX_TURNS="${MAX_TURNS:-4}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/runs/swift_smoke/${ROLE}_tasks.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/swift_smoke/${ROLE}_grpo}"
PLUGIN="${REPO_ROOT}/plugins/embodied_swift_grpo.py"
ROLLOUT_ADAPTER_PATH="${ROLLOUT_ADAPTER_PATH:-}"
TRAIN_ADAPTER_PATH="${TRAIN_ADAPTER_PATH:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
VLLM_MODE="${VLLM_MODE:-server}"

if [[ ! -x "${SWIFT_BIN}" ]]; then
  echo "Swift runtime not found: ${SWIFT_BIN}" >&2
  echo "Run scripts/setup_embodied_swift_env.sh first." >&2
  exit 2
fi

if [[ "${ROLE}" == "bob" ]]; then
  SCHEDULER="embodied_bob_scheduler"
  REWARD_FUNC="embodied_bob_reward"
elif [[ "${ROLE}" == "alice" ]]; then
  SCHEDULER="embodied_alice_scheduler"
  REWARD_FUNC="embodied_alice_reward"
else
  echo "ROLE must be alice or bob" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENETA_SWIFT_ARTIFACT_ROOT="${OPENETA_SWIFT_ARTIFACT_ROOT:-${OUTPUT_DIR}/rollouts}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ "${MODE}" == "rollout" ]]; then
  export CUDA_VISIBLE_DEVICES="${ROLLOUT_GPU}"
  ROLLOUT_INIT_ARGS=()
  if [[ -n "${ROLLOUT_ADAPTER_PATH}" ]]; then
    ROLLOUT_INIT_ARGS+=(
      --adapters "${ROLLOUT_ADAPTER_PATH}"
      --vllm_enable_lora true
      --vllm_max_lora_rank "${VLLM_MAX_LORA_RANK:-16}"
    )
  fi
  exec "${SWIFT_BIN}" rollout \
    --model "${MODEL_PATH}" \
    --load_args false \
    "${ROLLOUT_INIT_ARGS[@]}" \
    --infer_backend vllm \
    --external_plugins "${PLUGIN}" \
    --multi_turn_scheduler "${SCHEDULER}" \
    --max_turns "${MAX_TURNS}" \
    --max_new_tokens 12 \
    --max_pixels "${MAX_PIXELS:-12544}" \
    --vllm_use_async_engine true \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-8192}" \
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS:-4}" \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.50}" \
    --vllm_mm_processor_cache_gb "${VLLM_MM_PROCESSOR_CACHE_GB:-0}" \
    --vllm_engine_kwargs "{\"kv_cache_memory_bytes\": ${VLLM_KV_CACHE_MEMORY_BYTES:-2147483648}}" \
    --vllm_enforce_eager true \
    --vllm_limit_mm_per_prompt "{\"image\": ${MAX_TURNS}}" \
    --host "${ROLLOUT_HOST}" \
    --port "${ROLLOUT_PORT}"
fi

if [[ "${MODE}" != "train" ]]; then
  echo "Usage: ROLE=bob|alice $0 rollout|train" >&2
  exit 2
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 2
fi

if [[ "${ROLE}" == "alice" && -z "${OPENETA_BOB_EVALUATOR_URL:-}" ]]; then
  echo "Alice training requires OPENETA_BOB_EVALUATOR_URL pointing to a frozen Bob rollout server." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"
# TRAIN_GPU may be a comma-separated list (for example 2,3,4,5).  Swift's
# launcher reads NPROC_PER_NODE and creates one DDP worker per visible GPU.
if [[ -n "${TRAIN_NPROC_PER_NODE:-}" ]]; then
  export NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}"
else
  IFS=',' read -r -a OPENETA_TRAIN_GPU_LIST <<<"${TRAIN_GPU}"
  export NPROC_PER_NODE="${#OPENETA_TRAIN_GPU_LIST[@]}"
fi
TRAIN_INIT_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  TRAIN_INIT_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
elif [[ -n "${TRAIN_ADAPTER_PATH}" ]]; then
  TRAIN_INIT_ARGS+=(--adapters "${TRAIN_ADAPTER_PATH}")
fi
VLLM_TRAIN_ARGS=(
  --use_vllm true
  --vllm_mode "${VLLM_MODE}"
)
if [[ "${VLLM_MODE}" == "server" ]]; then
  VLLM_TRAIN_ARGS+=(
    --vllm_server_host "${ROLLOUT_HOST}"
    --vllm_server_port "${ROLLOUT_PORT}"
    --vllm_server_timeout "${VLLM_SERVER_TIMEOUT:-600}"
    --vllm_server_pass_dataset true
  )
elif [[ "${VLLM_MODE}" == "colocate" ]]; then
  VLLM_TRAIN_ARGS+=(
    --multi_turn_scheduler "${SCHEDULER}"
    --max_turns "${MAX_TURNS}"
    --completion_length_limit_scope total
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-65536}"
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS:-1}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
    --vllm_mm_processor_cache_gb "${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
    --vllm_engine_kwargs "{\"kv_cache_memory_bytes\": ${VLLM_KV_CACHE_MEMORY_BYTES:-8589934592}}"
    --vllm_enforce_eager true
    --vllm_limit_mm_per_prompt "{\"image\": ${MAX_TURNS}}"
    --sleep_level "${VLLM_SLEEP_LEVEL:-2}"
  )
else
  echo "VLLM_MODE must be server or colocate" >&2
  exit 2
fi
exec "${SWIFT_BIN}" rlhf \
  --rlhf_type grpo \
  --model "${MODEL_PATH}" \
  "${TRAIN_INIT_ARGS[@]}" \
  --external_plugins "${PLUGIN}" \
  --reward_funcs "${REWARD_FUNC}" \
  --dataset "${DATASET_PATH}" \
  --load_from_cache_file false \
  --dataset_num_proc 1 \
  "${VLLM_TRAIN_ARGS[@]}" \
  --max_length "${MAX_LENGTH:-8192}" \
  --max_completion_length "${MAX_COMPLETION_LENGTH:-128}" \
  --max_pixels "${MAX_PIXELS:-12544}" \
  --num_train_epochs "${TRAIN_EPOCHS:-3}" \
  --max_steps "${TRAIN_STEPS:--1}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --num_generations "${NUM_GENERATIONS:-8}" \
  --generation_batch_size "${GENERATION_BATCH_SIZE:-${NUM_GENERATIONS:-8}}" \
  --steps_per_generation "${STEPS_PER_GENERATION:-${NUM_GENERATIONS:-8}}" \
  --learning_rate "${LR:-5e-6}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-constant}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-8}" \
  --lora_alpha "${LORA_ALPHA:-16}" \
  --target_modules all-linear \
  --gradient_checkpointing true \
  --use_liger_kernel "${USE_LIGER_KERNEL:-false}" \
  --bf16 true \
  --beta "${BETA:-0.01}" \
  --epsilon "${EPSILON:-0.2}" \
  --loss_type grpo \
  --importance_sampling_level token \
  --temperature "${TEMPERATURE:-0.7}" \
  --top_p "${TOP_P:-0.9}" \
  --logging_steps 1 \
  --save_steps 1 \
  --save_total_limit 2 \
  --output_dir "${OUTPUT_DIR}" \
  --log_completions true \
  --report_to none
