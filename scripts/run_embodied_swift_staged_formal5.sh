#!/usr/bin/env bash
# Launch the formal five-round, eight-GPU staged-colocate self-play run.
#
# All settings below are defaults. Override any of them on the command line:
#
#   RUN_NAME=my_formal ROUNDS=10 MAX_TURNS=40 \
#     ./scripts/run_embodied_swift_staged_formal5.sh
#
# Validate and print the resolved configuration without starting training:
#
#   DRY_RUN=true ./scripts/run_embodied_swift_staged_formal5.sh
#
# Reusing RUN_NAME/RUN_ROOT resumes completed rounds from summary.json and
# lets Swift resume the active round from its latest checkpoint.
set -euo pipefail

ulimit -c 0 2>/dev/null || true

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# FLA 0.5.2 rejects Triton 3.4--3.6 for a gated-delta backward kernel on
# Hopper because that combination can silently return wrong gradients.  Keep
# Torch's bundled Triton untouched and load the supported TileLang backend
# from a private, shared overlay instead.
OPENETA_TILELANG_OVERLAY="${OPENETA_TILELANG_OVERLAY:-/inspire/qb-ilm/project/exploration-topic/ky26060/openeta-runtime/tilelang-0.1.14-py312}"
OPENETA_REQUIRE_TILELANG="${OPENETA_REQUIRE_TILELANG:-true}"

if [[ -z "${VK_ICD_FILENAMES:-}" && \
      -s /etc/vulkan/icd.d/nvidia_icd.json ]] && \
   grep -q '"library_path"' /etc/vulkan/icd.d/nvidia_icd.json; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
if [[ -e /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ]]; then
  OPENETA_VULKAN_LOADER=/usr/lib/x86_64-linux-gnu/libvulkan.so.1
else
  OPENETA_VULKAN_LOADER=sapien-bundled
fi

# ---------------------------------------------------------------------------
# 1. Experiment identity and scale
# ---------------------------------------------------------------------------
RUN_NAME="${RUN_NAME:-staged_colocate_5round}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/${RUN_NAME}}"
DRY_RUN="${DRY_RUN:-false}"
ROUNDS="${ROUNDS:-5}"
TASKS_PER_ROUND="${TASKS_PER_ROUND:-4}"

# ---------------------------------------------------------------------------
# 2. Eight-rank staged execution
# ---------------------------------------------------------------------------
TRAIN_GPU="${TRAIN_GPU:-0,1,2,3,4,5,6,7}"
DATASET_GPU="${DATASET_GPU:-${TRAIN_GPU%%,*}}"
# Keep CUDA compute distributed across all eight ranks.  Each rank delegates
# ManiSkill to a fresh spawn child so SAPIEN's Vulkan/CUDA context never shares
# a process with vLLM's sleep-mode CuMem allocator.  The worker resets its own
# CUDA visibility before importing SAPIEN, so cuda:0 means physical GPU 0 even
# after Swift rewrites CUDA_VISIBLE_DEVICES separately in every training rank.
OPENETA_MANISKILL_PROCESS_ISOLATION="${OPENETA_MANISKILL_PROCESS_ISOLATION:-true}"
OPENETA_MANISKILL_RENDER_GPU="${OPENETA_MANISKILL_RENDER_GPU:-0}"
OPENETA_MANISKILL_RENDER_BACKEND="${OPENETA_MANISKILL_RENDER_BACKEND:-sapien_cuda:0}"
VLLM_MODE=colocate
OPENETA_STAGED_COLOCATE=true
MAX_TURNS="${MAX_TURNS:-32}"

# One prompt produces an eight-completion GRPO group. TP=1 is enforced by the
# staged entry, so every visible GPU owns one local vLLM engine and one DDP rank.
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-8}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_STEPS="${TRAIN_STEPS:--1}"

# ---------------------------------------------------------------------------
# 3. Alice/Bob evaluation semantics
# ---------------------------------------------------------------------------
# 这里需要看一下 *************************************
OPENETA_BOB_EVALUATIONS="${OPENETA_BOB_EVALUATIONS:-2}"
OPENETA_BOB_EVAL_RETRIES="${OPENETA_BOB_EVAL_RETRIES:-1}"
OPENETA_TARGET_SUCCESS_RATE="${OPENETA_TARGET_SUCCESS_RATE:-0.45}"
OPENETA_FORMAT_PENALTY="${OPENETA_FORMAT_PENALTY:-0.10}"

# ---------------------------------------------------------------------------
# 4. Generation and context lengths
# ---------------------------------------------------------------------------
OPENETA_ENABLE_THINKING="${OPENETA_ENABLE_THINKING:-true}"
OPENETA_THINKING_MAX_TOKENS="${OPENETA_THINKING_MAX_TOKENS:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-24576}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
MAX_PIXELS="${MAX_PIXELS:-4096}"

# ---------------------------------------------------------------------------
# 5. vLLM and training memory/runtime settings
# ---------------------------------------------------------------------------
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.28}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-4294967296}"
VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-2}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}"
OPENETA_KEEP_LOGITS_BF16="${OPENETA_KEEP_LOGITS_BF16:-true}"

LR="${LR:-5e-6}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
BETA="${BETA:-0.01}"
EPSILON="${EPSILON:-0.2}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"

# ---------------------------------------------------------------------------
# 6. Model and initial policies
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}"
INITIAL_ALICE_ADAPTER="${INITIAL_ALICE_ADAPTER:-${REPO_ROOT}/runs/alice_dagger_curriculum/round-0007/alice_adapter}"
INITIAL_BOB_ADAPTER="${INITIAL_BOB_ADAPTER:-${REPO_ROOT}/runs/embodied_selfplay_grpo16/round-0001/bob_adapter}"
BOOTSTRAP_ALICE_RUN="${BOOTSTRAP_ALICE_RUN:-${REPO_ROOT}/runs/embodied_selfplay_grpo16}"

# Holdout evaluation still uses one temporary Bob server after each round.
BOB_ROLLOUT_GPU="${BOB_ROLLOUT_GPU:-0}"
BOB_PORT="${BOB_PORT:-8121}"
ALICE_PORT="${ALICE_PORT:-8122}"

# ---------------------------------------------------------------------------
# 7. Fail-fast validation
# ---------------------------------------------------------------------------
IFS=',' read -r -a OPENETA_GPU_LIST <<<"${TRAIN_GPU}"
if [[ "${#OPENETA_GPU_LIST[@]}" -ne 8 ]]; then
  echo "TRAIN_GPU must contain exactly eight comma-separated GPU IDs" >&2
  exit 2
fi
if [[ ! -f "${INITIAL_ALICE_ADAPTER}/adapter_config.json" ]]; then
  echo "Alice adapter is missing: ${INITIAL_ALICE_ADAPTER}" >&2
  exit 2
fi
if [[ ! -f "${INITIAL_BOB_ADAPTER}/adapter_config.json" ]]; then
  echo "Bob adapter is missing: ${INITIAL_BOB_ADAPTER}" >&2
  exit 2
fi
if [[ ! -d "${BOOTSTRAP_ALICE_RUN}" ]]; then
  echo "Bootstrap Alice run is missing: ${BOOTSTRAP_ALICE_RUN}" >&2
  exit 2
fi
if [[ "${OPENETA_REQUIRE_TILELANG,,}" =~ ^(1|true|yes|on)$ && \
      ! -d "${OPENETA_TILELANG_OVERLAY}" ]]; then
  echo "TileLang overlay is missing: ${OPENETA_TILELANG_OVERLAY}" >&2
  echo "Run scripts/setup_fla_tilelang_overlay.sh before formal training." >&2
  exit 2
fi
if [[ "${ROUNDS}" -lt 1 || "${MAX_TURNS}" -lt 1 ]]; then
  echo "ROUNDS and MAX_TURNS must be positive" >&2
  exit 2
fi
EXPECTED_GENERATION_BATCH_SIZE=$((
  PER_DEVICE_TRAIN_BATCH_SIZE * 8 * STEPS_PER_GENERATION
))
if [[ "${GENERATION_BATCH_SIZE}" -ne "${EXPECTED_GENERATION_BATCH_SIZE}" ]]; then
  echo "invalid GRPO batch settings: GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE}, but" >&2
  echo "PER_DEVICE_TRAIN_BATCH_SIZE * world_size * STEPS_PER_GENERATION = ${EXPECTED_GENERATION_BATCH_SIZE}" >&2
  exit 2
fi

echo "Starting staged embodied self-play"
echo "  run_root:       ${RUN_ROOT}"
echo "  rounds:         ${ROUNDS}"
echo "  train_gpus:     ${TRAIN_GPU}"
echo "  dataset_gpu:    ${DATASET_GPU}"
echo "  render_backend: ${OPENETA_MANISKILL_RENDER_BACKEND}"
echo "  render_gpu:     ${OPENETA_MANISKILL_RENDER_GPU}"
echo "  render_process: ${OPENETA_MANISKILL_PROCESS_ISOLATION}"
echo "  tilelang:       ${OPENETA_TILELANG_OVERLAY}"
echo "  vulkan_icd:     ${VK_ICD_FILENAMES:-auto-detect}"
echo "  vulkan_loader:  ${OPENETA_VULKAN_LOADER}"
echo "  max_turns:      ${MAX_TURNS}"
echo "  generations:    ${NUM_GENERATIONS}"
echo "  alice_adapter:  ${INITIAL_ALICE_ADAPTER}"
echo "  bob_adapter:    ${INITIAL_BOB_ADAPTER}"

if [[ "${DRY_RUN,,}" =~ ^(1|true|yes|on)$ ]]; then
  echo "Dry run complete; training was not started."
  exit 0
fi

mkdir -p "${RUN_ROOT}"

export RUN_ROOT ROUNDS TASKS_PER_ROUND
export TRAIN_GPU DATASET_GPU OPENETA_MANISKILL_RENDER_BACKEND
export OPENETA_MANISKILL_PROCESS_ISOLATION OPENETA_MANISKILL_RENDER_GPU
export OPENETA_TILELANG_OVERLAY OPENETA_REQUIRE_TILELANG
export VLLM_MODE OPENETA_STAGED_COLOCATE MAX_TURNS
export NUM_GENERATIONS GENERATION_BATCH_SIZE STEPS_PER_GENERATION
export PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS
export TRAIN_EPOCHS TRAIN_STEPS
export OPENETA_BOB_EVALUATIONS OPENETA_BOB_EVAL_RETRIES
export OPENETA_TARGET_SUCCESS_RATE OPENETA_FORMAT_PENALTY
export OPENETA_ENABLE_THINKING OPENETA_THINKING_MAX_TOKENS
export MAX_COMPLETION_LENGTH MAX_LENGTH VLLM_MAX_MODEL_LEN MAX_PIXELS
export VLLM_MAX_NUM_SEQS VLLM_GPU_MEMORY_UTILIZATION
export VLLM_KV_CACHE_MEMORY_BYTES VLLM_SLEEP_LEVEL VLLM_ENFORCE_EAGER
export USE_LIGER_KERNEL OPENETA_KEEP_LOGITS_BF16
export LR LR_SCHEDULER_TYPE TEMPERATURE TOP_P BETA EPSILON
export LORA_RANK LORA_ALPHA MODEL_PATH
export INITIAL_ALICE_ADAPTER INITIAL_BOB_ADAPTER BOOTSTRAP_ALICE_RUN
export BOB_ROLLOUT_GPU BOB_PORT ALICE_PORT

exec "${REPO_ROOT}/scripts/run_embodied_swift_formal10.sh"
