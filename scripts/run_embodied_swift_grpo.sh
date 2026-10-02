#!/usr/bin/env bash
set -euo pipefail
# 出错立即退出、未定义变量报错、管道任一段失败算整体失败。

# 中文总览
# --------
# 这是 Swift 命令的“拼装脚本”，负责把环境变量翻译成两条实际的命令行：
#   MODE=rollout -> swift rollout  起一个多轮 vLLM 推理服务（被 server 模式
#                   的训练端、以及 holdout 评估复用）；
#   MODE=train   -> swift rlhf --rlhf_type grpo  启动 GRPO 训练。
# 它本身不实现任何训练/推理逻辑：多轮环境交互（scheduler）和 reward 都注册在
# plugins/embodied_swift_grpo.py 里，通过 --external_plugins 注入 Swift。
# 调用方是 run_embodied_swift_formal10.sh 主循环（以及 staged 变体脚本）。

# Swift 的 rollout server/client 在本机端口通信，必须绕过环境中的代理。
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Swift 独立运行环境（venv）位置；swift 可执行文件即训练/rollout 入口。
SWIFT_VENV="${SWIFT_VENV:-/opt/openeta-swift}"
SWIFT_BIN="${SWIFT_VENV}/bin/swift"
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}"
# ROLE 决定注册哪个 scheduler/reward：alice（出题方）或 bob（复现方）。
ROLE="${ROLE:-bob}"
# 运行模式：第一个位置参数，rollout（起服务）或 train（训练）。
MODE="${1:-rollout}"
ROLLOUT_GPU="${ROLLOUT_GPU:-0}"
# 训练卡，可写逗号分隔多卡（如 0,1,...,7），train 模式会据此起多个 DDP rank。
TRAIN_GPU="${TRAIN_GPU:-1}"
ROLLOUT_HOST="${ROLLOUT_HOST:-127.0.0.1}"
ROLLOUT_PORT="${ROLLOUT_PORT:-8111}"
# 单条 episode 与 ManiSkill 最多交互的轮数。
MAX_TURNS="${MAX_TURNS:-4}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/runs/swift_smoke/${ROLE}_tasks.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/swift_smoke/${ROLE}_grpo}"
# 外部插件：注册 embodied_alice/bob_scheduler 与 embodied_alice/bob_reward。
PLUGIN="${REPO_ROOT}/plugins/embodied_swift_grpo.py"
# rollout 服务要加载的 LoRA（如冻结 Bob / 被评估的 Alice）；为空则直接用基座。
ROLLOUT_ADAPTER_PATH="${ROLLOUT_ADAPTER_PATH:-}"
# 训练起点 LoRA（上一轮的 adapter）；与 RESUME_FROM_CHECKPOINT 二选一。
TRAIN_ADAPTER_PATH="${TRAIN_ADAPTER_PATH:-}"
# 断点续训：从某个 checkpoint-N 继续（优先于 TRAIN_ADAPTER_PATH）。
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
# server：训练端连外部 rollout 服务；colocate：rollout 引擎与训练同卡管理。
VLLM_MODE="${VLLM_MODE:-server}"

# 前置检查：swift 可执行文件必须存在，否则提示先跑环境安装脚本。
if [[ ! -x "${SWIFT_BIN}" ]]; then
  echo "Swift runtime not found: ${SWIFT_BIN}" >&2
  echo "Run scripts/setup_embodied_swift_env.sh first." >&2
  exit 2
fi

# 按角色选择插件里注册的 scheduler（多轮环境交互逻辑）和 reward 函数名。
# 这两个名字会被拼进 swift 命令行，由 --external_plugins 加载后解析。
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

# 让 Swift 进程能 import 到仓库里的插件和 adapter 包。
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# rollout artifact（轨迹、快照、评估结果 JSON）的输出根目录。
export OPENETA_SWIFT_ARTIFACT_ROOT="${OPENETA_SWIFT_ARTIFACT_ROOT:-${OUTPUT_DIR}/rollouts}"
# PyTorch 显存分配器使用可扩展段，缓解长序列训练下的显存碎片。
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# ===========================================================================
# rollout 模式：起一个常驻的多轮 vLLM 推理服务
# ===========================================================================
# 用途有两种：1) server 训练模式下作为 Alice/Bob 的推理后端；
# 2) 每轮结束后的 holdout 评估（临时起一个 Bob server，temperature=0）。
if [[ "${MODE}" == "rollout" ]]; then
  export CUDA_VISIBLE_DEVICES="${ROLLOUT_GPU}"
  ROLLOUT_INIT_ARGS=()
  # 指定了 LoRA adapter 时开启 vLLM 的 LoRA 支持（冻结 Bob / 待评估 Alice）。
  if [[ -n "${ROLLOUT_ADAPTER_PATH}" ]]; then
    ROLLOUT_INIT_ARGS+=(
      --adapters "${ROLLOUT_ADAPTER_PATH}"
      --vllm_enable_lora true
      --vllm_max_lora_rank "${VLLM_MAX_LORA_RANK:-16}"
    )
  fi
  # 注意：--max_new_tokens 12 只是 CLI 默认上限；thinking 开启时插件内的
  # scheduler 会用 OPENETA_THINKING_MAX_TOKENS 覆盖每个环境 turn 的实际上限。
  # --vllm_mm_processor_cache_gb 0：关闭多模态 processor 缓存，避免多进程
  # 共卡时额外占显存。--vllm_limit_mm_per_prompt image=MAX_TURNS：多轮交互
  # 每轮都有一张环境观测图，所以每个 prompt 最多 MAX_TURNS 张图。
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

# ===========================================================================
# train 模式：启动 GRPO 训练（swift rlhf --rlhf_type grpo）
# ===========================================================================
if [[ "${MODE}" != "train" ]]; then
  echo "Usage: ROLE=bob|alice $0 rollout|train" >&2
  exit 2
fi

# 训练数据集（create_embodied_swift_dataset.py 生成的 JSONL）必须存在。
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 2
fi

# Alice 的 reward 需要一个“冻结的 Bob”做难度评估：server 模式下必须给出
# 外部 Bob rollout server 的地址。（staged colocate 模式由同 engine 内的
# 冻结 Bob LoRA 评估，不走这个脚本分支。）
if [[ "${ROLE}" == "alice" && -z "${OPENETA_BOB_EVALUATOR_URL:-}" ]]; then
  echo "Alice training requires OPENETA_BOB_EVALUATOR_URL pointing to a frozen Bob rollout server." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"
# TRAIN_GPU may be a comma-separated list (for example 2,3,4,5).  Swift's
# launcher reads NPROC_PER_NODE and creates one DDP worker per visible GPU.
# 中文说明：每张可见训练卡对应一个 DDP rank；NPROC_PER_NODE 可显式覆盖，
# 否则自动等于 TRAIN_GPU 列表里的卡数。
if [[ -n "${TRAIN_NPROC_PER_NODE:-}" ]]; then
  export NPROC_PER_NODE="${TRAIN_NPROC_PER_NODE}"
else
  IFS=',' read -r -a OPENETA_TRAIN_GPU_LIST <<<"${TRAIN_GPU}"
  export NPROC_PER_NODE="${#OPENETA_TRAIN_GPU_LIST[@]}"
fi
# 训练初始化参数：优先从 checkpoint 断点续训；否则以上一轮 LoRA adapter
# 作为起点；两者都没有则从基座模型冷启动。
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
  # server 模式：rollout 走独立的外部 vLLM 服务（本脚本 rollout 模式起的），
  # 训练进程只作为客户端连过去；pass_dataset 表示把数据行直接发给 server，
  # 由 server 侧的 scheduler 驱动多轮环境交互。
  VLLM_TRAIN_ARGS+=(
    --vllm_server_host "${ROLLOUT_HOST}"
    --vllm_server_port "${ROLLOUT_PORT}"
    --vllm_server_timeout "${VLLM_SERVER_TIMEOUT:-600}"
    --vllm_server_pass_dataset true
  )
elif [[ "${VLLM_MODE}" == "colocate" ]]; then
  # colocate 模式：rollout 引擎与训练进程同卡，由 Swift 统一调度
  # sleep/wake；scheduler 在训练进程内直接驱动 ManiSkill 多轮交互。
  # completion_length_limit_scope=total：MAX_COMPLETION_LENGTH 按整条多轮
  # completion 的总长度计算，而不是每个 turn。
  VLLM_TRAIN_ARGS+=(
    --multi_turn_scheduler "${SCHEDULER}"
    --max_turns "${MAX_TURNS}"
    --completion_length_limit_scope total
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-65536}"
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS:-1}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
    --vllm_mm_processor_cache_gb "${VLLM_MM_PROCESSOR_CACHE_GB:-0}"
    --vllm_engine_kwargs "{\"kv_cache_memory_bytes\": ${VLLM_KV_CACHE_MEMORY_BYTES:-8589934592}}" \
    --vllm_enforce_eager true
    --vllm_limit_mm_per_prompt "{\"image\": ${MAX_TURNS}}"
    # sleep_level=2：训练阶段把 vLLM 权重和 KV cache 都卸载，腾出显存给训练。
    --sleep_level "${VLLM_SLEEP_LEVEL:-2}"
  )
else
  echo "VLLM_MODE must be server or colocate" >&2
  exit 2
fi
# 正式拼出 GRPO 训练命令。要点：
# - --rlhf_type grpo / --loss_type grpo / --importance_sampling_level token：
#   组内相对优势（GRPO），重要性采样校正按 token 粒度做；
# - --beta KL 系数、--epsilon clip 范围、--temperature/--top_p 采样参数；
# - --tuner_type lora + all-linear：只训练 LoRA（rank/alpha），基座冻结；
# - --num_generations / --generation_batch_size / --steps_per_generation：
#   每个 prompt 的组大小与 rollout/训练步的配比（上游脚本已校验自洽）；
# - --max_steps -1：不按固定步数停，跑满 --num_train_epochs；
# - --save_steps 1 --save_total_limit 2：每步存 checkpoint、只留最近 2 个，
#   保证断点续训始终有最新现场且不占爆磁盘；
# - --load_from_cache_file false --dataset_num_proc 1：数据集含环境快照，
#   不走缓存、单进程加载，保证 snapshot 语义不被预处理破坏。
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
  --report_to "${REPORT_TO:-tensorboard}" \
  --logging_dir "${TENSORBOARD_LOGGING_DIR:-${OUTPUT_DIR}/tensorboard}"
