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
#
# 中文总览
# --------
# 这是一个“配置包装脚本”，本身不实现训练逻辑。它的职责只有三件事：
#   1. 为 5 轮、8 卡的 staged-colocate Alice/Bob self-play 正式实验设定
#      一套经过验证的默认参数（全部可用环境变量在命令行覆盖）；
#   2. 做 fail-fast 的前置检查（GPU 数量、初始 adapter、TileLang overlay、
#      GRPO batch 配置是否自洽），不满足就直接退出，不浪费排队/开机时间；
#   3. 把所有参数 export 后，用 exec 把进程替换为真正的主循环脚本
#      scripts/run_embodied_swift_formal10.sh。
#
# 断点续跑：复用同一个 RUN_NAME/RUN_ROOT 重启时，已完成的轮次靠
# summary.json 跳过，当前轮由 Swift 从最新 checkpoint 继续。
set -euo pipefail
# -e：命令失败立即退出；-u：引用未定义变量报错；pipefail：管道任一段失败
# 都算失败。防止训练失败后还继续导出配置、进入主循环。

ulimit -c 0 2>/dev/null || true
# 禁止生成 core dump。SAPIEN/Vulkan 崩溃时 core 文件可达数 GB，会撑爆磁盘；
# 真正有用的报错信息在 Python traceback 和日志里。

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# 无论从哪里调用本脚本，都切到仓库根目录，保证后续相对路径一致。

# FLA 0.5.2 rejects Triton 3.4--3.6 for a gated-delta backward kernel on
# Hopper because that combination can silently return wrong gradients.  Keep
# Torch's bundled Triton untouched and load the supported TileLang backend
# from a private, shared overlay instead.
# 中文说明：FLA 0.5.2 在 Hopper 上对 gated-delta 反向 kernel 拒绝使用
# Triton 3.4~3.6（该组合会静默给出错误梯度）。这里不动 PyTorch 自带的
# Triton，而是从一个独立的共享 overlay 目录加载受支持的 TileLang 后端。
# OPENETA_REQUIRE_TILELANG=true 表示 overlay 缺失时直接拒绝启动（见第 7 节）。
OPENETA_TILELANG_OVERLAY="${OPENETA_TILELANG_OVERLAY:-/inspire/qb-ilm/project/exploration-topic/ky26060/openeta-runtime/tilelang-0.1.14-py312}"
OPENETA_REQUIRE_TILELANG="${OPENETA_REQUIRE_TILELANG:-true}"

# 自动探测 NVIDIA Vulkan ICD：仅在用户没有手动指定、且
# /etc/vulkan/icd.d/nvidia_icd.json 非空且确实声明了 library_path 时才导出，
# 避免把空占位文件塞给 Vulkan 导致 ErrorIncompatibleDriver。
if [[ -z "${VK_ICD_FILENAMES:-}" && \
      -s /etc/vulkan/icd.d/nvidia_icd.json ]] && \
   grep -q '"library_path"' /etc/vulkan/icd.d/nvidia_icd.json; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi
# 优先使用系统 Vulkan loader；没有则回退到 SAPIEN 自带的 loader。
if [[ -e /usr/lib/x86_64-linux-gnu/libvulkan.so.1 ]]; then
  OPENETA_VULKAN_LOADER=/usr/lib/x86_64-linux-gnu/libvulkan.so.1
else
  OPENETA_VULKAN_LOADER=sapien-bundled
fi

# ---------------------------------------------------------------------------
# 1. Experiment identity and scale
# ---------------------------------------------------------------------------
# 实验身份与规模：RUN_NAME 决定输出目录；复用同名 RUN_ROOT 即可断点续跑。
RUN_NAME="${RUN_NAME:-staged_colocate_5round}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/${RUN_NAME}}"
DRY_RUN="${DRY_RUN:-false}"
ROUNDS="${ROUNDS:-5}"
TASKS_PER_ROUND="${TASKS_PER_ROUND:-4}"
OPENETA_COLD_START="${OPENETA_COLD_START:-false}"
SWIFT_VENV="${SWIFT_VENV:-${REPO_ROOT}/../.venv_selfplay_embodied}"
PYTHON="${SWIFT_VENV}/bin/python"
# Future runs write local TensorBoard event files by default. Set
# REPORT_TO=wandb after installing/logging into wandb to use the remote backend.
REPORT_TO="${REPORT_TO:-tensorboard}"
TENSORBOARD_LOGGING_DIR="${TENSORBOARD_LOGGING_DIR:-${RUN_ROOT}/tensorboard}"

# ---------------------------------------------------------------------------
# 2. Eight-rank staged execution
# ---------------------------------------------------------------------------
# 8 卡 staged 执行：每张训练卡同时是一个 DDP rank、一个本地 vLLM engine。
TRAIN_GPU="${TRAIN_GPU:-0,1,2,3,4,5,6,7}"
# 数据集/snapshot 创建是单进程任务，固定用第一张训练卡，避免 torch fork_rng
# 在分布式训练启动前把 8 张卡全部初始化。
DATASET_GPU="${DATASET_GPU:-${TRAIN_GPU%%,*}}"
# Keep CUDA compute distributed across all eight ranks.  Each rank delegates
# ManiSkill to a fresh spawn child so SAPIEN's Vulkan/CUDA context never shares
# a process with vLLM's sleep-mode CuMem allocator.  The worker resets its own
# CUDA visibility before importing SAPIEN, so cuda:0 means physical GPU 0 even
# after Swift rewrites CUDA_VISIBLE_DEVICES separately in every training rank.
# 中文说明：CUDA 计算分散在全部 8 个 rank 上；每个 rank 把 ManiSkill 仿真
# 委托给一个独立 spawn 出来的子进程（进程隔离），使 SAPIEN 的 Vulkan/CUDA
# 上下文不与 vLLM sleep 模式的 CuMem 分配器同进程，避免互相干扰。
# 渲染固定走物理 GPU 0 的 sapien_cuda:0 后端。
OPENETA_MANISKILL_PROCESS_ISOLATION="${OPENETA_MANISKILL_PROCESS_ISOLATION:-true}"
OPENETA_MANISKILL_RENDER_GPU="${OPENETA_MANISKILL_RENDER_GPU:-0}"
OPENETA_MANISKILL_RENDER_BACKEND="${OPENETA_MANISKILL_RENDER_BACKEND:-sapien_cuda:0}"
# colocate：rollout 引擎与训练进程同卡、由 Swift 统一管理（sleep/wake）。
# OPENETA_STAGED_COLOCATE=true：在同一个 engine 内按“Alice LoRA(id=1) 出题
# -> 冻结 Bob LoRA(id=2) 评估 -> GRPO 训练”三阶段调度，训练期间不起外部
# rollout server。
VLLM_MODE=colocate
OPENETA_STAGED_COLOCATE=true
# 单条 episode 与 ManiSkill 最多交互多少轮；模型可以提前 DONE 结束。
MAX_TURNS="${MAX_TURNS:-32}"

# One prompt produces an eight-completion GRPO group. TP=1 is enforced by the
# staged entry, so every visible GPU owns one local vLLM engine and one DDP rank.
# GRPO batch 配置：一个 prompt 采 8 条 completion 构成一个组，计算组内相对
# 优势。TP=1，每张可见 GPU 对应一个本地 vLLM engine + 一个 DDP rank。
# micro batch=1/卡，梯度累积 8 步 -> 一次 optimizer update 消耗
# 1*8卡*8累积 = 64 条 completion；STEPS_PER_GENERATION=1 表示每个 micro step
# 都重新 rollout，完全 on-policy。TRAIN_STEPS=-1 表示按 epoch（3 轮）训练，
# 不用固定 step 数提前终止。
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
# 对抗评估语义（Alice 的 reward 边界）：
# - OPENETA_BOB_EVALUATIONS=2：Alice 每提出一个有效题目，由当前“冻结的
#   Bob”独立复现 2 次（每次最多 MAX_TURNS 轮），用成功率衡量题目难度；
# - OPENETA_BOB_EVAL_RETRIES=1：评估请求失败时最多重试 1 次；
# - OPENETA_TARGET_SUCCESS_RATE=0.45：Alice 的奖励在 Bob 成功率接近 0.45
#   时最高——太容易（Bob 总能成功）或太难（Bob 从不能成功）的题目都扣分，
#   以此驱动 Alice 产出“卡在 Bob 能力边界上”的题目；
# - OPENETA_FORMAT_PENALTY=0.10：输出不符合 `</think>\n合法动作` 格式时
#   单独扣除的最大罚分。
OPENETA_BOB_EVALUATIONS="${OPENETA_BOB_EVALUATIONS:-2}"
OPENETA_BOB_EVAL_RETRIES="${OPENETA_BOB_EVAL_RETRIES:-1}"
OPENETA_TARGET_SUCCESS_RATE="${OPENETA_TARGET_SUCCESS_RATE:-0.45}"
OPENETA_FORMAT_PENALTY="${OPENETA_FORMAT_PENALTY:-0.10}"

# ---------------------------------------------------------------------------
# 4. Generation and context lengths
# ---------------------------------------------------------------------------
# 生成与上下文长度：开启 thinking，每个环境 turn 最多生成 1024 个思考
# token；最坏情况一条多轮 completion 总长 = MAX_TURNS * 1024。
# MAX_COMPLETION_LENGTH 是整条 completion 的总上限；MAX_LENGTH 还要容纳
# system/user prompt、图像 token 和环境反馈；VLLM_MAX_MODEL_LEN 是 engine
# 的上下文窗口。MAX_PIXELS 限制输入图像分辨率（降低视觉 token 数和显存）。
OPENETA_ENABLE_THINKING="${OPENETA_ENABLE_THINKING:-true}"
OPENETA_THINKING_MAX_TOKENS="${OPENETA_THINKING_MAX_TOKENS:-1024}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-24576}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
MAX_PIXELS="${MAX_PIXELS:-4096}"
OPENETA_HISTORY_MODE="${OPENETA_HISTORY_MODE:-full}"
OPENETA_TRAJECTORY_MAX_TOKENS="${OPENETA_TRAJECTORY_MAX_TOKENS:-${MAX_COMPLETION_LENGTH}}"
OPENETA_CONTEXT_MAX_TOKENS="${OPENETA_CONTEXT_MAX_TOKENS:-${MAX_LENGTH}}"

# ---------------------------------------------------------------------------
# 5. vLLM and training memory/runtime settings
# ---------------------------------------------------------------------------
# vLLM 与训练的显存/运行时设置：
# - MAX_NUM_SEQS=1：每个 engine 同时只处理 1 条序列，压低长序列峰值显存；
# - GPU_MEMORY_UTILIZATION=0.28 + 固定 4GiB KV cache：colocate 下同一张卡
#   还要放训练模型/优化器，所以给 vLLM 的显存配额很小且精确可控；
# - SLEEP_LEVEL=2：训练阶段把 vLLM 权重和 KV cache 都卸载到 CPU，释放显存；
# - ENFORCE_EAGER=true：禁用 CUDA graph，换取 sleep/wake 切换的稳定性；
# - USE_LIGER_KERNEL=false：不用 Liger 融合 kernel；
# - KEEP_LOGITS_BF16=true：GRPO 的 old/reference 前向不再额外物化一份 fp32
#   的 logits 副本，显著省显存。
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.28}"
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-4294967296}"
VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-2}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-true}"
USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}"
OPENETA_KEEP_LOGITS_BF16="${OPENETA_KEEP_LOGITS_BF16:-true}"

# 优化与采样超参：GRPO（BETA 是 KL 系数，EPSILON 是 clip 范围），
# 策略只训练 LoRA（rank=8, alpha=16），基座模型冻结。
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
# 基座模型与初始策略：常规模式从已有 Alice/Bob LoRA 起步。冷启动模式
# 则把两个角色都标记为 __base_model__：Round 0 不加载任何已有 adapter，
# 由 Swift 在训练开始时为 Alice/Bob 分别新建 LoRA。
MODEL_PATH="${MODEL_PATH:-/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B}"
OPENETA_BASE_MODEL_POLICY="${OPENETA_BASE_MODEL_POLICY:-__base_model__}"
if [[ "${OPENETA_COLD_START,,}" =~ ^(1|true|yes|on)$ ]]; then
  COLD_START_ROOT="${COLD_START_ROOT:-${RUN_ROOT}/cold_start}"
  INITIAL_ALICE_ADAPTER="${INITIAL_ALICE_ADAPTER:-${OPENETA_BASE_MODEL_POLICY}}"
  INITIAL_BOB_ADAPTER="${INITIAL_BOB_ADAPTER:-${OPENETA_BASE_MODEL_POLICY}}"
  BOOTSTRAP_ALICE_RUN="${BOOTSTRAP_ALICE_RUN:-${COLD_START_ROOT}/bootstrap_alice_run}"
  if [[ ! -x "${PYTHON}" ]]; then
    echo "Cold-start Python is missing: ${PYTHON}" >&2
    exit 2
  fi
  if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Cold-start base model is missing or incomplete: ${MODEL_PATH}" >&2
    exit 2
  fi
  if [[ ! -d "${BOOTSTRAP_ALICE_RUN}" ]] || \
     [[ $(find "${BOOTSTRAP_ALICE_RUN}" -maxdepth 1 -name 'cold_task_*.json' 2>/dev/null | wc -l) -lt 8 ]]; then
    CUDA_VISIBLE_DEVICES="${DATASET_GPU}" "${PYTHON}" \
      "${REPO_ROOT}/scripts/create_embodied_cold_bootstrap.py" \
      --output "${BOOTSTRAP_ALICE_RUN}" --count 8
  fi
else
  INITIAL_ALICE_ADAPTER="${INITIAL_ALICE_ADAPTER:-${REPO_ROOT}/runs/alice_dagger_curriculum/round-0007/alice_adapter}"
  INITIAL_BOB_ADAPTER="${INITIAL_BOB_ADAPTER:-${REPO_ROOT}/runs/embodied_selfplay_grpo16/round-0001/bob_adapter}"
  BOOTSTRAP_ALICE_RUN="${BOOTSTRAP_ALICE_RUN:-${REPO_ROOT}/runs/embodied_selfplay_grpo16}"
fi

# Holdout evaluation still uses one temporary Bob server after each round.
# 每轮结束后的 holdout 评估仍会临时起一个 Bob server（用完即停）。
BOB_ROLLOUT_GPU="${BOB_ROLLOUT_GPU:-0}"
BOB_PORT="${BOB_PORT:-8121}"
ALICE_PORT="${ALICE_PORT:-8122}"

# ---------------------------------------------------------------------------
# 7. Fail-fast validation
# ---------------------------------------------------------------------------
# 前置校验：任何一项不满足都 exit 2，宁可不启动也不带病跑。
# 1) staged 模式硬性要求恰好 8 张训练卡（每卡一个 engine + 一个 DDP rank）；
IFS=',' read -r -a OPENETA_GPU_LIST <<<"${TRAIN_GPU}"
if [[ "${#OPENETA_GPU_LIST[@]}" -ne 8 ]]; then
  echo "TRAIN_GPU must contain exactly eight comma-separated GPU IDs" >&2
  exit 2
fi
# 2) 非冷启动策略必须是真实存在的 LoRA；__base_model__ 明确表示无 adapter。
if [[ "${INITIAL_ALICE_ADAPTER}" != "${OPENETA_BASE_MODEL_POLICY}" && \
      ! -f "${INITIAL_ALICE_ADAPTER}/adapter_config.json" ]]; then
  echo "Alice adapter is missing: ${INITIAL_ALICE_ADAPTER}" >&2
  exit 2
fi
if [[ "${INITIAL_BOB_ADAPTER}" != "${OPENETA_BASE_MODEL_POLICY}" && \
      ! -f "${INITIAL_BOB_ADAPTER}/adapter_config.json" ]]; then
  echo "Bob adapter is missing: ${INITIAL_BOB_ADAPTER}" >&2
  exit 2
fi
# 3) bootstrap 历史 run 目录必须存在（Bob 数据集兜底来源）；
if [[ ! -d "${BOOTSTRAP_ALICE_RUN}" ]]; then
  echo "Bootstrap Alice run is missing: ${BOOTSTRAP_ALICE_RUN}" >&2
  exit 2
fi
# 4) 强制 TileLang 时 overlay 目录必须就绪（先跑 setup_fla_tilelang_overlay.sh）；
if [[ "${OPENETA_REQUIRE_TILELANG,,}" =~ ^(1|true|yes|on)$ && \
      ! -d "${OPENETA_TILELANG_OVERLAY}" ]]; then
  echo "TileLang overlay is missing: ${OPENETA_TILELANG_OVERLAY}" >&2
  echo "Run scripts/setup_fla_tilelang_overlay.sh before formal training." >&2
  exit 2
fi
# 5) 轮数和单条轨迹长度必须为正；
if [[ "${ROUNDS}" -lt 1 || "${MAX_TURNS}" -lt 1 ]]; then
  echo "ROUNDS and MAX_TURNS must be positive" >&2
  exit 2
fi
# 6) GRPO batch 必须自洽：generation_batch_size == per_device_batch * 8卡
#    * steps_per_generation，否则 Swift 的 rollout/训练 step 对不上。
EXPECTED_GENERATION_BATCH_SIZE=$((
  PER_DEVICE_TRAIN_BATCH_SIZE * 8 * STEPS_PER_GENERATION
))
if [[ "${GENERATION_BATCH_SIZE}" -ne "${EXPECTED_GENERATION_BATCH_SIZE}" ]]; then
  echo "invalid GRPO batch settings: GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE}, but" >&2
  echo "PER_DEVICE_TRAIN_BATCH_SIZE * world_size * STEPS_PER_GENERATION = ${EXPECTED_GENERATION_BATCH_SIZE}" >&2
  exit 2
fi

# 打印解析后的最终配置，便于在日志中确认本次实验的真实参数。
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
echo "  token_budget:   ${OPENETA_THINKING_MAX_TOKENS}/turn, ${OPENETA_TRAJECTORY_MAX_TOKENS}/trajectory"
echo "  context_limit:  ${OPENETA_CONTEXT_MAX_TOKENS}"
echo "  history_mode:   ${OPENETA_HISTORY_MODE}"
echo "  generations:    ${NUM_GENERATIONS}"
echo "  alice_adapter:  ${INITIAL_ALICE_ADAPTER}"
echo "  bob_adapter:    ${INITIAL_BOB_ADAPTER}"

# DRY_RUN=true 时只做校验和打印，不启动训练。
if [[ "${DRY_RUN,,}" =~ ^(1|true|yes|on)$ ]]; then
  echo "Dry run complete; training was not started."
  exit 0
fi

mkdir -p "${RUN_ROOT}"

# 把全部解析后的参数 export 给下游脚本继承。
export RUN_ROOT ROUNDS TASKS_PER_ROUND
export OPENETA_COLD_START OPENETA_BASE_MODEL_POLICY SWIFT_VENV
export REPORT_TO TENSORBOARD_LOGGING_DIR
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
export OPENETA_HISTORY_MODE OPENETA_TRAJECTORY_MAX_TOKENS OPENETA_CONTEXT_MAX_TOKENS
export MAX_COMPLETION_LENGTH MAX_LENGTH VLLM_MAX_MODEL_LEN MAX_PIXELS
export VLLM_MAX_NUM_SEQS VLLM_GPU_MEMORY_UTILIZATION
export VLLM_KV_CACHE_MEMORY_BYTES VLLM_SLEEP_LEVEL VLLM_ENFORCE_EAGER
export USE_LIGER_KERNEL OPENETA_KEEP_LOGITS_BF16
export LR LR_SCHEDULER_TYPE TEMPERATURE TOP_P BETA EPSILON
export LORA_RANK LORA_ALPHA MODEL_PATH
export INITIAL_ALICE_ADAPTER INITIAL_BOB_ADAPTER BOOTSTRAP_ALICE_RUN
export BOB_ROLLOUT_GPU BOB_PORT ALICE_PORT

# exec 用主循环脚本替换当前进程（同一 PID，不保留本脚本栈）。
# 真正的多轮编排（建数据集 -> 训 Alice -> 构 Bob 数据集 -> 训 Bob ->
# holdout 评估 -> 写 summary.json）在 run_embodied_swift_formal10.sh 中。
exec "${REPO_ROOT}/scripts/run_embodied_swift_formal10.sh"
