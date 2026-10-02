#!/usr/bin/env bash
# Formal asymmetric self-play experiment. Server mode and the original
# colocate mode remain available. OPENETA_STAGED_COLOCATE=true selects the
# three-phase, eight-rank Alice -> frozen Bob -> GRPO path without rollout
# servers during training.
#
# 中文总览
# --------
# 这个脚本负责完整的 Alice/Bob 非对称 self-play 实验编排，本身不实现
# ManiSkill 动作或 GRPO loss。核心流程是：
#   1. 准备固定的 bootstrap/holdout 数据集；
#   2. 每轮让 Alice 在环境中真实操作，创造新的方块目标位置；
#   3. 用当前冻结的 Bob 对 Alice 的有效题目做难度评估，并训练 Alice；
#   4. 把 Alice 的有效题目整理成 Bob 数据集，再训练 Bob；
#   5. 用固定 holdout 评估新 Bob，写出 summary.json；
#   6. 将本轮 Alice/Bob checkpoint 传给下一轮。
#
# 具体的多轮环境、动作解析、reward 和 token loss mask 位于：
#   plugins/embodied_swift_grpo.py
# 真正拼接 `swift rollout` / `swift rlhf` 命令的脚本位于：
#   scripts/run_embodied_swift_grpo.sh
set -euo pipefail

# SAPIEN's native Vulkan destructor aborts after a device-lost exception and
# otherwise writes multi-gigabyte core files into the repository.  The Python
# traceback and launcher logs contain the actionable failure information.
ulimit -c 0 2>/dev/null || true

# `-e`：任一未处理命令失败时退出；`-u`：使用未定义变量时报错；
# `pipefail`：管道中任意一段失败都算整个管道失败。这样可避免训练失败后
# 脚本仍误写 summary 或进入下一轮。

# 本实验的 vLLM/训练客户端都通过 127.0.0.1 通信。运行环境可能设置
# http(s)_proxy；若不排除环回地址，健康检查和 rollout 请求会被错误地
# 转发到代理端口并一直等待，看起来像 vLLM 卡死。
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

# Some cluster nodes install NVIDIA's Vulkan ICD under /etc instead of the
# /usr/share path used by SAPIEN's discovery helper.  Other nodes expose a
# zero-byte placeholder at the same path; forcing that placeholder makes
# Vulkan fail with ErrorIncompatibleDriver even though SAPIEN's own discovery
# works.  Respect an operator-provided value.  For automatic selection, only
# use a non-empty manifest that actually declares an ICD library.
if [[ -z "${VK_ICD_FILENAMES:-}" && \
      -s /etc/vulkan/icd.d/nvidia_icd.json ]] && \
   grep -q '"library_path"' /etc/vulkan/icd.d/nvidia_icd.json; then
  export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
fi

# ---------------------------------------------------------------------------
# 一、路径与实验规模
# ---------------------------------------------------------------------------
# `${VAR:-default}` 表示：VAR 未设置或为空时使用 default。启动命令传入的
# 环境变量会覆盖这里的默认值，所以判断某次已运行实验时还应查看 args.json
# 或进程环境，而不能只看本文件的默认值。
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWIFT_VENV="${SWIFT_VENV:-${REPO_ROOT}/../.venv_selfplay_embodied}"
PYTHON="${SWIFT_VENV}/bin/python"
# 所有数据集、轨迹、checkpoint 和 summary 都写到 RUN_ROOT。
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/swift_formal10}"
# 总 self-play 轮数。脚本名是 formal10，但可在启动时覆盖成 5 等其他值。
ROUNDS="${ROUNDS:-10}"

# ---------------------------------------------------------------------------
# 二、GPU、vLLM 服务与端口
# ---------------------------------------------------------------------------
# ROLLOUT_GPU 是兼容旧配置的公共默认卡；正式实验通常分别指定 Bob/Alice 卡。
ROLLOUT_GPU="${ROLLOUT_GPU:-0}"
# 默认保持向后兼容：两个 rollout 服务共用 ROLLOUT_GPU。
# 显存允许时可分别设置 BOB_ROLLOUT_GPU/ALICE_ROLLOUT_GPU，避免两个
# 独立 vLLM EngineCore 同驻一张卡时互相阻塞健康检查。
BOB_ROLLOUT_GPU="${BOB_ROLLOUT_GPU:-${ROLLOUT_GPU}}"
ALICE_ROLLOUT_GPU="${ALICE_ROLLOUT_GPU:-${ROLLOUT_GPU}}"
# TRAIN_GPU 可写成 `0,1,...,7`，下层脚本会据此启动对应数量的 DDP rank。
TRAIN_GPU="${TRAIN_GPU:-1}"
# Dataset/snapshot creation is single-process.  Restrict it to one of the
# training GPUs so torch fork_rng does not initialize all eight devices before
# distributed training starts.  This also makes SAPIEN's render-device choice
# deterministic across heterogeneous cluster nodes.
DATASET_GPU="${DATASET_GPU:-${TRAIN_GPU%%,*}}"
# server：独立 vLLM 服务；colocate：rollout 引擎与训练进程同卡管理。
VLLM_MODE="${VLLM_MODE:-server}"
# staged colocate 仍使用 Swift 的 colocate mode；这个额外开关决定是否启用
# 同一 rank/engine 内 Alice LoRA(id=1) -> 冻结 Bob LoRA(id=2) 的阶段调度。
OPENETA_STAGED_COLOCATE="${OPENETA_STAGED_COLOCATE:-false}"
BOB_PORT="${BOB_PORT:-8121}"
ALICE_PORT="${ALICE_PORT:-8122}"

if [[ "${OPENETA_STAGED_COLOCATE,,}" =~ ^(1|true|yes|on)$ ]]; then
  STAGED_COLOCATE=true
  if [[ "${VLLM_MODE}" != "colocate" ]]; then
    echo "OPENETA_STAGED_COLOCATE=true requires VLLM_MODE=colocate" >&2
    exit 2
  fi
else
  STAGED_COLOCATE=false
fi

if [[ "${STAGED_COLOCATE}" == "true" ]]; then
  TRAIN_COMMAND=(./scripts/run_embodied_swift_staged_colocate.sh)
else
  TRAIN_COMMAND=(./scripts/run_embodied_swift_grpo.sh train)
fi

# ---------------------------------------------------------------------------
# 三、环境轨迹长度与 GRPO batch
# ---------------------------------------------------------------------------
# 单条 Alice/Bob episode 最多与 ManiSkill 交互多少轮。模型可提前 DONE，
# 所以轨迹可以少于 MAX_TURNS；实际发生的所有 turn 都会进入训练。
if [[ "${STAGED_COLOCATE}" == "true" ]]; then
  DEFAULT_MAX_TURNS=32
else
  DEFAULT_MAX_TURNS=24
fi
MAX_TURNS="${MAX_TURNS:-${DEFAULT_MAX_TURNS}}"
# 每轮创建多少个不同初始 snapshot/prompt。
TASKS_PER_ROUND="${TASKS_PER_ROUND:-4}"
# One GRPO group contains eight independent trajectories for the same prompt.
# 同一个 prompt 独立采样多少条 completion，用于计算组内相对优势。
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
# Negative max_steps lets Hugging Face/Swift honor num_train_epochs instead of
# stopping after the old single optimizer step.
# Bash 中 `${TRAIN_STEPS:--1}` 的 `:-` 是默认值运算符，最后的 `-1` 才是
# 默认值。Swift/Hugging Face 把 max_steps=-1 解释为“不用固定 step 覆盖
# epoch 设置”，不是训练负一步，也不是无限训练。
TRAIN_STEPS="${TRAIN_STEPS:--1}" # 不使用固定 step 数提前终止
TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
# 每张卡的 micro batch 在下层固定为 1；这里控制累积多少个 micro step
# 再执行一次 optimizer update。
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
# 一次 generation batch 中的总 completion 数，通常与 NUM_GENERATIONS 相同。
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-${NUM_GENERATIONS}}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
# 每隔多少 trainer micro step 重新 rollout。设为 1 表示每个 micro step
# 都生成新的 on-policy group，不复用上一组 completion。
if [[ "${STAGED_COLOCATE}" == "true" ]]; then
  STAGED_GENERATION_DENOMINATOR=$((PER_DEVICE_TRAIN_BATCH_SIZE * 8))
  if (( GENERATION_BATCH_SIZE % STAGED_GENERATION_DENOMINATOR != 0 )); then
    echo "GENERATION_BATCH_SIZE must be divisible by per-device batch * 8 ranks" >&2
    exit 2
  fi
  DEFAULT_STEPS_PER_GENERATION=$((
    GENERATION_BATCH_SIZE / STAGED_GENERATION_DENOMINATOR
  ))
else
  DEFAULT_STEPS_PER_GENERATION="${GENERATION_BATCH_SIZE}"
fi
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-${DEFAULT_STEPS_PER_GENERATION}}"
if [[ "${STAGED_COLOCATE}" == "true" ]] &&
   (( GENERATION_BATCH_SIZE != PER_DEVICE_TRAIN_BATCH_SIZE * 8 * STEPS_PER_GENERATION )); then
  echo "invalid staged GRPO batch settings: generation_batch_size must equal" >&2
  echo "per_device_train_batch_size * 8 * steps_per_generation" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# 四、思考模式、动作格式与长序列显存保护
# ---------------------------------------------------------------------------
OPENETA_ENABLE_THINKING="${OPENETA_ENABLE_THINKING:-true}"
# thinking 开启时，scheduler 会用这个值覆盖 rollout CLI 的短默认上限；
# 它是“每个环境 turn”的最大生成 token 数。
OPENETA_THINKING_MAX_TOKENS="${OPENETA_THINKING_MAX_TOKENS:-1024}"
OPENETA_HISTORY_MODE="${OPENETA_HISTORY_MODE:-full}"
# 输出不是严格的 `</think>\n合法动作` 时单独扣除的最大格式罚分。
OPENETA_FORMAT_PENALTY="${OPENETA_FORMAT_PENALTY:-0.10}"
# Avoid Accelerate materializing a second, fp32 copy of sequence-sized logits
# during GRPO's no-grad old/reference-policy passes.  The plugin still scores
# every completion token and performs the selected log-softmax in row chunks.
OPENETA_KEEP_LOGITS_BF16="${OPENETA_KEEP_LOGITS_BF16:-true}"

# ---------------------------------------------------------------------------
# 五、嵌套 Bob 评估和 vLLM 超时
# ---------------------------------------------------------------------------
# A valid Alice proposal is evaluated by two independent, up-to-32-turn Bob
# episodes.  With thinking enabled those episodes can legitimately take well
# over the scheduler's 300 second library default.  Treating that latency as
# an evaluator outage replaces the adversarial boundary reward with a neutral
# reward, so the formal run uses a one-hour request budget by default.
OPENETA_BOB_EVALUATOR_TIMEOUT="${OPENETA_BOB_EVALUATOR_TIMEOUT:-3600}"
OPENETA_BOB_EVALUATIONS="${OPENETA_BOB_EVALUATIONS:-2}"
# The GRPO client waits on the Alice rollout server, and that server may in
# turn spend the full Bob-evaluator budget before returning the trajectory.
VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT:-4200}"

# ---------------------------------------------------------------------------
# 六、上下文长度与 vLLM KV cache
# ---------------------------------------------------------------------------
# Worst case is MAX_TURNS x OPENETA_THINKING_MAX_TOKENS generated tokens.
# MAX_COMPLETION_LENGTH 是整条多轮 assistant completion 的总上限；
# MAX_LENGTH 还要容纳 system/user prompt、图像 token 和环境反馈。
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-24576}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
OPENETA_TRAJECTORY_MAX_TOKENS="${OPENETA_TRAJECTORY_MAX_TOKENS:-${MAX_COMPLETION_LENGTH}}"
OPENETA_CONTEXT_MAX_TOKENS="${OPENETA_CONTEXT_MAX_TOKENS:-${MAX_LENGTH}}"
export OPENETA_HISTORY_MODE OPENETA_TRAJECTORY_MAX_TOKENS OPENETA_CONTEXT_MAX_TOKENS
# 每个 vLLM engine 同时处理的序列数。设为 1 可降低长序列峰值显存。
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.28}"
# 使用固定 KV cache 字节数，比单纯按显存比例更容易控制多进程共卡峰值。
VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-4294967296}"

# 创建 JSONL 数据集时显式记录是否启用 thinking，保证数据配置与 rollout
# scheduler 一致。
if [[ "${OPENETA_ENABLE_THINKING,,}" =~ ^(1|true|yes|on)$ ]]; then
  THINKING_DATASET_FLAG="--enable-thinking"
else
  THINKING_DATASET_FLAG="--no-enable-thinking"
fi

# ---------------------------------------------------------------------------
# 七、初始 adapter 与公共数据路径
# ---------------------------------------------------------------------------
# Round 0 从已有 Alice/Bob LoRA adapter 起步；后续 round 会自动使用上一轮
# 新生成的 checkpoint。
INITIAL_ALICE_ADAPTER="${INITIAL_ALICE_ADAPTER:-${REPO_ROOT}/runs/alice_dagger_curriculum/round-0007/alice_adapter}"
INITIAL_BOB_ADAPTER="${INITIAL_BOB_ADAPTER:-${REPO_ROOT}/runs/embodied_selfplay_grpo16/round-0001/bob_adapter}"
OPENETA_BASE_MODEL_POLICY="${OPENETA_BASE_MODEL_POLICY:-__base_model__}"
# 当本轮 Alice 没有足够有效题目时，Bob 数据集会回退到这个历史 run。

# ***** 这里需要修改 ****************************
BOOTSTRAP_ALICE_RUN="${BOOTSTRAP_ALICE_RUN:-${REPO_ROOT}/runs/embodied_selfplay_grpo16}"
DATASET_DIR="${RUN_ROOT}/datasets"
LOG_DIR="${RUN_ROOT}/logs"
BOOTSTRAP_BOB_DATASET="${DATASET_DIR}/bob_bootstrap_train.jsonl"
BOB_HOLDOUT_DATASET="${DATASET_DIR}/bob_holdout.jsonl"
ALICE_FIXED_DATASET="${DATASET_DIR}/alice_fixed.jsonl"

# 创建顶层数据与日志目录。各 round 子目录在主循环中按需创建。
mkdir -p "${DATASET_DIR}" "${LOG_DIR}"
# 这些变量保存当前外部 vLLM 服务的进程组 PID 和真实监听端口。
BOB_SERVER_PID=""
ALICE_SERVER_PID=""
BOB_SERVER_PORT=""
ALICE_SERVER_PORT=""

# ---------------------------------------------------------------------------
# 八、vLLM 服务生命周期管理
# ---------------------------------------------------------------------------
# 停止整个服务进程组，而不只是最外层 swift 进程。这样 Uvicorn/EngineCore
# 不会变成孤儿进程继续占用 GPU 和端口。
stop_server() {
  local pid="$1"
  local attempt
  if [[ -z "${pid}" ]] || ! kill -0 -- "-${pid}" 2>/dev/null; then
    return 0
  fi
  kill -- "-${pid}" 2>/dev/null || true
  for attempt in $(seq 1 50); do
    # Swift rollout 会再派生 Uvicorn 和 EngineCore。只等待外层 pid 会在
    # 子进程仍占用端口时过早返回，随后新服务被 find_free_port 改到下一端口。
    kill -0 -- "-${pid}" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 -- "-${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

# 无论脚本正常结束、报错、收到 Ctrl-C 还是 TERM，都尽量回收两个服务。
cleanup() {
  # start_server 尚在等待健康检查时，角色 PID 还没来得及回填；此时也要
  # 清理由 STARTED_PID 记录的启动中进程组。
  stop_server "${STARTED_PID:-}"
  stop_server "${ALICE_SERVER_PID}"
  stop_server "${BOB_SERVER_PID}"
}
trap cleanup EXIT INT TERM

# 轮询 /health/，直到服务可用。Swift 可能因端口仍被占用而自动选择下一个
# 空闲端口，因此这里从 Uvicorn 日志提取“真实端口”，而不是盲信请求端口。
wait_for_server() {
  local requested_port="$1"
  local pid="$2"
  local label="$3"
  local log_file="$4"
  local attempt
  local actual_port=""
  for attempt in $(seq 1 300); do
    # DeployArguments.__post_init__ 会调用 find_free_port。刚关闭的端口可能
    # 仍处于 TIME_WAIT，因此 Swift 即使命令行收到 8121，也可能最终监听
    # 8122/8123。以 Uvicorn 的实际监听日志为准，不能继续假定请求端口。
    actual_port="$(sed -nE \
      's/.*Uvicorn running on http:\/\/127\.0\.0\.1:([0-9]+).*/\1/p' \
      "${log_file}" 2>/dev/null | tail -1)"
    if [[ -n "${actual_port}" ]] && \
       curl --connect-timeout 2 --max-time 5 -fsS \
         "http://127.0.0.1:${actual_port}/health/" >/dev/null 2>&1; then
      STARTED_PORT="${actual_port}"
      echo "${label} ready on port ${actual_port} (requested ${requested_port})"
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "${label} exited before health check passed" >&2
      return 1
    fi
    sleep 2
  done
  echo "timed out waiting for ${label}" >&2
  return 1
}

# 启动指定角色的 vLLM 多轮 rollout server。
# 参数依次是：role、期望端口、LoRA adapter、轨迹输出目录、日志、GPU。
# `setsid` 创建独立进程组，便于 stop_server 一次回收所有子进程。
start_server() {
  local role="$1"
  local port="$2"
  local adapter="$3"
  local output_dir="$4"
  local log_file="$5"
  local rollout_gpu="$6"
  local bob_evaluator_port="${BOB_SERVER_PORT:-${BOB_PORT}}"
  (
    cd "${REPO_ROOT}"
    exec setsid env \
      ROLE="${role}" \
      ROLLOUT_GPU="${rollout_gpu}" \
      ROLLOUT_PORT="${port}" \
      MAX_TURNS="${MAX_TURNS}" \
      MAX_PIXELS="${MAX_PIXELS:-4096}" \
      VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" \
      VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
      VLLM_GPU_MEMORY_UTILIZATION="${TRAIN_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION}}" \
      VLLM_KV_CACHE_MEMORY_BYTES="${TRAIN_VLLM_KV_CACHE_MEMORY_BYTES:-${VLLM_KV_CACHE_MEMORY_BYTES}}" \
      VLLM_MM_PROCESSOR_CACHE_GB=0 \
      ROLLOUT_ADAPTER_PATH="${adapter}" \
      OUTPUT_DIR="${output_dir}" \
      OPENETA_BOB_EVALUATOR_URL="http://127.0.0.1:${bob_evaluator_port}" \
      OPENETA_BOB_EVALUATIONS="${OPENETA_BOB_EVALUATIONS}" \
      OPENETA_BOB_MAX_STEPS="${MAX_TURNS}" \
      OPENETA_BOB_EVALUATOR_TIMEOUT="${OPENETA_BOB_EVALUATOR_TIMEOUT}" \
      OPENETA_BOB_POLICY_VERSION="${CURRENT_BOB_ADAPTER}" \
      OPENETA_ENABLE_THINKING="${OPENETA_ENABLE_THINKING}" \
      OPENETA_THINKING_MAX_TOKENS="${OPENETA_THINKING_MAX_TOKENS}" \
      OPENETA_FORMAT_PENALTY="${OPENETA_FORMAT_PENALTY}" \
      ./scripts/run_embodied_swift_grpo.sh rollout
  ) >"${log_file}" 2>&1 &
  STARTED_PID=$!
  STARTED_PORT=""
  wait_for_server "${port}" "${STARTED_PID}" "${role}-server" "${log_file}"
}

# Swift 每个 save step 会创建 checkpoint-N；按自然版本顺序取编号最大的一个。
latest_checkpoint() {
  find "$1" -type d -name 'checkpoint-*' | sort -V | tail -1
}

# ---------------------------------------------------------------------------
# 九、启动前检查与一次性固定数据集
# ---------------------------------------------------------------------------
# 正式启动前要求两个默认端口都空闲，避免误连到旧实验遗留的服务。
if curl --connect-timeout 2 --max-time 5 -fsS \
     "http://127.0.0.1:${BOB_PORT}/health/" >/dev/null 2>&1 || \
   curl --connect-timeout 2 --max-time 5 -fsS \
     "http://127.0.0.1:${ALICE_PORT}/health/" >/dev/null 2>&1; then
  echo "ports ${BOB_PORT}/${ALICE_PORT} must be free before starting" >&2
  exit 2
fi

# bootstrap train、固定 holdout 和 Alice 固定集只在文件不存在/为空时创建。
# 重启脚本会复用它们，从而保证跨 round、跨恢复运行的评测口径一致。
if [[ ! -s "${BOOTSTRAP_BOB_DATASET}" ]]; then
  CUDA_VISIBLE_DEVICES="${DATASET_GPU}" \
  "${PYTHON}" "${REPO_ROOT}/scripts/create_embodied_swift_dataset.py" \
    --role bob --source-run "${BOOTSTRAP_ALICE_RUN}" --output "${BOOTSTRAP_BOB_DATASET}" \
    --count 4 --skip 0 --max-steps "${MAX_TURNS}" --camera-resolution 64 \
    "${THINKING_DATASET_FLAG}"
fi
if [[ ! -s "${BOB_HOLDOUT_DATASET}" ]]; then
  CUDA_VISIBLE_DEVICES="${DATASET_GPU}" \
  "${PYTHON}" "${REPO_ROOT}/scripts/create_embodied_swift_dataset.py" \
    --role bob --source-run "${BOOTSTRAP_ALICE_RUN}" --output "${BOB_HOLDOUT_DATASET}" \
    --count 4 --skip 4 --max-steps "${MAX_TURNS}" --camera-resolution 64 \
    "${THINKING_DATASET_FLAG}"
fi
if [[ ! -s "${ALICE_FIXED_DATASET}" ]]; then
  CUDA_VISIBLE_DEVICES="${DATASET_GPU}" \
  "${PYTHON}" "${REPO_ROOT}/scripts/create_embodied_swift_dataset.py" \
    --role alice --output "${ALICE_FIXED_DATASET}" --count 4 --seed-start 30000 \
    --max-steps "${MAX_TURNS}" --camera-resolution 64 "${THINKING_DATASET_FLAG}"
fi

# Round 0 使用初始 adapter；每轮结束后这两个变量会更新为新 checkpoint。
CURRENT_ALICE_ADAPTER="${INITIAL_ALICE_ADAPTER}"
CURRENT_BOB_ADAPTER="${INITIAL_BOB_ADAPTER}"

# ---------------------------------------------------------------------------
# 十、正式 Alice/Bob self-play 主循环
# ---------------------------------------------------------------------------
for round_index in $(seq 0 $((ROUNDS - 1))); do
  round_name="$(printf 'round-%04d' "${round_index}")"
  round_dir="${RUN_ROOT}/${round_name}"
  mkdir -p "${round_dir}"

  # 断点续跑：summary.json 只在该 round 的 Alice 训练、Bob 训练和 holdout
  # 全部完成后写入。因此存在非空 summary 即可安全跳过整轮，并从其中恢复
  # 两个角色的新 checkpoint 路径。
  if [[ -s "${round_dir}/summary.json" ]]; then
    CURRENT_ALICE_ADAPTER="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"]["alice"])' "${round_dir}/summary.json")"
    CURRENT_BOB_ADAPTER="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoints"]["bob"])' "${round_dir}/summary.json")"
    echo "${round_name} already complete; resuming after it"
    continue
  fi
  echo "===== ${round_name}/${ROUNDS} ====="

  # -------------------------------------------------------------------------
  # 10.1 为本轮 Alice 创建 TASKS_PER_ROUND 个初始环境 snapshot
  # -------------------------------------------------------------------------
  # 每轮使用不同 seed 区间；Alice 的任务不是去现有 goal，而是通过真实动作
  # 创造一个新的、可达且尽量困难的方块目标状态。
  alice_dataset="${round_dir}/alice_tasks.jsonl"
  if [[ ! -s "${alice_dataset}" ]]; then
    CUDA_VISIBLE_DEVICES="${DATASET_GPU}" \
    "${PYTHON}" "${REPO_ROOT}/scripts/create_embodied_swift_dataset.py" \
      --role alice --output "${alice_dataset}" --count "${TASKS_PER_ROUND}" \
      --seed-start $((31000 + round_index * TASKS_PER_ROUND)) \
      --max-steps "${MAX_TURNS}" --camera-resolution 64 "${THINKING_DATASET_FLAG}"
  fi

  IFS=$'\t' read -r alice_checkpoint alice_complete < <(
    "${PYTHON}" "${REPO_ROOT}/scripts/inspect_embodied_checkpoint.py" "${round_dir}/alice_train"
  )
  if [[ "${alice_checkpoint}" == "-" ]]; then
    alice_checkpoint=""
  fi

  # -------------------------------------------------------------------------
  # 10.2 启动需要的 rollout 服务
  # -------------------------------------------------------------------------
  # Alice 的 reward 需要当前 Bob 作为冻结边界：每个有效 Alice proposal 会
  # 交给 Bob 独立尝试两次。训练 Alice 时绝不更新这个 Bob adapter。
  # staged colocate 在 8 个训练 rank 的本地 engine 内顺序执行 Alice/Bob，
  # 因此训练阶段不启动任何外部 server。普通 colocate 仍只需外部 Bob；
  # server 模式则为 Alice 和 Bob 各启动一个服务。
  if [[ "${alice_complete}" != "1" && "${STAGED_COLOCATE}" != "true" ]]; then
    start_server bob "${BOB_PORT}" "${CURRENT_BOB_ADAPTER}" \
      "${round_dir}/bob_evaluator_rollout" "${LOG_DIR}/${round_name}-bob-server.log" \
      "${BOB_ROLLOUT_GPU}"
    BOB_SERVER_PID="${STARTED_PID}"
    BOB_SERVER_PORT="${STARTED_PORT}"
    if [[ "${VLLM_MODE}" == "server" ]]; then
      start_server alice "${ALICE_PORT}" "${CURRENT_ALICE_ADAPTER}" \
        "${round_dir}/alice_rollout" "${LOG_DIR}/${round_name}-alice-server.log" \
        "${ALICE_ROLLOUT_GPU}"
      ALICE_SERVER_PID="${STARTED_PID}"
      ALICE_SERVER_PORT="${STARTED_PORT}"
    fi
  fi

  # -------------------------------------------------------------------------
  # 10.3 训练 Alice
  # -------------------------------------------------------------------------
  # 下层脚本会启动 `swift rlhf --rlhf_type grpo`。Alice 的 scheduler 在
  # ManiSkill 中执行最多 MAX_TURNS 个真实动作，编译有效 cube_at_position
  # goal，再由外部 evaluator 或 staged 本地冻结 Bob 做两次复现评估。所有
  # rollout artifact 写入 alice_rollout/rollouts，训练 checkpoint 写入 alice_train。
  if [[ "${alice_complete}" == "1" ]]; then
    NEW_ALICE_ADAPTER="${alice_checkpoint}"
    echo "${round_name} Alice completed at ${NEW_ALICE_ADAPTER}; skipping retraining"
  else
  if [[ -n "${alice_checkpoint}" ]]; then
    echo "${round_name} resuming Alice from ${alice_checkpoint}"
  fi
  ALICE_TRAIN_ADAPTER="${CURRENT_ALICE_ADAPTER}"
  FROZEN_BOB_ADAPTER="${CURRENT_BOB_ADAPTER}"
  FROZEN_BOB_BASE=false
  if [[ "${ALICE_TRAIN_ADAPTER}" == "${OPENETA_BASE_MODEL_POLICY}" ]]; then
    ALICE_TRAIN_ADAPTER=""
  fi
  if [[ "${FROZEN_BOB_ADAPTER}" == "${OPENETA_BASE_MODEL_POLICY}" ]]; then
    FROZEN_BOB_ADAPTER=""
    FROZEN_BOB_BASE=true
  fi
  (
    cd "${REPO_ROOT}"
    ROLE=alice TRAIN_GPU="${TRAIN_GPU}" CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
      NPROC_PER_NODE=8 VLLM_MODE="${VLLM_MODE}" \
      ROLLOUT_PORT="${ALICE_SERVER_PORT:-${ALICE_PORT}}" MAX_TURNS="${MAX_TURNS}" \
      DATASET_PATH="${alice_dataset}" OUTPUT_DIR="${round_dir}/alice_train" \
      OPENETA_SWIFT_ARTIFACT_ROOT="${round_dir}/alice_rollout/rollouts" \
      OPENETA_STAGED_ARTIFACT_ROOT="${round_dir}" OPENETA_ROUND_ID="${round_name}" \
      OPENETA_ALICE_POLICY_VERSION="${CURRENT_ALICE_ADAPTER}" \
      OPENETA_FROZEN_BOB_ADAPTER="${FROZEN_BOB_ADAPTER}" \
      OPENETA_FROZEN_BOB_BASE="${FROZEN_BOB_BASE}" \
      TRAIN_ADAPTER_PATH="${ALICE_TRAIN_ADAPTER}" \
      RESUME_FROM_CHECKPOINT="${alice_checkpoint}" \
      OPENETA_BOB_EVALUATOR_URL="http://127.0.0.1:${BOB_SERVER_PORT:-${BOB_PORT}}" \
      OPENETA_BOB_EVALUATIONS="${OPENETA_BOB_EVALUATIONS}" OPENETA_BOB_MAX_STEPS="${MAX_TURNS}" \
      OPENETA_BOB_EVALUATOR_TIMEOUT="${OPENETA_BOB_EVALUATOR_TIMEOUT}" \
      OPENETA_BOB_POLICY_VERSION="${CURRENT_BOB_ADAPTER}" \
      OPENETA_ENABLE_THINKING="${OPENETA_ENABLE_THINKING}" \
      OPENETA_THINKING_MAX_TOKENS="${OPENETA_THINKING_MAX_TOKENS}" \
      MAX_PIXELS="${MAX_PIXELS:-4096}" MAX_LENGTH="${MAX_LENGTH}" MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH}" \
      VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
      VLLM_GPU_MEMORY_UTILIZATION="${TRAIN_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION}}" \
      VLLM_KV_CACHE_MEMORY_BYTES="${TRAIN_VLLM_KV_CACHE_MEMORY_BYTES:-${VLLM_KV_CACHE_MEMORY_BYTES}}" \
      TRAIN_STEPS="${TRAIN_STEPS}" TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
      GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
      NUM_GENERATIONS="${NUM_GENERATIONS}" \
      GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE}" STEPS_PER_GENERATION="${STEPS_PER_GENERATION}" \
      VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT}" \
      OPENETA_KEEP_LOGITS_BF16="${OPENETA_KEEP_LOGITS_BF16}" \
      OPENETA_FORMAT_PENALTY="${OPENETA_FORMAT_PENALTY}" \
      USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}" VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-2}" \
      TEMPERATURE="${TEMPERATURE:-1.0}" LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}" \
      "${TRAIN_COMMAND[@]}"
  ) 2>&1 | tee "${LOG_DIR}/${round_name}-alice-trainer.log"

  # Alice 训练必须至少产生一个 checkpoint，否则 `test` 会让脚本立即失败，
  # 不会拿旧 adapter 冒充本轮结果继续训练 Bob。
  IFS=$'\t' read -r NEW_ALICE_ADAPTER alice_complete < <(
    "${PYTHON}" "${REPO_ROOT}/scripts/inspect_embodied_checkpoint.py" "${round_dir}/alice_train"
  )
  [[ "${alice_complete}" == "1" ]] || { echo "Alice finished without a complete checkpoint" >&2; exit 1; }
  fi
  if [[ -n "${ALICE_SERVER_PID}" ]]; then
    stop_server "${ALICE_SERVER_PID}"
    ALICE_SERVER_PID=""
    ALICE_SERVER_PORT=""
  fi
  # colocate 模式下，训练 Bob 前停止冻结 evaluator，释放其 GPU 给 8 卡 DDP。
  # server 模式会继续复用外部 Bob 服务作为 Bob 自己的 rollout server。
  if [[ "${VLLM_MODE}" == "colocate" && -n "${BOB_SERVER_PID}" ]]; then
    stop_server "${BOB_SERVER_PID}"
    BOB_SERVER_PID=""
    BOB_SERVER_PORT=""
  fi

  # -------------------------------------------------------------------------
  # 10.4 从本轮有效 Alice proposal 构造 Bob 训练集
  # -------------------------------------------------------------------------
  # Bob 看到绿色 goal marker、instruction 和自己的 observation，目标是复现
  # Alice 创造的最终方块状态。如果本轮没有足够可用的 Alice 题目，生成器
  # 返回失败，脚本改用历史 bootstrap 数据，避免整个正式实验直接中断。
  bob_dataset="${round_dir}/bob_tasks.jsonl"
  bob_dataset_mode_file="${round_dir}/bob_dataset_mode.txt"
  if [[ -s "${bob_dataset}" ]]; then
    # Older runs did not write a provenance marker. A generated Bob dataset
    # already present must be preserved, including its frozen snapshots.
    if [[ -s "${bob_dataset_mode_file}" ]]; then
      bob_dataset_mode="$(<"${bob_dataset_mode_file}")"
    else
      bob_dataset_mode="new_alice"
    fi
  else
    if "${PYTHON}" "${REPO_ROOT}/scripts/create_embodied_swift_dataset.py" \
        --role bob --source-run "${round_dir}/alice_rollout" --output "${bob_dataset}" \
        --count "${TASKS_PER_ROUND}" --max-steps "${MAX_TURNS}" --camera-resolution 64 \
        "${THINKING_DATASET_FLAG}"; then
      bob_dataset_mode="new_alice"
    else
      "${PYTHON}" "${REPO_ROOT}/scripts/create_embodied_swift_dataset.py" \
        --role bob --source-run "${BOOTSTRAP_ALICE_RUN}" --output "${bob_dataset}" \
        --count 4 --skip 0 --max-steps "${MAX_TURNS}" --camera-resolution 64 \
        "${THINKING_DATASET_FLAG}"
      bob_dataset_mode="bootstrap_fallback"
    fi
    printf '%s\n' "${bob_dataset_mode}" > "${bob_dataset_mode_file}"
  fi

  IFS=$'\t' read -r bob_checkpoint bob_complete < <(
    "${PYTHON}" "${REPO_ROOT}/scripts/inspect_embodied_checkpoint.py" "${round_dir}/bob_train"
  )
  if [[ "${bob_checkpoint}" == "-" ]]; then
    bob_checkpoint=""
  fi

  # -------------------------------------------------------------------------
  # 10.5 训练 Bob
  # -------------------------------------------------------------------------
  # Bob 使用本轮开始时的 CURRENT_BOB_ADAPTER 作为训练起点；Alice 新题目只
  # 提供任务/奖励，不会把 Alice 的动作轨迹当作 Bob 的目标输出。Bob rollout
  # 与 checkpoint 分别写入 bob_rollout 和 bob_train。
  if [[ "${bob_complete}" == "1" ]]; then
    NEW_BOB_ADAPTER="${bob_checkpoint}"
    echo "${round_name} Bob completed at ${NEW_BOB_ADAPTER}; skipping retraining"
  else
  if [[ -n "${bob_checkpoint}" ]]; then
    echo "${round_name} resuming Bob from ${bob_checkpoint}"
  fi
  BOB_TRAIN_ADAPTER="${CURRENT_BOB_ADAPTER}"
  if [[ "${BOB_TRAIN_ADAPTER}" == "${OPENETA_BASE_MODEL_POLICY}" ]]; then
    BOB_TRAIN_ADAPTER=""
  fi
  (
    cd "${REPO_ROOT}"
    ROLE=bob TRAIN_GPU="${TRAIN_GPU}" CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
      NPROC_PER_NODE=8 VLLM_MODE="${VLLM_MODE}" \
      ROLLOUT_PORT="${BOB_SERVER_PORT:-${BOB_PORT}}" MAX_TURNS="${MAX_TURNS}" \
      DATASET_PATH="${bob_dataset}" OUTPUT_DIR="${round_dir}/bob_train" \
      OPENETA_SWIFT_ARTIFACT_ROOT="${round_dir}/bob_rollout/rollouts" \
      TRAIN_ADAPTER_PATH="${BOB_TRAIN_ADAPTER}" \
      RESUME_FROM_CHECKPOINT="${bob_checkpoint}" \
      OPENETA_ENABLE_THINKING="${OPENETA_ENABLE_THINKING}" \
      OPENETA_THINKING_MAX_TOKENS="${OPENETA_THINKING_MAX_TOKENS}" \
      MAX_PIXELS="${MAX_PIXELS:-4096}" MAX_LENGTH="${MAX_LENGTH}" MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH}" \
      VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}" VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
      VLLM_GPU_MEMORY_UTILIZATION="${TRAIN_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION}}" \
      VLLM_KV_CACHE_MEMORY_BYTES="${TRAIN_VLLM_KV_CACHE_MEMORY_BYTES:-${VLLM_KV_CACHE_MEMORY_BYTES}}" \
      TRAIN_STEPS="${TRAIN_STEPS}" TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
      GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
      NUM_GENERATIONS="${NUM_GENERATIONS}" \
      GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE}" STEPS_PER_GENERATION="${STEPS_PER_GENERATION}" \
      VLLM_SERVER_TIMEOUT="${VLLM_SERVER_TIMEOUT}" \
      OPENETA_KEEP_LOGITS_BF16="${OPENETA_KEEP_LOGITS_BF16}" \
      OPENETA_FORMAT_PENALTY="${OPENETA_FORMAT_PENALTY}" \
      USE_LIGER_KERNEL="${USE_LIGER_KERNEL:-false}" VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-2}" \
      TEMPERATURE="${TEMPERATURE:-1.0}" LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}" \
      "${TRAIN_COMMAND[@]}"
  ) 2>&1 | tee "${LOG_DIR}/${round_name}-bob-trainer.log"

  # 与 Alice 相同：必须找到本轮真实生成的 Bob checkpoint 才能继续。
  IFS=$'\t' read -r NEW_BOB_ADAPTER bob_complete < <(
    "${PYTHON}" "${REPO_ROOT}/scripts/inspect_embodied_checkpoint.py" "${round_dir}/bob_train"
  )
  [[ "${bob_complete}" == "1" ]] || { echo "Bob finished without a complete checkpoint" >&2; exit 1; }
  fi
  if [[ -n "${BOB_SERVER_PID}" ]]; then
    stop_server "${BOB_SERVER_PID}"
    BOB_SERVER_PID=""
    BOB_SERVER_PORT=""
  fi

  # -------------------------------------------------------------------------
  # 10.6 用固定 holdout 评估训练后的新 Bob
  # -------------------------------------------------------------------------
  # 重新以 NEW_BOB_ADAPTER 启动干净的 Bob server，temperature=0，每个固定
  # holdout task 评估一次。结果用于跨 round 比较 Bob 能力是否提升。
  if [[ ! -s "${round_dir}/bob_holdout_eval.json" ]]; then
  start_server bob "${BOB_PORT}" "${NEW_BOB_ADAPTER}" \
    "${round_dir}/bob_eval_rollout" "${LOG_DIR}/${round_name}-bob-eval-server.log" \
    "${BOB_ROLLOUT_GPU}"
  BOB_SERVER_PID="${STARTED_PID}"
  BOB_SERVER_PORT="${STARTED_PORT}"
  "${PYTHON}" "${REPO_ROOT}/scripts/evaluate_embodied_swift_server.py" \
    --dataset "${BOB_HOLDOUT_DATASET}" --url "http://127.0.0.1:${BOB_SERVER_PORT}" \
    --output "${round_dir}/bob_holdout_eval.json" --samples-per-task 1 \
    --temperature 0 --run-id "${round_name}-bob-holdout"
  stop_server "${BOB_SERVER_PID}"
  BOB_SERVER_PID=""
  BOB_SERVER_PORT=""
  fi

  # -------------------------------------------------------------------------
  # 10.7 汇总本轮，并把新 adapter 传给下一轮
  # -------------------------------------------------------------------------
  # summary.json 记录 Alice rollout 指标、Bob holdout、数据来源模式以及两个
  # checkpoint 路径。它同时是整轮完成标记和下次断点续跑的权威依据。
  "${PYTHON}" "${REPO_ROOT}/scripts/summarize_embodied_formal_round.py" \
    --round "${round_index}" --alice-rollouts "${round_dir}/alice_rollout" \
    --bob-eval "${round_dir}/bob_holdout_eval.json" --bob-dataset-mode "${bob_dataset_mode}" \
    --alice-adapter "${NEW_ALICE_ADAPTER}" --bob-adapter "${NEW_BOB_ADAPTER}" \
    --output "${round_dir}/summary.json"
  CURRENT_ALICE_ADAPTER="${NEW_ALICE_ADAPTER}"
  CURRENT_BOB_ADAPTER="${NEW_BOB_ADAPTER}"
done

# 只有所有 round 都成功写完 summary 并走出主循环，才会打印这条完成信息。
echo "formal ${ROUNDS}-round experiment completed: ${RUN_ROOT}"
