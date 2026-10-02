# 分阶段 Swift 实验：代码上传与跨平台运行

适用入口：`scripts/run_embodied_swift_staged_formal5.sh`。本说明只整理代码；历史轨迹、训练输出、模型权重和现有 Python 环境均不进入 Git 仓库。

## 代码调用链

```text
run_embodied_swift_staged_formal5.sh
  -> run_embodied_swift_formal10.sh         多轮编排、数据集、训练、评估与续跑
     -> create_embodied_swift_dataset.py   ManiSkill snapshot 和任务 JSONL
     -> run_embodied_swift_staged_colocate.sh
        -> run_embodied_staged_colocate.py  安装分阶段 Swift trainer
           -> agent/training/embodied_staged_colocate.py
        -> plugins/embodied_swift_grpo.py   rollout、reward、Alice/Bob scheduler
     -> run_embodied_swift_grpo.sh          每轮结束后启动 Bob holdout server
     -> evaluate_embodied_swift_server.py
     -> inspect_embodied_checkpoint.py
     -> summarize_embodied_formal_round.py
```

这些 Python 入口还使用 `adapter/maniskill_sim.py`、`adapter/maniskill_process.py`、`adapter/protocol.py`、`adapter/sim.py`、`agent/runtime/embodied_*.py` 和 `agent/training/embodied_grpo.py`。建议在现有 OpenETA 仓库分支中提交整个 `adapter/`、`agent/`、`scripts/`、`plugins/` 的相关代码，而不是只复制一个 shell 脚本。`sim/` 是仓库其他功能的一部分；上述入口的静态导入链没有直接依赖它。

这条 staged Swift 调用链没有导入相邻的 `adversarial-mle-agents`。旧入口 `scripts/run_embodied_selfplay.py` 直接导入其 `mle_agent_rl` 包；`scripts/train_alice_dagger.py` 和 `scripts/bootstrap_embodied_sft.py` 又导入旧入口，因此也间接依赖它。`scripts/run_alice_bob_curriculum.py` 仅在检测到相邻仓库时使用它。若只运行本页所述 staged Swift 实验，无需上传或安装 `adversarial-mle-agents`。

## 建议提交与排除

提交：入口及其调用的脚本、上述 Python 包、`configs/requirements-embodied-swift.txt`、本目录的说明和相关单元测试。staged 训练使用独立的需求文件；项目根目录的 `pyproject.toml` / `uv.lock` 用于其他 OpenETA 环境，不是这条 Swift 环境的安装锁。`SFT_data/` 属于另一条数据采集流程，不是此实验入口的运行依赖。

排除：`runs/`、`SFT_data/{raw,accepted,exported,reports,manifests}/`、虚拟环境、`.runtime/`、`assets/local/`、core dump、密钥和本机 `.env`。对应规则已加入 `.gitignore`。不要执行无差别的 `git add .`；先用 `git status --short` 和 `git diff --cached --stat` 检查暂存内容。

本地工作树已经含有大量尚未提交的相关代码。若从当前分支直接推送，必须把调用链里的新文件和现有已修改文件一并提交；只提交新入口会导致 GitHub 上缺少训练模块。现有仓库的 `origin` 指向 `OpenMOSS/OpenETA`，`fork` 指向 `yatao-zhuozhuo/embodied-adversarial`。

可以按下面的清单暂存本实验文件，再逐项检查 `git diff --cached`。已有仓库中未修改的公共模块会随原有提交保留。

```bash
git add .gitignore \
  configs/requirements-embodied-swift.txt \
  configs/embodied-swift-staged.example.env \
  adapter/maniskill_sim.py adapter/maniskill_process.py \
  agent/runtime/embodied_goal.py agent/runtime/embodied_snapshot.py \
  agent/runtime/embodied_selfplay_prompt.py agent/runtime/embodied_task_compiler.py \
  agent/training/embodied_grpo.py agent/training/embodied_staged_colocate.py \
  agent/training/embodied_staged_contracts.py agent/training/__init__.py \
  plugins/embodied_swift_grpo.py \
  scripts/run_embodied_swift_staged_formal5.sh \
  scripts/run_embodied_swift_formal10.sh \
  scripts/run_embodied_swift_staged_colocate.sh \
  scripts/run_embodied_staged_colocate.py \
  scripts/run_embodied_swift_grpo.sh \
  scripts/run_embodied_swift_staged_job.sh \
  scripts/create_embodied_swift_dataset.py \
  scripts/inspect_embodied_checkpoint.py \
  scripts/evaluate_embodied_swift_server.py \
  scripts/summarize_embodied_formal_round.py \
  scripts/setup_embodied_swift_env.sh scripts/setup_fla_tilelang_overlay.sh \
  scripts/probe_fla_tilelang.py scripts/probe_maniskill_isolated.py \
  tests/test_embodied_staged_contracts.py tests/test_maniskill_render_backend.py \
  docs/embodied-swift-staged-portability.zh-CN.md
git diff --cached --check
git diff --cached --stat
```

其他新文档、测试以及 `pyproject.toml` 的改动可在审阅后另行加入。

## Fork 并上传此分支

GitHub 仓库 `yatao-zhuozhuo/embodied-adversarial` 已从 `OpenMOSS/OpenETA` fork，且本地已添加 `fork` remote。当前分支是 `feat/embodied-asymmetric-self-play`。无需重新 clone；`origin` 保留为原项目，避免误推送。

```bash
cd OpenETA
git remote -v
git config user.name "<你的提交名称>"
git config user.email "<你的 GitHub 邮箱>"
# 先执行上面的逐项 git add 清单
git diff --cached --name-only
git diff --cached --check
git commit -m "Add portable staged embodied Swift training"
git push -u fork feat/embodied-asymmetric-self-play
```

推送当前分支会同时带上它相对原项目 `main` 已有的 9 个本地提交；这些提交是旧的具身实验代码。新提交只包含暂存区中经过检查的文件。HTTPS 推送需要 GitHub 认证；不要把令牌写进仓库或远程 URL。上传后在 `https://github.com/yatao-zhuozhuo/embodied-adversarial/tree/feat/embodied-asymmetric-self-play` 检查文件。

## 首次运行所需的外部输入

代码仓库之外仍需准备：

| 输入 | 环境变量 | 说明 |
| --- | --- | --- |
| Qwen3.5-4B 基座模型 | `MODEL_PATH` | 与 LoRA adapter 匹配的模型目录 |
| 初始 Alice LoRA | `INITIAL_ALICE_ADAPTER` | 目录中应有 `adapter_config.json` 和权重 |
| 初始 Bob LoRA | `INITIAL_BOB_ADAPTER` | 同上 |
| 可用 Alice 题目来源 | `BOOTSTRAP_ALICE_RUN` | 需至少有 8 个不同的有效 Alice 任务及对应 snapshot；前 4 个用于 bootstrap train，后 4 个用于固定 holdout |

当前入口在验证阶段就检查上述两个 adapter 和 `BOOTSTRAP_ALICE_RUN` 目录。后者的 JSON 任务里有 `snapshot.state_uri` 绝对路径，snapshot 文件也必须存在，因此把旧轨迹排除出 Git 后，原脚本无法仅靠代码从零直接启动。可以在目标平台本地先生成这批有效任务，或通过独立于 Git 的工件存储提供它们；如果复制旧工件，需要修正其中 `state_uri` 到目标机器路径。没有这批任务时，不应拿空目录冒充 bootstrap 来源。

## 目标平台安装与检查

此配置以 Linux x86_64、NVIDIA CUDA/Vulkan、8 张可见 GPU、Python 3.12 为目标。`configs/requirements-embodied-swift.txt` 固定了本机已安装的 ms-swift 4.4.2、vLLM 0.19.1、Torch 2.10.0、Triton 3.6.0、Transformers 5.12.1、Accelerate 1.15.0 和 PEFT 0.19.1。vLLM 0.19.1 自身也要求 Torch 2.10.0；ms-swift 4.4.2 要求 `peft<0.20`。根目录 `pyproject.toml` 的 `grpo` extra 当前要求 `peft>=0.21`，因此不要用 `uv sync --extra grpo` 来构建此 Swift 环境。

需求文件中的 `causal-conv1d` URL 是 Linux x86_64 / Python 3.12 / Torch 2.10 的预编译 wheel；其他架构需要替换对应 wheel。FLA 的 TileLang 后端通过 `setup_fla_tilelang_overlay.sh` 安装到独立目录，训练时由 `PYTHONPATH` 加载，不覆盖 Torch 自带的 Triton。两个 setup 脚本都是安装配方，现有虚拟环境本身不需上传。`decord` 不在 staged 图像训练调用链中，已从专用需求文件移除。

```bash
cd OpenETA
source configs/embodied-swift-staged.example.env
# 编辑或覆盖 MODEL_PATH、INITIAL_*、BOOTSTRAP_ALICE_RUN 等路径
PYTHON_BIN=python3.12 ./scripts/setup_embodied_swift_env.sh
./scripts/setup_fla_tilelang_overlay.sh
DRY_RUN=true ./scripts/run_embodied_swift_staged_formal5.sh
```

通过路径检查后，再在目标 GPU 节点运行 `scripts/run_embodied_swift_staged_job.sh` 做 FLA 与 ManiSkill 原生后端检查，或直接运行 `scripts/run_embodied_swift_staged_formal5.sh`。`RUN_NAME`、`RUN_ROOT`、GPU 编号、端口和显存参数都可用环境变量覆盖。训练入口默认值中的 `/inspire/...`、`/opt/openeta-swift` 是原集群配置；示例环境文件覆盖这些值。TileLang 安装脚本现在默认使用公开 PyPI，也可用 `INDEX_URL` 覆盖。

已有 `RUN_ROOT` 的 `summary.json` 记录绝对 checkpoint 路径。迁移续跑时需重写这些路径并校验 checkpoint；新平台首次运行建议使用新的 `RUN_NAME`。

## Bounded GRPO 公共入口（2026-10）

同步 colocate 训练现在由 `scripts/run_embodied_bounded_colocate.py` 统一启动。`OPENETA_TRAIN_ROLE=alice` 注册 `StagedEmbodiedGRPOTrainer`，仍执行 Alice → 冻结 Bob → Alice reward；`OPENETA_TRAIN_ROLE=bob` 注册 `BoundedEmbodiedGRPOTrainer`，只执行 Bob 单策略 rollout。两者共同使用请求级每-turn/累计 token 预算、输出对齐、vLLM wake/sync/sleep 和阶段日志。旧的 `run_embodied_staged_colocate.py` 仅保留为 Alice 兼容 wrapper。

第一阶段固定 `OPENETA_HISTORY_MODE=full`，默认 `OPENETA_THINKING_MAX_TOKENS=1024`、`OPENETA_TRAJECTORY_MAX_TOKENS=24576`、`OPENETA_CONTEXT_MAX_TOKENS=32768`。每个 episode 的 artifact 会记录逐 turn token 数、累计 token 与预算终止原因；每个 rank 的 `phase_timing.rank-NN.jsonl` 会记录 wake、权重同步、rollout、sleep 以及分阶段 allocator 显存。可在独立 canary 完成后执行：

```bash
python scripts/probe_embodied_bounded_grpo.py \
  --artifact-root /path/to/canary-round \
  --output /path/to/canary-round/bounded-probe.json
```

当前仓库默认使用共享目录旁的 `.venv_selfplay_embodied`。原模型冷启动使用 `OPENETA_COLD_START=true` 和 `__base_model__`，不要求预先存在 Alice/Bob LoRA；已有 LoRA 模式仍校验 `adapter_config.json`。正式恢复前先以新的 `RUN_ROOT` 做至少两个 Bob optimizer step 和一个完整 Alice 三阶段更新，不能把 CPU helper 测试视为 GPU 验收。
