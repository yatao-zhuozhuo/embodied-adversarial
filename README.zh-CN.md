# OpenETA

<p align="center">
  <img src="docs/assets/openeta-header-eta-clear.png" alt="OpenETA — 具身任务智能体" width="100%">
</p>

[English](README.md) · [简体中文](README.zh-CN.md)

[📄 论文](https://arxiv.org/abs/2608.03924) ·
[🌐 项目主页](https://openmoss.ai/OpenETA/) ·
[💻 GitHub](https://github.com/OpenMOSS/OpenETA)

> **一种将智能从数字世界迁移到物理世界的全新智能体范式。**

物理智能的本质，是与世界形成闭合的因果循环：理解当前状态，采取行动改变它，观察真实结果，更新底层信念，并在持续交互中积累经验。前沿智能体模型已经开始在数字领域展现这种能力：它们围绕目标进行推理、调用工具、检查执行结果，并通过反馈完成长程任务。它们仍然缺少的，是一种将这一循环可靠地延伸到物理世界的方法。

OpenETA 将从数字智能体（Digital Agent）到物理智能体（Physical Agent）的转变探索为一种新范式。它不把具身智能简化为单一的观测到动作模型，而是将智能体置于世界的因果链中，使感知、行动、验证与学习成为一个连续过程。智能由此超越对数字信息的处理，开始理解并改变现实。

[架构](docs/architecture.md) ·
[动作流水线](docs/agent-action-pipeline.md) ·
[并行评估](docs/parallel-simulator-evaluation.md) ·
[真机部署](real/README.md)

<a id="yicloud-rl"></a>
## Yicloud 平台 RL（`feat/rl-yi`）

本分支基于 `feat/rl`，提供共享 GPFS 环境下的两节点 GRPO 训练与完整断点恢复。

| 项目 | 配置 |
| --- | --- |
| 模型 | `YangYongzhuo/Qwen3.5-4B-embodied-mixed` 或 `YangYongzhuo/Qwen3.5-9B-embodied-mixed` |
| 每个训练任务 | 2 个节点，各 8 张 H200；提交后生成两个 RJob |
| 训练题目 | 简单 16 题、中等 32 题，来自 MetaWorld、LIBERO、ManiSkill、RoboCasa |
| 基线隔离 | 排除 `configs/single_model_rl_baseline_v2_exclusion.json` 冻结的 69 道 baseline 题目，包含任务别名校验 |
| 更新方式 | 每阶段 3 Epoch，每批 4 题，每题 16 条 rollout，每批更新 1 次 |
| 更新总数 | 简单阶段 12 次、中等阶段 24 次，累计 36 次 |
| 训练协议 | 保持 `openeta.two_tool_input.v1` / `openeta.two_tool_xml.v1`、原生任务成功奖励及原有产物结构 |

### 环境与配置

4B 和 9B 分别使用 [`single_model_rl_curriculum_mixed_yi_2n16.json`](configs/single_model_rl_curriculum_mixed_yi_2n16.json)
和 [`single_model_rl_curriculum_mixed_yi_9b_2n16.json`](configs/single_model_rl_curriculum_mixed_yi_9b_2n16.json)，两者仅模型路径不同。
`single_model_rl_curriculum_mixed_yi.json` 保留原始单节点配置，用于原配置验收。

平台路径集中在 [`scripts/single_model_rl_yi_env.sh`](scripts/single_model_rl_yi_env.sh)。
当前部署依赖共享存储中已有的 Swift 训练环境、baseline coordinator、四个模拟器环境和本地模型目录，
并要求 `brainctl` 已登录且具有 `fdj-infra` 命名空间的任务权限。
[`setup_single_model_rl_yi.sh`](scripts/setup_single_model_rl_yi.sh) 用于补充平台依赖与原始 4B 模型，
不是从零安装所有模拟器的脚本；9B 模型需事先放到对应配置指定的目录。

### 提交两节点任务

在仓库根目录执行。每次启动使用新的 run 目录及任务名；下例为 4B，9B 改用对应配置并单独命名。
`original_update_acceptance.json` 应来自已经完成实际更新的原始配置验收。

```bash
run_root="runs/rl_yi_4b_$(date +%Y%m%d_%H%M%S)"
job_name="embodied-rl-yi-4b-$(date +%m%d%H%M%S)"
acceptance="runs/single_model_rl_curriculum_mixed_yi_start04/diagnostics/original_update_acceptance.json"

python3 scripts/run_single_model_curriculum.py \
  --config configs/single_model_rl_curriculum_mixed_yi_2n16.json \
  --run-root "$run_root" --prepare-only

# 启用每卡渲染检查及 DrawTriangle 原生 300 步边界检查。
python3 - "$run_root/curriculum_plan.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
plan = json.loads(p.read_text())
plan['startup_maniskill_gpu_probe'] = True
p.write_text(json.dumps(plan, indent=2) + '\n')
PY

python3 scripts/submit_single_model_rl_yi_jobs.py \
  --run-root "$run_root" --name "$job_name" \
  --original-acceptance "$acceptance" \
  --exclude-node lg-cmc-gpu-0941.host.lg.shzhisuan.com \
  --exclude-node lg-cmc-gpu-1207.host.lg.shzhisuan.com
```

两个排除节点曾出现原生 Vulkan 渲染失败，节点恢复后可重新验证再调整。
提交器在 `jobs/run_job.sh` 保存启动脚本快照，防止共享脚本后续编辑影响正在执行的 Bash。
启动顺序为 GPU 检查、16 卡通信、回归测试、两节点各 48 道原生题目检查、模型训练。

### 断点恢复

使用新的 run 目录，准备计划时额外传入以下参数，再按上述方式开启 GPU 检查并提交：

```bash
python3 scripts/run_single_model_curriculum.py \
  --config configs/single_model_rl_curriculum_mixed_yi_9b_2n16.json \
  --run-root runs/rl_yi_9b_resume --prepare-only \
  --resume-checkpoint /absolute/path/to/checkpoint-N \
  --history-plan /absolute/path/to/previous_run/curriculum_plan.json
```

断点须包含 adapter 权重、`optimizer.pt`、`scheduler.pt` 和 `trainer_state.json`；配置必须与原任务一致。
恢复会跳过已完成阶段，保留累计步数和优化器，并按当前阶段已经完成的更新数跳过数据批次。
例如累计 step 25 表示中等阶段完成 13 次更新，下一批是该阶段的第 6 批。
达到 step 36 即完成全部训练，无需再次提交。

### 产物与 TensorBoard

- `curriculum_status.json`：课程状态；`stages/*/stage_result.json`：阶段结果。
- `formal/trainer/<stage>/v*/checkpoint-*`：完整恢复断点，每次更新保存，每阶段保留最近两个。
- `formal/trainer/<stage>/rollouts/`：原始轨迹与输入审计；`nodes/`：节点日志及检查证据。
- `tensorboard/continuous/`：根据 checkpoint 历史生成的连续全局步数曲线。

在能读取共享存储的工作空间启动 TensorBoard：

```bash
source scripts/single_model_rl_yi_env.sh
"$SWIFT_VENV/bin/python" -m tensorboard.main \
  --logdir /absolute/path/to/run/tensorboard/continuous \
  --host 127.0.0.1 --port 6006
```

本机建立 SSH 转发后访问 `http://127.0.0.1:26006`。将 `WORKSPACE_SSH_USER` 替换为运行 TensorBoard 的工作空间 SSH 用户：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:26006:127.0.0.1:6006 \
  -p 2202 WORKSPACE_SSH_USER@ssh.gate.yicloud.com.cn
```

本地端口被占用时，更换 `26006` 并使用对应浏览器地址。恢复任务使用新 run 目录时，TensorBoard 的 logdir 也应指向新目录。
运行目录、模型权重、依赖环境与认证信息留在平台，不纳入 Git 提交。

## 最新动态

- **2026-09-06** — **Stage 2 Agent Harness：** 我们将 Stage 2 闭环具身智能体 Harness
  合入 `main`，新增 35 个经过验证的工具契约、支持持久化 Rollout 的有界多模态上下文、
  统一评测入口、具备可达性与碰撞感知的运动能力，以及 Human VLM 人工操作工作台。
- **2026-08-03** — **OpenETA for Codex：** 我们发布了
  [`openeta-light`](https://github.com/OpenMOSS/OpenETA/tree/openeta-light)
  分支，通过六个类型化工具与版本化 Operator 上下文，将 Codex TUI 接入 LIBERO。使用
  `gpt-5.6-sol`（medium reasoning effort），Codex + OpenETA 在全部 130 个 LIBERO 任务上
  取得 **70.8% Pass@1** 和 **90.0% Pass@5**。
- **2026-07-27** — **真机部署：** 我们发布了
  [`dev/real-robot-deployment`](https://github.com/OpenMOSS/OpenETA/tree/dev/real-robot-deployment)
  分支，其中包含真机部署栈、硬件接口和物理智能体工具。
- **2026-07-25** — **物理世界 Hello World：** OpenETA 首次公开发布，将智能体循环从数字领域延伸至物理世界。

## 核心特性

| 能力 | OpenETA 系统边界 |
| --- | --- |
| 因果闭环 | 每次改变世界的动作都会产生一次全新观测义务；在该义务完成前，不得继续推理、恢复或完成任务。 |
| 可替换的 Planner | 不同的 LLM 和 VLM 通过兼容 OpenAI 的接口接入，并支持主端点和回退端点。 |
| 可组合的物理能力 | 感知、抓取、放置、运动以及未来的原生策略通过稳定的 Tool 和 MCP 接口进行组合。 |
| Tool / Skill 分离 | Tool 是由 Host 持有的原子能力；Skill 是可读、可审查的经验，绝不是隐藏执行。 |
| Host 持有的监督权 | Schema、来源、安全门、审批模式和审查者均处于 Agent 的权限之外。 |
| 可审计的 Memory 与 Rollout | Session 工作记忆与不可变证据分开存储，以便重放成功、失败和不确定性。 |
| 统一的 sim-to-real 边界 | 模拟器和机器人共享观测、动作、结果及 MCP 生命周期契约。 |
| 可验证的经验演进 | 经验最初是 session 局部的候选项，只有经过审查、金丝雀验证和留出集验证后才能晋级。 |

核心循环为：

```text
observe -> reason -> act -> verify world change -> observe again -> adapt
```

## 架构

<p align="center">
  <img src="docs/assets/openeta-framework-black.png" alt="OpenETA 框架：智能体智能跨越 Host 接口进入模拟器和机器人，并由世界反馈闭合因果循环" width="100%">
</p>

系统受到三重边界约束：

- **认知 / 执行边界**：Agent 提出意图，Host 决定是否执行，而 Simulator 或 Robot 保留对环境真实状态的决定权。
- **动作边界**：只读能力可以并行运行，而会改变世界的 AtomAction 每次只能执行一个，并且必须先满足全新观测义务，才能执行下一个动作。
- **证据边界**：Tool 完成、世界状态变化和任务成功是三个不同的主张。只有可信的环境回执、奖励、终止信号或检查器才能确认任务完成。

## 功能矩阵

<table>
  <thead>
    <tr>
      <th>智能体 Planner</th>
      <th>Tools</th>
      <th>模拟器</th>
      <th>真实世界</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td valign="top">
        <ul>
          <li>兼容 OpenAI 的 LLM / VLM ✅</li>
          <li>自定义 Planner ✅</li>
          <li>主 / 回退端点 ✅</li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><strong>语义 Tools</strong>
            <ul>
              <li><strong>感知</strong>
                <ul>
                  <li>SAM 3 ✅</li>
                  <li>UniDepth V2 ✅</li>
                  <li>MolmoPoint-8B ✅</li>
                </ul>
              </li>
              <li><strong>抓取与放置</strong>
                <ul>
                  <li>AnyGrasp ✅</li>
                  <li>GraspGenX ✅</li>
                  <li>AnyPlace ✅</li>
                  <li>Contact-GraspNet</li>
                </ul>
              </li>
            </ul>
          </li>
          <li><strong>控制 Tools</strong>
            <ul>
              <li><strong>策略适配器</strong>
                <ul>
                  <li>OpenVLA</li>
                  <li>OpenVLA-OFT</li>
                  <li>openpi</li>
                  <li>GR00T N1.6 / N1.7</li>
                </ul>
              </li>
              <li><strong>控制器与安全</strong>
                <ul>
                  <li>robosuite <code>OSC_POSE</code> ✅</li>
                  <li>cuRobo ✅</li>
                </ul>
              </li>
            </ul>
          </li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><strong>操作环境</strong>
            <ul>
              <li>MetaWorld / Sawyer ✅</li>
              <li>ManiSkill / 多种机器人 ✅</li>
              <li>LIBERO / Franka Panda ✅</li>
              <li>RoboCasa365 / PandaOmron ✅</li>
              <li>BEHAVIOR-1K / R1Pro ✅</li>
              <li>Genesis / Franka ✅</li>
            </ul>
          </li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><strong>观测</strong>
            <ul>
              <li>RealSense D400 / L515 ✅</li>
              <li>Webcam / RTSP ✅</li>
              <li>UR5e 状态 ✅</li>
            </ul>
          </li>
          <li><strong>运动</strong>
            <ul>
              <li>UR5e ✅</li>
              <li>Franka</li>
            </ul>
          </li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

✅ 已适配。未标记的项目尚未适配。模拟器后端使用隔离的 venv；安装方法、环境数量和动作空间请参阅[模拟层](sim/README.md)。

## 快速开始

### 安装与运行

需要 Python 3.10+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync --extra dev
uv run openeta
```

在 TUI 中使用 `/provider` 配置兼容 OpenAI 的 Planner，然后直接输入任务。常用命令包括 `/tools`、`/memory`、`/approvement`、`/run`、`/sessions` 和 `/resume`。

如需运行有明确边界的单项任务：

```bash
uv run openeta --once "inspect the current scene" --max-turns 8
```

### 接口

| 入口 | 用途 |
| --- | --- |
| `uv run openeta` | 用于配置 provider、检查 Tool、执行任务和恢复 session 的交互式 Agent TUI |
| `uv run openeta --once "..."` | 具有有限轮次的单项任务执行 |
| `uv run openeta-batch --manifest ...` | 采用隔离 episode 和人工恢复的并行评估 |
| `uv run openeta --command preflight/run/iterate/inspect ...` | 无人值守的实验与经验迭代 |
| `uv run openeta-replay EPISODE_DIR --output replay.mp4` | 根据 episode 日志重建时间线和视频 |
| `uv run python -m sim.mcp_server` | 模拟器 MCP、REST API 和 Web Dashboard |
| `uv run openeta-robocasa-benchmark ...` | RoboCasa365 manifest、任务套件和基准测试实用工具 |

完整的 CLI 参考请参阅 `uv run openeta --help`、`uv run openeta-batch --help` 以及各子命令的 `--help`。

### 模拟器与 Dashboard

安装一个模拟器后端，然后启动 MCP 服务和实时 Dashboard：

```bash
bash scripts/setup_envs.sh libero
uv run python -m sim.mcp_server
# Open http://localhost:8765/
```

### 评估

在不连接模型或 MCP 服务的情况下验证并行评估 manifest：

```bash
uv run openeta-batch \
  --manifest examples/parallel_libero_eval.json \
  --validate-only
```

仓库包含 LIBERO 金丝雀集、训练集、留出集和完整评估 manifest。连接 Planner 和所需 MCP 服务后，并行运行相互隔离的 episode：

```bash
uv run openeta-batch \
  --manifest examples/parallel_libero_eval.json \
  --concurrency 10 \
  --approvement reviewed_autonomy \
  --output outputs/parallel-libero.json
```

无人值守实验分为四个阶段：`preflight -> run -> iterate -> inspect`。

```bash
uv run openeta --command preflight \
  --manifest examples/parallel_libero_eval.json

uv run openeta --command iterate \
  --train-manifest examples/parallel_libero_six_train.json \
  --validation-manifest examples/parallel_libero_six_holdout.json \
  --experiment-id libero-skill-v1 \
  --rounds 3 \
  --approvement reviewed_autonomy \
  --on-need-human fail

uv run openeta --command inspect \
  --experiment-id libero-skill-v1
```

进程完成并不代表任务成功。最终结果由可信的环境回执、奖励、检查器和保留的 rollout 证据确定。

## 路线图



## 文档

| 文档 | 范围 |
| --- | --- |
| [架构](docs/architecture.md) | Agent、Adapter、Simulator 和 MCP 各层之间的边界 |
| [Agent 命令流水线](docs/agent-action-pipeline.md) | 命令 schema、Tool 契约、AtomAction 和安全门 |
| [模拟层](sim/README.md) | 安装、环境注册表、MCP、REST 和 Dashboard |
| [并行模拟器评估](docs/parallel-simulator-evaluation.md) | 并发、预算、人工恢复和经验晋级 |
| [Rollout 数据契约](docs/rollout-data-contract.md) | Session 数据层与不可变证据 |
| [代码策略运行时](docs/code-policy-runtime.md) | 有边界的代码后端与沙箱边界 |
| [标定生命周期](docs/calibration-lifecycle.md) | 标定提案、审查和发布 |
| [真机部署](real/README.md) | 相机、机械臂、真实环境 MCP 和安全边界 |

## 贡献

OpenETA 仍处于积极研究和开发阶段。新的 Planner、Tool、模拟器、机器人和 Skill 应保留现有的观测、命令与结果契约，并包含针对副作用、失败语义和资源清理的测试。对 schema、Tool 契约、任务成功语义或安全门的更改必须同步更新相应的设计文档。

```bash
uv sync --extra dev
uv run pytest
```

## 致谢

OpenETA 最初的模拟器集成迁移并改编自 [RLinf](https://github.com/RLinf/RLinf)。本项目建立在 LIBERO、MetaWorld、ManiSkill、RoboCasa、BEHAVIOR、MuJoCo、SAPIEN 和 OmniGibson 等开源机器人生态系统之上。其感知和操作能力受益于 SAM 3、UniDepth、AnyGrasp、AnyPlace、Contact-GraspNet、GraspGenX 和 Molmo。

## 社区

<p align="center">
  <img src="https://raw.githubusercontent.com/SII-ZhangYiFei/group-qr/main/group.png" alt="OpenETA 微信交流群" width="240">
</p>

欢迎加入 OpenETA 微信技术交流群，扫描上方二维码即可入群。

## 引用与许可证

如果 OpenETA 对你的研究有帮助，请引用：

```bibtex
@misc{chen2026etanewagenticparadigm,
      title={ETA: A New Agentic Paradigm for Embodied Tasks},
      author={Yitong Chen and Zezheng Huai and Sixian Li and Yubang Wang and Haozhe Zhang and Yifei Zhang and Hechang Chen and Jingjing Gong and Yu-Gang Jiang and Xipeng Qiu},
      year={2026},
      eprint={2608.03924},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.03924}
}
```

OpenETA 采用 [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) 许可。第三方组件和迁移的代码仍受各自上游许可证及保留的源文件声明约束。
