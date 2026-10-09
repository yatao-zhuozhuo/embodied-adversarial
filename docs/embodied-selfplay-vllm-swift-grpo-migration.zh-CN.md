# OpenETA 具身 Self-Play 迁移到 vLLM + ms-swift GRPO

> 状态：MVP 已实现并在 2×RTX 4090 上端到端实跑（2026-09-24）
> 适用范围：当前 `PickCube-v1`、`cube_at_position`、Qwen3.5-4B、Alice/Bob 非对称 Self-Play 代码
> 目标：把当前可审计的单进程 `Transformers + PEFT + 手写 GRPO` 实现，分阶段迁移为 `vLLM rollout + ms-swift GRPO`，同时保持 snapshot、真实环境执行、任务编译和 trusted reward 的语义不变。

## 0. 当前落地状态

本文后半部分保留了完整迁移设计和扩展门槛；本节记录已经落地、可以直接复现的 MVP，避免把历史规划误读成“尚未编码”。

当前锁定环境位于 `/opt/openeta-swift`：

```text
Python                     3.12.14
PyTorch                    2.10.0+cu128
ms-swift                   4.4.2
vLLM                       0.19.1
ManiSkill                  3.0.1
flash-linear-attention     0.5.2
causal-conv1d              1.7.0+cu12torch2.10
```

已经实现：

- `plugins/embodied_swift_grpo.py`：Alice/Bob 多轮 scheduler、每 request 独立 ManiSkill session、snapshot 恢复、动态图像、regex 动作约束、action-only loss mask 和 trusted ORM；
- Alice 隐藏原始 PickCube goal，真实动作结束后 replay，并编译 `cube_at_position`；
- 有效 Alice proposal 通过独立 HTTP 调用固定 Bob rollout server 做 `K` 次评估，使用 Bob success rate 的 competence-boundary reward；
- 无效 proposal 不调用 Bob；Bob 服务故障记作 evaluator error 和中性 reward，不伪装成策略失败；
- `scripts/run_embodied_swift_grpo.sh`：vLLM server mode 和 Swift GRPO trainer，支持新训练、adapter 加载和 checkpoint resume；
- `scripts/run_embodied_swift_selfplay_smoke.sh`：双 4090 的 Bob→Alice 冒烟轮次编排；
- `scripts/create_embodied_swift_dataset.py`：Alice snapshot 数据和只来自有效 Alice artifact 的 Bob 数据；
- `scripts/validate_embodied_swift_scheduler.py`：固定动作、真实 ManiSkill、真实冻结 Bob evaluator 的确定性 canary。

本次实跑结果：

| 验证项 | 结果 |
|---|---|
| Qwen3.5-4B vLLM 视觉生成 | 通过 |
| Bob 4-turn 真实环境 rollout | 通过，action token mask/logprob 对齐 |
| Bob Swift GRPO 1 step | 通过，保存 `checkpoint-1` |
| Alice 80-step 已知有效轨迹 replay | 通过，位移 `0.0560m` |
| 冻结 Bob 对该 Alice task 评估 | 2 次，成功率 `0/2` |
| Alice competence-boundary reward | `0.431818`，stage=`frozen_bob_boundary` |
| Alice Swift GRPO 1 step | 通过，保存 `checkpoint-1` |
| Bob LoRA checkpoint 重新加载到 vLLM | 通过，并完成 4-turn 环境 rollout |

实跑 checkpoint：

```text
runs/swift_smoke/bob_grpo_fla/v0-20260924-151004/checkpoint-1
runs/swift_smoke/alice_grpo/v0-20260924-152050/checkpoint-1
```

这两次最小 GRPO group 的组内 reward 恰好相同，因此 `reward_std=0`、loss/grad 为 0；这是 GRPO 在零方差 group 上的正确行为。它证明了完整执行和 checkpoint 链路，但不代表策略能力已经提升。正式训练必须增加 generation 数、温度、任务多样性和训练步数，并监控 zero-variance group 比例。

尚未完成的是生产规模优化：多进程 simulator pool、跨故障幂等 evaluator cache、多个 round 的 manifest/resume、固定 holdout 能力曲线以及 Megatron/colocate。当前 MVP 已覆盖单机双卡的完整算法边界，但不要把 1-step smoke 当成收敛实验。

最短复现命令：

```bash
cd /inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA
./scripts/setup_embodied_swift_env.sh

# 默认从已有有效 Alice artifacts 引导 Bob，依次训练 Bob 和 Alice，退出时关闭两个服务。
RUN_ROOT="$PWD/runs/swift_selfplay_smoke_replay" \
  ./scripts/run_embodied_swift_selfplay_smoke.sh
```

## 1. 结论先行

这次迁移不应理解成“把 `model.generate()` 替换成 vLLM 的一个调用”。当前脚本同时承担了模型推理、环境状态机、轨迹记录、奖励计算、GRPO loss、LoRA 更新和 Alice/Bob 交替训练。迁移必须先拆开这些职责。

推荐路线是：

1. 保留当前单进程实现，作为语义基线和回归测试 oracle；
2. 先把 **Bob rollout** 接入 ms-swift `MultiTurnScheduler`，由 vLLM 逐步生成动作，由独立 ManiSkill session 执行动作；
3. 再让 ms-swift 接管 Bob 的 GRPO、reference/KL、importance ratio、LoRA 保存和分布式训练；
4. Bob 通过一致性门槛后，再迁移 Alice proposal rollout；
5. Alice 的终局 reward 不能只看 Alice 自己是否成功，而要由固定版本 Bob 对该 proposal 做 `K` 次 rollout 后计算；
6. 最后由一个外层 round coordinator 交替冻结 Alice/Bob，并编排两个独立的 Swift 训练任务；
7. 第一版使用 vLLM **server mode**，方便隔离训练进程、rollout 进程和 ManiSkill worker；语义稳定后再评估 Megatron-SWIFT colocate 模式。

不能直接复用 `adversarial-mle-agents/plugins/mle_agent_grpo.py` 的环境逻辑。它可以复用的是 ms-swift 插件形状和多轮生命周期，而不是 CSV/Kaggle 工具环境本身。具身 scheduler 必须单独实现图像 observation、snapshot 恢复、动作执行、goal checker 和 simulator cleanup。

### 1.1 Alice 和 Bob 的任务必须这样定义

迁移后的所有 prompt、scheduler、reward 和评测都必须服从下面的角色定义。这不是措辞上的说明，而是 Self-Play 的任务契约。

#### Alice：出题者 / task proposer

Alice 的任务是：

> 从给定初始 snapshot 出发，通过真实、合法的环境动作，把红色方块移动到一个与初始状态明显不同、物理上可达的新位置，从而创造一个困难但可解的新任务。

具体要求：

- Alice 必须在环境中真实执行动作，不能只输出坐标或语言描述；
- Alice 的最终 cube position 是候选任务真值；
- 候选位置必须通过最小位移、工作区边界、真实操作证据和 replay 一致性检查；
- Alice 应当提高 Bob 的解题难度，但不能通过越界、物理 bug、不可观察目标或根本无解的状态获得高奖励；
- “尽可能难”应实现为寻找 **Bob 当前能力边界附近** 的任务，而不是让 Bob 成功率越低越好；
- 当前默认目标是让 Bob success rate 接近 `target_bob_success_rate=0.45`，并同时奖励有效性与新颖性；
- Alice 完成后，host 把最终位置编译成 `cube_at_position` predicate、task ID 和自然语言 instruction。

Alice 的目标输出不是一句文本，而是一条经过环境验证的轨迹及其终局状态：

```text
Alice(initial snapshot, current Bob capability)
  -> real action trajectory
  -> final cube position
  -> replay/validity gate
  -> cube_at_position task
```

#### Bob：解题者 / task solver

Bob 的任务是：

> 从与 Alice 完全相同的初始 snapshot 出发，在看不到 Alice 动作轨迹的情况下，独立地把红色方块移动到 Alice 创造的最终目标位置。

这里的“复现 Alice”特指 **复现 Alice 创造的最终状态/任务结果**，不是逐动作模仿 Alice：

- Bob 不接收 Alice action sequence、rationale 或中间状态；
- Bob 不需要使用和 Alice 相同的路径、动作数量或抓取策略；
- host 将绿色 goal marker 设置到 Alice 的最终 cube position；
- Bob 只接收自己的 observation、目标 marker、instruction 和允许公开的状态 projection；
- Bob 是否成功只由 trusted `cube_at_position` checker 判断；
- 只要最终方块进入目标容差，Bob 使用任何合法轨迹都算成功。

二者关系可以概括为：

```text
相同 initial snapshot
       |
       +--> Alice：真实移动方块，创造困难但可达的目标 G
       |                 |
       |                 +--> host 验证并编译 predicate(G)
       |
       +--> Bob：看不到 Alice 轨迹，独立把方块移动到 G
                              |
                              +--> success/failure 反馈给 Alice
```

因此，这个项目训练的不是 trajectory imitation，而是 **Alice 生成环境目标、Bob 复现目标状态、Bob 的成功率反过来塑造 Alice 课程难度**。

面向迁移实现的角色级 prompt 核心句应固定为：

```text
Alice:
You are Alice, the task proposer. Starting from the current state, use real legal
actions to move the red cube to a new reachable position that is difficult but
solvable for the current Bob. You are creating the goal state, not solving an
existing goal. Finish with DONE only after the new goal state has been established.

Bob:
You are Bob, the task solver. Starting from the same initial state, independently
move the red cube to the visible green target created from Alice's final state.
You do not have Alice's trajectory and do not need to imitate her actions. Succeed
by reaching the target state with any legal action sequence.
```

具体 observation、动作列表和安全控制说明可以逐步追加，但不能改变这两段角色语义。

#### 迁移前基线与目标任务定义的差距

迁移前的 `ALICE_INSTRUCTION` 已经要求 Alice 创造新位置，但 Alice 和 Bob 仍共用 `_prompt()` 中的控制规则。Alice 还能看到 ManiSkill 原始 goal 的坐标和 marker，并被提示“抓取后向 goal 对齐”。这会把 Alice 强烈推向原始 PickCube 目标，而不是主动创造多样化的新目标。新实现已在 `agent/runtime/embodied_selfplay_prompt.py` 和 Swift scheduler 中修正这些问题。

迁移时应先保留旧 prompt 完成 engine parity，然后单独完成角色语义修正：

- Alice prompt 改成明确的 `task proposer`，而不是 `task-demonstration agent`；
- Alice proposal 阶段隐藏或中和 ManiSkill 原始 goal marker；
- Alice prompt 不再使用 Bob 的 `goal_minus_cube_xyz` 和“向现有 goal 对齐”控制规则；
- Alice 可以看到合法工作区、动作集合、剩余步数以及聚合后的 Bob 能力信息，但不能看到会直接泄漏 reward/checker 的私有字段；
- Bob prompt 继续面向 Alice 编译后的 goal marker，并明确要求独立完成目标；
- prompt 语义修正作为独立实验提交，不能与 vLLM/Swift engine parity 混成一个无法归因的改动。

## 2. 为什么最初实现没有直接使用 vLLM / Swift

最初代码是有意构建的 correctness scaffold，而不是最终吞吐训练器：

- `scripts/train_embodied_grpo.py` 的模块说明明确写着它是“single-process correctness entrypoint”；
- `agent/training/embodied_grpo.py` 明确不依赖 trainer framework，目的是先让 reward、action token 和 loss 可以人工检查；
- `scripts/run_embodied_selfplay.py` 在一个显式 Python 循环中完成 snapshot、Alice proposal、replay、Bob 评估、课程选择和更新，便于定位物理环境错误；
- 当前模型每一步只生成一个离散 action，性能瓶颈尚未与 simulator、图像预处理和嵌套 Bob 评估分离，过早接入 vLLM 会把错误来源混在一起；
- 当时 OpenETA 环境中没有安装 `swift` 和 `vllm`；本次迁移已在隔离环境 `/opt/openeta-swift` 中安装并验证；
- 相邻 `adversarial-mle-agents` 已经使用 ms-swift/vLLM 的部分，是固定 Agent2 任务的多轮 GRPO，不是完整 Alice/Bob 具身 Self-Play。

因此，Transformers 版本不是最终框架选择，而是环境契约和 GRPO 语义的参考实现。现在仍保留它作为回归 oracle。

## 3. 当前代码基线

### 3.1 当前职责分布

| 职责 | 当前代码 | 当前行为 | 迁移后归属 |
|---|---|---|---|
| ManiSkill 环境 | `adapter/maniskill_sim.py` | 单个 `num_envs=1` 的 `PickCube-v1`；离散动作转连续控制 | OpenETA simulator worker |
| snapshot | `adapter/maniskill_sim.py` | 保存完整 state、episode step、gripper state，并校验 SHA-256 | OpenETA，保持权威 |
| action schema | `agent/training/embodied_grpo.py` | 16 个允许动作；兼容裸 action 和旧 JSON | OpenETA 共享纯函数 |
| prompt/图像构造 | `scripts/train_embodied_grpo.py` | Qwen chat template，一张 RGB 图，加机器人/方块/目标相对位置 | OpenETA 共享 projection；Swift scheduler 调用 |
| 本地生成 | `scripts/train_embodied_grpo.py::_generate_action` | Transformers `model.generate()`，prefix token allowlist | vLLM |
| 环境 rollout | `scripts/train_embodied_grpo.py::_rollout_episode` | 恢复 snapshot，逐步生成并执行 action | Swift multi-turn scheduler + OpenETA worker |
| Bob reward | `agent/training/embodied_grpo.py::episode_reward` | success、grasp、dense reward、非法动作惩罚 | Swift ORM，从 trusted receipt 读取 |
| GRPO | `agent/training/embodied_grpo.py`、`scripts/train_embodied_grpo.py::_train_group` | 手写 advantage、clip、KL 和逐轨迹反传 | ms-swift GRPO |
| Alice proposal | `scripts/run_embodied_selfplay.py` | Alice 真执行，回放，编译 `cube_at_position` | Alice scheduler + OpenETA task compiler |
| Alice reward | `scripts/run_embodied_selfplay.py` + `adversarial-mle-agents` curriculum | 根据 validity、novelty 和 Bob 成功率给分 | Alice ORM + 固定 Bob evaluator service |
| 交替训练 | `scripts/run_embodied_selfplay.py` | 同一进程持有两个模型和一个环境 | 外层 round coordinator，两个独立 Swift job |
| checkpoint | `save_pretrained` + optimizer state | 自定义目录和 resume state | Swift checkpoint + round manifest |

### 3.2 必须保持不变的语义

迁移 engine 时，以下语义不能顺便修改：

- 同一 GRPO prompt group 的所有 generation 必须来自同一个 snapshot digest；
- 每个 generation 使用独立环境实例，不能共享正在变化的 ManiSkill 对象；
- Alice 的目标必须来自真实执行后的 cube 最终位置，而不是语言声明；
- Alice proposal 必须经过最小位移、工作区边界、动作证据、replay 一致性和 predicate 编译检查；
- Bob 从 Alice 的初始 snapshot 开始，只看到目标 marker、instruction 和允许的 observation projection；
- checker、snapshot 内容、Alice trajectory 和 evaluator 私有字段不能泄漏到 Bob prompt；
- reward 只能从 host-owned simulator receipt 计算，不能让模型自报成功；
- policy loss 只覆盖模型生成的 action token；host observation、图像占位符、工具回执和 teacher action 不进入 GRPO loss；
- Alice 与 Bob 不同时更新。训练 Alice 时固定 Bob policy version，训练 Bob 时固定 Alice/curriculum；
- holdout seeds/snapshots 不进入 proposal 选择和训练。

当前 `_prompt()` 会把 TCP、cube、goal 位置和相对向量写入模型文本。它与“只靠 RGB”并不等价，但属于现有 baseline。engine 迁移阶段应先原样保留以做等价对照；是否去掉这些状态信息应作为后续独立消融实验，不能和 vLLM 迁移混在一次改动中。

## 4. 目标架构

```text
                         round coordinator
                    freeze / select / resume / eval
                      /                       \
          Alice Swift GRPO job          Bob Swift GRPO job
          Alice adapter vA              Bob adapter vB
                 |                            |
          vLLM rollout server           vLLM rollout server
                 |                            |
       embodied Alice scheduler       embodied Bob scheduler
                 |                            |
          simulator worker pool        simulator worker pool
                 |                            |
       snapshot/replay/compiler       goal marker/checker/reward
                 \                            /
                    trusted artifacts store
       snapshot + trajectory + token/mask + receipt + policy version
```

职责边界：

- **OpenETA**：环境 adapter、snapshot、observation projection、动作 parser、predicate compiler/checker、轨迹 artifact、环境 reward receipt；
- **adversarial-mle-agents**：ms-swift 外部插件、Alice/Bob reward ORM、round coordinator、训练启动脚本；
- **ms-swift**：GRPO batch/group、loss、KL/reference policy、importance sampling、LoRA/optimizer/checkpoint、分布式训练；
- **vLLM**：高吞吐模型生成、rollout logprobs、训练权重同步；
- **ManiSkill worker**：每个 trajectory 的有状态物理 session。scheduler 不应把一个 `ManiSkillSimulatorAdapter` 同时分给多个 request。

## 5. 版本和运行模式决策

### 5.1 第一阶段使用 server mode

建议先使用：

```text
swift rlhf process  ──HTTP──>  swift rollout + vLLM process
                                      |
                                simulator workers
```

原因：

- 训练和推理显存边界清楚；
- vLLM rollout server 可以独立重启，环境资源泄漏不会直接杀死 optimizer；
- scheduler、动态图像和 simulator worker 更容易逐层观测；
- Alice 的终局 Bob evaluator 可以作为单独服务固定版本，而不是在同一个训练进程内递归调用另一个模型；
- 与 `adversarial-mle-agents/scripts/run_agent2_swift_grpo.sh` 的已实现路径最接近。

server mode 通过一致性和稳定性门槛后，再评估 Megatron-SWIFT colocate。colocate 的优势是硬件利用率和权重同步路径更紧凑，但它把 trainer、vLLM 和 simulator 的显存/进程生命周期耦合得更深，不适合作为第一步。

### 5.2 先锁版本，再写插件

相邻仓库的真实验证基线是：

- ms-swift `4.4.2`；
- vLLM `0.19.1`；
- Megatron-Core `0.19`；
- mcore-bridge `1.6.2`；
- 模型为 `Qwen3-4B`，H200 上跑过 colocate GRPO。

OpenETA 当前默认模型是 `Qwen3.5-4B`，并且是视觉输入路径。不能因为文本 Qwen3-4B 跑通过，就假设 Qwen3.5-4B 的 vLLM multimodal、chat template、LoRA target 和图像 processor 完全兼容。

实施前必须在目标训练镜像执行一个版本探针，保存输出：

```bash
python - <<'PY'
import inspect
import swift
import vllm
from swift.rollout.multi_turn import MultiTurnScheduler

print("swift:", swift.__version__)
print("vllm:", vllm.__version__)
print("MultiTurnScheduler:", inspect.getsourcefile(MultiTurnScheduler))
print("methods:", sorted(name for name in dir(MultiTurnScheduler) if not name.startswith("__")))
PY
```

ms-swift 当前主线文档中的 scheduler hook 与本地已验证 `4.4.2` 插件的 `check_finished()/step()` 接口可能不同。实现必须针对锁定版本编译和测试，不能混用“官网 main 分支示例”和旧镜像 API。

## 6. 建议的代码拆分

下面保留最初的拆分设计。MVP 最终把 Swift 角色插件合并在 `plugins/embodied_swift_grpo.py`，共享 prompt 放在 `agent/runtime/embodied_selfplay_prompt.py`；生产化时仍可按本节进一步拆分。

### 6.1 OpenETA 内部

建议增加：

```text
OpenETA/
  agent/runtime/embodied_rollout_contract.py
  agent/runtime/embodied_env_session.py
  agent/runtime/embodied_observation_projection.py
  scripts/export_embodied_swift_dataset.py
  scripts/validate_embodied_swift_rollout.py
  tests/test_embodied_rollout_contract.py
  tests/test_embodied_swift_parity.py
```

各文件职责：

- `embodied_rollout_contract.py`：`ALLOWED_ACTIONS`、parse、termination reason、trajectory/receipt schema；不能依赖 Transformers 或 Swift；
- `embodied_env_session.py`：一个 request 对应一个 adapter，负责 reset/restore、step、marker、关闭和超时回收；
- `embodied_observation_projection.py`：把当前 `_prompt()` 和 image extraction 从训练脚本中抽出，保证 Transformers 与 vLLM 使用同一 projection；
- `export_embodied_swift_dataset.py`：生成只包含任务入口和私有环境 metadata 的 JSONL；
- `validate_embodied_swift_rollout.py`：不训练，只运行固定动作/固定种子的 scheduler canary；
- parity tests：比较旧 runner 与新 scheduler 的环境 receipt、reward 和 token mask。

当前 `agent/training/embodied_grpo.py` 中的 action/reward 纯函数可以在抽取期保留兼容 re-export，避免旧脚本立即失效。

### 6.2 adversarial-mle-agents 内部

建议增加：

```text
adversarial-mle-agents/
  plugins/embodied_bob_grpo.py
  plugins/embodied_alice_grpo.py
  scripts/run_embodied_bob_swift_grpo.sh
  scripts/run_embodied_alice_swift_grpo.sh
  scripts/run_embodied_selfplay_rounds.py
  configs/embodied/
```

- Bob 和 Alice 第一版分开插件，避免一个 scheduler 内出现过多角色分支；
- 共享 schema 从 OpenETA import，或发布为两个仓库共同安装的小包；不要复制两套 action/reward 实现；
- `run_embodied_selfplay_rounds.py` 只做编排，不在自身实现 GRPO loss；
- 现有 `mle_agent_grpo.py` 保持不变，防止破坏已经验证的 Agent2 训练。

## 7. Swift 数据集与可见性契约

### 7.1 Bob 输入记录

推荐 JSONL 逻辑结构如下：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nYou are Bob...\nTask: move the cube to the visible green marker..."
    }
  ],
  "images": ["/absolute/path/to/initial_rgb.png"],
  "role": "bob",
  "task_id": "cube-at-...",
  "env_request": {
    "env_id": "PickCube-v1",
    "snapshot_uri": "/absolute/path/to/maniskill-....pt",
    "snapshot_sha256": "...",
    "goal_position": [0.03, -0.04, 0.12],
    "goal_tolerance": 0.025,
    "max_steps": 40,
    "seed": 123
  },
  "policy_context": {
    "alice_policy_version": "alice-round-0003",
    "bob_policy_version": "bob-round-0002",
    "round": 3
  }
}
```

`messages` 和 `images` 是模型可见输入；`env_request` 是 scheduler/host 私有字段。启动训练时必须启用等价于 `--vllm_server_pass_dataset true` 的配置，使 server 收到环境 metadata，但 scheduler 不得把整个 `data_dict` 拼入 prompt。

初始图像既可以在导出时写入，也可以由 scheduler 从 snapshot 恢复后生成。为了避免图像与 snapshot 不一致，正式训练建议由 scheduler 恢复 snapshot 后生成，并校验 observation digest；JSONL 的初始图像只用于 dataset/template canary。

### 7.2 Alice 输入记录

Alice 数据不能提前包含目标位置，因为目标应由 Alice 的真实轨迹创造：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<image>\nYou are Alice. Create a new reachable cube goal state..."
    }
  ],
  "images": ["/absolute/path/to/initial_rgb.png"],
  "role": "alice",
  "proposal_group_id": "round-0003-group-0012",
  "env_request": {
    "env_id": "PickCube-v1",
    "snapshot_uri": "/absolute/path/to/maniskill-....pt",
    "snapshot_sha256": "...",
    "max_steps": 40,
    "min_goal_displacement": 0.04,
    "goal_tolerance": 0.025
  }
}
```

同一 `proposal_group_id` 的 `G` 个 generation 使用相同 snapshot，但必须恢复到 `G` 个互不共享的 session。

### 7.3 图像和消息格式

ms-swift 使用标准 `messages` 数据格式；多模态样本中 `<image>` 的数量要与 `images` 对齐。每个环境 step 之后，scheduler 需要把下一张 observation image 加到下一轮上下文。

当前 ms-swift 文档允许 multi-turn scheduler 通过 `rollout_infos` 动态覆盖 `images/audios/videos`。但锁定的 `4.4.2` 是否具有完全相同字段语义，必须通过版本探针和两轮图像 canary 验证。验收时至少检查：

- 第二轮模型实际收到的是 step 1 之后的图像，不是初始图像；
- `<image>` placeholder 数量与 image list 数量相等；
- server 和 trainer 对相同 image/template 得到相同 prompt token IDs；
- image 文件直到 rollout 完成才被回收；
- RGB dtype、通道顺序、resize/max-pixels 与旧 `AutoProcessor` 路径一致。

不要在 dataset 中序列化完整 numpy 数组。使用内容寻址 PNG/JPEG artifact 或受控的 data URL，并记录 SHA-256。

## 8. Multi-turn scheduler 设计

### 8.1 每个 trajectory 的状态

Bob scheduler 至少维护：

```python
@dataclass
class EmbodiedEpisodeState:
    request_id: str
    role: str
    env_session_id: str
    snapshot_sha256: str
    task_id: str | None
    goal_predicate: dict | None
    step_index: int
    actions: list[str]
    receipts: list[dict]
    image_refs: list[str]
    finished: bool
    termination_reason: str | None
```

还要记录 `round_id`、`group_id`、`generation_id`、`policy_version`、seed 和超时信息，以便恢复和排查。

### 8.2 Bob 生命周期

1. scheduler 收到 request，验证 dataset 私有字段；
2. 从 pool 租用独立 ManiSkill worker；
3. worker 加载 snapshot 并校验 SHA-256；
4. 设置 goal marker，读取初始 observation；
5. projection 生成与旧实现等价的 prompt + image；
6. vLLM 生成一个 action；
7. scheduler 截取第一个合法 action，严格同步修剪 token IDs/logprobs；
8. worker 执行动作并返回下一 observation 与 trusted receipt；
9. scheduler 把新 observation 作为下一轮 user/host 输入，但这些 token 的 loss mask 为 0；
10. 遇到 `DONE`、成功、terminated、truncated、max steps、非法动作上限或 worker error 时结束；
11. checker 计算终局结果，ORM 从 receipt 计算 reward；
12. 写 artifact，并在 `finally` 中关闭/归还 worker、删除 request state。

### 8.3 Alice 生命周期

Alice 前 10 步与 Bob 类似，但结束后多出：

1. host 读取最终 cube position；
2. 从初始 snapshot 重放 Alice actions；
3. 验证最终位置误差、最小位移、边界和动作证据；
4. 量化目标，编译 `cube_at_position`、task ID 和 instruction；
5. 对无效 proposal 直接给 validity-gated reward；
6. 对有效 proposal，提交给固定 `bob_policy_version` 的 evaluator，执行 `K` 次 Bob rollout；
7. 聚合 `BobTaskScore`，再计算 boundary/novelty/repetition reward；
8. 将聚合后的一个标量 reward 返回给该 Alice trajectory。

Alice scheduler 不能在 Python 调用栈里直接递归进入同一个 vLLM engine。推荐使用独立 Bob evaluator service/queue，并设置 deadline、幂等 proposal ID 和结果缓存。缓存键至少包含：

```text
(task_id, snapshot_sha256, bob_policy_version, evaluator_config_hash, eval_seed_set)
```

### 8.4 环境并发模型

当前 `ManiSkillSimulatorAdapter` 明确是 single-env，并保存 `_last_raw_obs`、`_last_info`、gripper 和 snapshot cache，不能并发复用。

第一阶段：

- scheduler 一次只运行一个 simulator session；
- 先证明多轮图像、receipt、mask 和 reward 正确；
- vLLM 可以 batch 模型请求，但环境 step 暂时串行。

第二阶段：

- 使用 bounded worker process pool；
- 每个 active trajectory 独占一个 worker；
- 进程间只传 action、图像引用和小型 receipt，不传 SAPIEN 对象；
- 对 simulator GPU 与 vLLM GPU 设置明确的设备归属；
- worker crash 时该 trajectory 标记 infrastructure failure，不把它误算成 policy failure。

不要只把 adapter 包一层线程锁当作最终并发方案；SAPIEN/GPU context、环境状态和清理生命周期更适合进程隔离。

## 9. 动作约束、token 和 loss mask

### 9.1 动作输出

第一版保持当前裸 action code：

```text
MOVE_X_POS_FINE
```

而不是扩大成自由文本或复杂 JSON。这样容易约束、解析和做 token mask。

vLLM 侧优先级：

1. 如果锁定版本支持逐请求 `choice`/regex structured output，并且 Swift 能透传，直接限制在 `ALLOWED_ACTIONS`；
2. 否则实现 action token trie/logits processor；
3. 如果上述两者在 Swift server path 不可用，允许自由生成，但只截取第一个 action，非法输出执行安全 `DONE` 并施加 penalty。

任何截断都必须同时作用于：文本、`response_token_ids`、rollout logprobs 和 `response_loss_mask`。不能让 vLLM 后续幻觉文本进入 GRPO loss。

### 9.2 loss mask

一个多轮 trajectory 的逻辑 mask：

```text
system/user prompt        0
image tokens              0
model action 1            1
host observation 1        0
model action 2            1
host observation 2        0
...
```

要求：

- `sum(response_loss_mask)` 等于全部受训 action token 数；
- 每个 mask=1 的 token 都有 rollout logprob；
- observation/image/tool receipt 没有 mask=1；
- teacher-forced action 不混入 on-policy GRPO group。

现有 DAgger teacher action 应迁成单独的 `swift sft` 阶段，或明确的 auxiliary SFT job。把 teacher trajectory 混入 GRPO 会破坏 on-policy 假设和 group reward 的含义。

### 9.3 old/reference logprob

当前手写实现：

- old logprob 来自 rollout 时策略；
- reference logprob 来自禁用 LoRA adapter 的 base model；
- 使用 token-level clipped ratio 和 KL。

Swift 配置必须明确复现这三个语义：

- reference 是冻结 base model，还是上一个 checkpoint；
- rollout 权重与 trainer 当前权重的同步频率；
- 是否使用 rollout logprobs 及 token-level importance sampling correction；
- `beta`、`epsilon`、loss type 和 group normalization 方式。

不要只对齐参数名。用一个固定的短 trajectory 比较旧实现和 Swift 的 token IDs、mask、old/reference logprob、advantage 与 loss 数值，容差通过后再训练。

## 10. Reward 插件设计

### 10.1 Bob ORM

Bob reward 插件只读取 scheduler 写入的 `rollout_infos`，例如：

```json
{
  "schema_version": "openeta.embodied_rollout.v1",
  "role": "bob",
  "task_id": "...",
  "snapshot_sha256": "...",
  "valid_action_fraction": 1.0,
  "max_environment_reward": 0.12,
  "ever_grasped": true,
  "success": false,
  "termination_reason": "max_steps",
  "infrastructure_error": null
}
```

迁移第一版应逐项复现 `episode_reward()`：success、grasp、bounded dense 和 invalid penalty。等价验证后，reward shaping 调整才另开实验。

infrastructure failure（worker crash、snapshot 文件损坏、图像上传失败）不能当作普通 `-1` policy reward 静默训练；应丢弃该 sample、重试并单独计数。只有模型非法动作、超步数和真实任务失败才属于 policy outcome。

### 10.2 Alice ORM

Alice reward 信息必须包含：

- proposal validity 和失败原因；
- displacement、goal cell、task ID；
- novelty/repetition；
- 固定 Bob policy version；
- `K` 次 Bob success 列表和 success rate；
- target success rate；
- 最终 boundary reward 各分量。

Alice ORM 不应自己从自由文本 completion 推断 task，也不应信任模型写出的坐标。task 真值来自 simulator + compiler。

### 10.3 分阶段解决 Alice 的嵌套评估

Alice 比 Bob 难，因为 reward 需要额外模型 rollout。建议：

- **A 阶段**：只迁 Bob GRPO；Alice 继续使用旧 runner 或 scripted proposer；
- **B 阶段**：Alice 用 vLLM 生成 proposal，但不训练；离线验证 task 编译和 Bob evaluator；
- **C 阶段**：Alice Swift GRPO 在线调用冻结 Bob evaluator，并把聚合 reward 返回原 trajectory；
- **D 阶段**：外层 coordinator 交替推进 Alice/Bob checkpoint。

不建议把旧 Alice 轨迹导出后直接做“离线 GRPO”来代替 C 阶段，因为训练 policy 与采样 policy 不一致时，普通 GRPO 的 on-policy 语义不成立。

## 11. Alice/Bob 外层训练编排

一个 round 应被拆成可恢复的状态机：

```text
ROUND_START
  -> pin Bob checkpoint/version
  -> Alice proposal rollout + Alice GRPO
  -> validate/replay/compile proposals
  -> select boundary tasks
  -> pin Alice checkpoint/curriculum manifest
  -> Bob GRPO on selected tasks
  -> Bob fixed holdout evaluation
  -> publish round manifest
  -> ROUND_COMPLETE
```

每个状态写原子 manifest，至少包含：

```json
{
  "round": 3,
  "status": "BOB_TRAINING",
  "alice_input_checkpoint": "...",
  "alice_output_checkpoint": "...",
  "bob_input_checkpoint": "...",
  "bob_output_checkpoint": null,
  "bob_evaluator_policy_version": "bob-round-0002",
  "proposal_manifest_sha256": "...",
  "selected_task_manifest_sha256": "...",
  "holdout_manifest_sha256": "...",
  "swift_version": "...",
  "vllm_version": "..."
}
```

恢复时只重跑未完成状态；已经产生且 hash 验证通过的 proposal/Bob evaluation 不重复消耗算力。训练 job 完成后先跑 checkpoint canary，再把 manifest 指针推进到新版本。

Bob 评估得到的 rollout 当前同时会被用于 Bob 训练。迁移时必须明确保留这一策略，或将“课程评分用 evaluation samples”和“Bob on-policy training samples”拆开；不能不记录地改变数据复用方式。

## 12. 命令模板

当前单机双卡应优先使用已经实跑的 `scripts/run_embodied_swift_grpo.sh` 或 `scripts/run_embodied_swift_selfplay_smoke.sh`。以下命令保留为扩展到多 GPU/独立机器时的底层模板，参数名仍须以锁定的 ms-swift 版本 `--help` 为准。

### 12.1 启动 Bob rollout server

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OPENETA_ROOT=/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA \
swift rollout \
  --model /path/to/Qwen3.5-4B \
  --infer_backend vllm \
  --external_plugins /path/to/adversarial-mle-agents/plugins/embodied_bob_grpo.py \
  --multi_turn_scheduler embodied_bob_scheduler \
  --max_turns 40 \
  --vllm_use_async_engine true \
  --vllm_tensor_parallel_size 2 \
  --vllm_max_model_len 16384 \
  --vllm_gpu_memory_utilization 0.80 \
  --host 127.0.0.1 \
  --port 8101
```

### 12.2 Bob GRPO trainer

```bash
CUDA_VISIBLE_DEVICES=2,3 \
NPROC_PER_NODE=2 \
swift rlhf \
  --rlhf_type grpo \
  --model /path/to/Qwen3.5-4B \
  --external_plugins /path/to/adversarial-mle-agents/plugins/embodied_bob_grpo.py \
  --reward_funcs embodied_bob_reward \
  --dataset /path/to/round-0003-bob-train.jsonl \
  --load_from_cache_file false \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port 8101 \
  --vllm_server_pass_dataset true \
  --num_generations 4 \
  --steps_per_generation 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --tuner_type lora \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --beta 0.001 \
  --temperature 1.0 \
  --top_p 1.0 \
  --log_completions true \
  --output_dir /path/to/runs/round-0003/bob
```

还需要按版本验证/补齐：

- vLLM LoRA 动态加载或 weight sync 参数；
- `importance_sampling_level=token` 或对应配置；
- clip epsilon/loss type；
- multimodal `max_pixels`/limit；
- DeepSpeed/Megatron 并行参数；
- rollout server 接收 dataset 私有字段的实际开关；
- scheduler 动态图像字段的实际返回格式。

### 12.3 Alice trainer

Alice 命令形状与 Bob 相同，但使用：

- `embodied_alice_scheduler`；
- `embodied_alice_reward`；
- 独立端口；
- Alice dataset；
- 环境变量或 config 指向冻结 Bob evaluator endpoint 和 `bob_policy_version`。

不要让 Alice 和 Bob trainer 写入同一个 `output_dir`、adapter 目录、optimizer state 或 rollout artifact 目录。

## 13. 分阶段实施计划与退出门槛

### Phase 0：冻结当前基线（MVP 已完成）

工作：

- 固定 8～16 个 train seeds 和独立 holdout seeds；
- 保存旧 runner 的 snapshot、prompt、image hash、action、receipt、reward 和 checkpoint；
- 增加固定动作 replay 测试；
- 记录当前 Transformers processor 输出的 prompt token IDs 和 image shape；
- 保存 Alice 有效、位移不足、越界、replay mismatch 四类样本。

退出门槛：同一代码/seed 重跑时 snapshot digest、固定动作终态、predicate 和 reward 一致。

### Phase 1：抽取共享契约（MVP 已完成）

工作：

- 从训练脚本抽取 action、prompt/image projection、receipt 和 termination schema；
- 旧脚本改为调用共享实现；
- 不接 Swift，不改变训练结果。

退出门槛：旧 runner 回归结果不变；纯函数测试通过。

### Phase 2：Swift scheduler 单轨迹 canary（已完成）

工作：

- 实现 Bob scheduler 和单 worker；
- 用 mock/fixed model action 跑两轮以上；
- 验证动态图像、消息追加、截断、cleanup 和 artifact；
- 暂不训练。

退出门槛：给定同一 snapshot 和固定动作序列，新旧路径的每步 trusted state、终态和 reward 完全一致或在已声明浮点容差内。

### Phase 3：Bob vLLM generation parity（已完成）

工作：

- 接 Qwen3.5-4B vLLM；
- 验证 chat template、禁用 thinking、图像 processor 和 action constraint；
- 对合法/非法/超长输出测试 token 修剪。

退出门槛：100 条短 rollout 无环境串扰、无 image-placeholder 错位、无未清理 request；所有受训 token 都是 action token。

### Phase 4：Bob Swift GRPO（1-step smoke 已完成）

工作：

- 实现 Bob ORM；
- 开启同 snapshot 的 `num_generations >= 2`；
- 对齐 advantage、reference、KL、clip 和 rollout logprob；
- 训练一个短 checkpoint 并恢复。

退出门槛：

- reward 与旧函数逐样本相等；
- 固定短轨迹的 loss 数值通过容差测试；
- 权重更新后 vLLM 确实使用新 adapter；
- checkpoint 恢复后 action 分布和评测可复现；
- holdout 不比随机/迁移前基线出现无法解释的灾难性回退。

### Phase 5：Bob 并发和吞吐（待生产化）

工作：

- 引入 bounded simulator process pool；
- 加 request timeout、worker restart、backpressure 和 metrics；
- 扩大 vLLM batch。

退出门槛：并发 trajectory 的 snapshot/task/action 日志无交叉；故障 trajectory 可重试；串行与并发 reward 分布在统计容差内。

### Phase 6：Alice vLLM proposal（已完成）

工作：

- 实现 Alice scheduler；
- 接 replay/compiler；
- 接冻结 Bob evaluator service；
- 先不更新 Alice。

退出门槛：有效性、goal cell、task ID、K 次 Bob score 和旧 self-play runner 对齐；相同幂等键不重复运行 evaluator。

### Phase 7：Alice Swift GRPO（1-step smoke 已完成）

工作：

- 实现 Alice ORM；
- 固定 Bob policy version 完成一个 Alice train job；
- 将 DAgger 移到独立 SFT job；
- 监控 validity、novelty、Bob success boundary 和 reward 方差。

退出门槛：Alice reward 能反向归因到唯一 proposal；job 期间 Bob 版本不变化；zero-variance group 被显式统计，不产生 NaN。

### Phase 8：完整交替 round（单 round smoke 脚本已完成；多 round/恢复待实现）

工作：

- coordinator 串联两个 Swift job；
- 原子 round manifest、resume、checkpoint canary 和 holdout；
- 至少跑 2～3 个短 round。

退出门槛：中断后可从最近状态恢复；Alice/Bob 版本关系可审计；holdout 与训练任务完全隔离。

### Phase 9：可选 Megatron/colocate

只有 Phase 8 稳定后，才把 server mode 与相邻仓库已经验证的 Megatron-SWIFT colocate 路径结合。先做吞吐/显存 A/B，不再改变 scheduler、reward 或数据契约。

## 14. 必测项目

### 14.1 环境与 snapshot

- snapshot 文件 hash 不匹配时拒绝运行；
- `restore -> fixed actions -> final state` 可复现；
- Alice replay 的 cube position 误差继续使用当前约束；
- 同一 group 的 snapshot digest 相同；
- 不同 generation 的 adapter/session object 不同；
- goal marker 不改变 checker 的任务真值；
- worker crash 后环境资源释放。

### 14.2 prompt 和多模态

- 新旧 projection 文本逐字符一致；
- Qwen template 和 `enable_thinking=False` 等价；
- 每轮 `<image>` 与 `images` 数量相等；
- 第 N 轮使用第 N 次 observation；
- evaluator 私有字段不会出现在 tokenized prompt；
- Alice trajectory 不会进入 Bob prompt。

### 14.3 token / GRPO

- legal action 单 token和多 token 两种情况；
- EOS 是否进入 loss 的策略固定；
- vLLM 超额文本截断后 token/logprob/mask 等长；
- host observation mask 全为 0；
- group reward/advantage 与旧实现对齐；
- group 全同 reward 时稳定处理；
- reference base policy 与 LoRA adapter 语义正确；
- rollout/trainer 权重版本被记录并一致。

### 14.4 Self-Play

- 无效 Alice proposal 不调用 Bob evaluator；
- 有效 proposal 恰好产生配置的 `K` 次 Bob rollout；
- Bob evaluator checkpoint 在整批 Alice rollout 中固定；
- novelty/repetition 使用持久化 goal-cell history；
- selected tasks 与排序/上限和旧逻辑一致；
- holdout snapshot 从未被写入训练 dataset。

## 15. 观测、日志和 artifact

每条 trajectory 必须能通过以下 ID 串起来：

```text
run_id / round_id / role / group_id / generation_id / request_id
snapshot_sha256 / task_id / policy_version / rollout_engine_version
```

至少保存：

- 原始可见 prompt 和独立的 private metadata 摘要；
- 每轮 image hash，不一定永久保存所有原图，但失败样本必须保留；
- raw completion、截断后 action、token IDs、rollout logprobs 和 loss mask；
- 每步 simulator receipt、termination reason；
- reward 分量，不只保存总分；
- Swift/vLLM/Transformers/ManiSkill/PyTorch 版本；
- base model、Alice adapter、Bob adapter 的路径和 hash；
- scheduler config hash 和 evaluator config hash；
- infrastructure retry 次数和最终归类。

关键 metrics：

- rollout latency：model、image preprocess、env step、Bob nested eval 分开；
- active/pending simulator sessions；
- invalid action rate；
- snapshot/replay mismatch rate；
- Alice proposal validity/novelty/repetition；
- Bob success/grasp/max-step rate；
- group reward std 和 zero-variance group rate；
- vLLM/trainer policy version lag；
- tokens/s、trajectories/hour、GPU memory 和 worker restart。

## 16. 主要风险与处理方式

| 风险 | 表现 | 处理 |
|---|---|---|
| 动态图像未更新 | 模型每轮看到初始画面 | 两轮 image-hash canary；检查锁定 Swift API |
| 环境串扰 | 一个 generation 的动作改变另一个 generation | 每 request 独占 session；并发隔离测试 |
| vLLM 权重陈旧 | trainer loss 在变，rollout 行为不变 | 记录 policy version；每次同步后固定 prompt action-distribution canary |
| 截断不一致 | 文本只保留 action，但多余 token 仍参与 loss | 同时裁剪文本、IDs、logprobs、mask |
| reward 泄漏 | Bob prompt 含 goal predicate/snapshot/Alice trajectory | model-visible/private schema 分离；tokenized prompt 审计 |
| 物理失败被当作策略失败 | worker crash 大量产生负奖励 | infrastructure error 单独重试/丢弃 |
| Alice nested eval 死锁 | Alice scheduler 占满 worker 等 Bob | 独立 Bob service/pool、deadline、队列和幂等缓存 |
| 显存冲突 | vLLM、trainer、SAPIEN 抢同一 GPU | 明确 CUDA 设备；先 server mode；环境 worker 限流 |
| GRPO group 方差为零 | advantage 全 0 或数值异常 | 统计并跳过/按 Swift 明确策略处理，不能静默 NaN |
| Qwen3.5 VLM 不兼容 | vLLM 启动或图像 template 失败 | 模型能力探针；不要用 Qwen3 文本验证替代 |
| API 版本漂移 | scheduler hook/返回字段不同 | 锁 wheel/镜像；保存 API probe；针对固定版本测试 |
| DAgger 污染 on-policy | teacher action 被当作采样 token | 独立 SFT job，GRPO mask 不覆盖 teacher token |

## 17. 回滚策略

在 Phase 8 通过前：

- 保留 `scripts/train_embodied_grpo.py` 和 `scripts/run_embodied_selfplay.py`；
- 增加 backend 选择，而不是原地删除旧路径，例如 `rollout_backend=transformers|swift_vllm`；
- snapshot、task、trajectory schema 保持向后可读；
- 每个 Swift checkpoint 发布前用旧 holdout runner 做独立评估；
- 如果 Swift job 失败，round manifest 不推进 checkpoint 指针；
- engine 回滚只切换模型推理/训练 backend，不回滚已经验证的环境、predicate 或 snapshot 数据。

完成迁移的定义不是“Swift 命令能启动”，而是新路径通过 parity、恢复、holdout 和至少两个交替 round 的门槛。达到该条件后，旧 trainer 才可以降级为测试工具。

## 18. 推荐 PR 顺序

1. **PR-1：共享 rollout contract**
   只抽取 action、projection、receipt、termination 和 artifact schema；旧行为不变。
2. **PR-2：Bob scheduler canary**
   单 worker、fixed actions、无训练，多轮图像与 mask 测试。
3. **PR-3：Bob vLLM + Swift GRPO**
   Bob ORM、数据导出、server/train 脚本、loss/reward parity。
4. **PR-4：simulator worker pool**
   并发、超时、cleanup、基础设施错误分类。
5. **PR-5：Alice proposal scheduler**
   replay/compiler、冻结 Bob evaluator、只评估。
6. **PR-6：Alice Swift GRPO**
   boundary reward、幂等缓存、独立 DAgger SFT。
7. **PR-7：round coordinator**
   交替冻结、manifest/resume、checkpoint canary、holdout。
8. **PR-8：性能优化**
   Megatron/colocate、批量参数和吞吐优化；不改变任务语义。

每个 PR 都应能单独回滚，并附一份机器可读 canary receipt。

## 19. 本次 MVP 对五个关键问题的取值

1. 锁定 ms-swift `4.4.2` 和 vLLM `0.19.1`，插件按该 wheel 的 `MultiTurnScheduler` API 实现；
2. Qwen3.5-4B 的视觉输入、chat template、server weight sync 和 LoRA checkpoint reload 均已实跑；vLLM reload 会忽略 vision tower LoRA，而当前训练冻结 ViT，语言/控制侧 LoRA 可以正常加载；
3. MVP 中 ManiSkill session 与 vLLM rollout 同进程，rollout 放 GPU0、trainer 放 GPU1；生产扩展再拆 simulator process pool；
4. Alice 的 `K` 次 Bob evaluator 使用同一 authoritative snapshot 的随机策略采样，receipt 记录固定 Bob policy version；跨 snapshot 泛化留给独立 holdout；
5. 为保持旧 baseline 的 observation policy，Bob 暂时仍看到精确 cube/goal/TCP projection；这不是 Alice 轨迹泄漏，后续可作为纯 RGB 消融单独移除。

这些选择只证明当前 MVP 可运行。启动长时间训练前仍需配置持久化 evaluator cache、holdout manifest、更多 generation 和资源监控。

## 20. 本地代码依据

- `OpenETA/agent/training/embodied_grpo.py`：当前 action、trusted reward、group advantage 和手写 clipped GRPO；
- `OpenETA/scripts/train_embodied_grpo.py`：Transformers 多模态生成、prefix action constraint、逐轨迹 logprob 和 LoRA 更新；
- `OpenETA/scripts/run_embodied_selfplay.py`：Alice proposal、replay/compiler、K 次 Bob 评估、课程选择、交替更新和 holdout；
- `OpenETA/adapter/maniskill_sim.py`：单环境 adapter、完整 snapshot/hash、goal marker 和 trusted task state；
- `OpenETA/agent/runtime/embodied_task_compiler.py`：`cube_at_position` 的位移、边界、量化和 task ID；
- `OpenETA/agent/runtime/embodied_selfplay_prompt.py`：角色隔离后的 Alice/Bob prompt；
- `OpenETA/plugins/embodied_swift_grpo.py`：已实跑的 Swift scheduler、ORM 和冻结 Bob evaluator；
- `OpenETA/scripts/run_embodied_swift_grpo.sh`：单角色 rollout/trainer 入口和 adapter/checkpoint 参数；
- `OpenETA/scripts/run_embodied_swift_selfplay_smoke.sh`：双卡 Bob→Alice 单轮编排；
- `adversarial-mle-agents/plugins/mle_agent_grpo.py`：已经存在的 ms-swift multi-turn scheduler/ORM 插件形状；
- `adversarial-mle-agents/scripts/run_agent2_swift_grpo.sh`：vLLM server mode 启动形状；
- `adversarial-mle-agents/scripts/run_agent2_megatron_grpo.sh`：Megatron-SWIFT colocate 启动形状；
- `adversarial-mle-agents/docs/agent2_rl_validation.md`：已验证的训练软件栈；
- `adversarial-mle-agents/docs/embodied_asymmetric_self_play_plan.zh-CN.md`：Self-Play 总体角色、reward 和课程设计。

## 21. 官方接口参考

- [ms-swift GRPO 文档](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md)
- [ms-swift Multi-turn GRPO 开发指南](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/multi_turn.md)
- [ms-swift 自定义数据集与多模态 messages 格式](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md)
- [vLLM Structured Outputs](https://docs.vllm.ai/en/v0.11.2/features/structured_outputs/)

这些在线文档反映的是对应页面当前版本；最终实现仍必须以锁定 wheel/镜像中的 `--help`、Python signature 和最小 canary 为准。
