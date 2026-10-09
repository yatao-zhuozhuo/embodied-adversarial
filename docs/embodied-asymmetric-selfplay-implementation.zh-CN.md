# Alice/Bob 具身 Asymmetric Self-Play：实现、阅读与运行指南

## 1. 这套系统要解决什么

目标是在同一个具身环境中交替训练两个角色：

- **Alice** 从一个可重放的初始状态出发，通过真实动作构造一个新的、可完成的目标状态；
- **Bob** 从与 Alice 完全相同的初始状态出发，只看到任务描述和目标，不看到 Alice
  的轨迹，尝试复现目标；
- host 使用模拟器状态判断 Alice 的任务是否有效、Bob 是否成功；
- Alice 从 Bob 的聚合成功率获得课程奖励，Bob 从任务完成度获得策略奖励；
- Alice 和 Bob 共享一个基座模型，但分别维护自己的 LoRA adapter。

核心 round 是：

```text
初始 simulator snapshot
  → Alice rollout
  → Alice 最终状态编译成 goal predicate 和 instruction
  → validity / replay / novelty gate
  → Bob 从相同 snapshot 独立尝试 K 次
  → Bob success rate
  → Alice boundary reward
  → 更新 Alice LoRA
  → 在有效边界任务上更新 Bob LoRA
  → 固定 holdout seeds 评测并保存 round state
```

## 2. 代码由哪两个仓库组成

### OpenETA

路径：

```text
/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA
```

负责：

- ManiSkill 环境创建和动作执行；
- RGB-D、机器人状态和对象状态；
- simulator snapshot 的保存、校验和恢复；
- Alice/Bob 多模态 rollout；
- trusted goal checker；
- GRPO token loss、LoRA 更新和 checkpoint；
- 视频、轨迹和评测 artifact；
- self-play round coordinator。

### adversarial-mle-agents

路径：

```text
/inspire/qb-ilm/project/exploration-topic/ky26060/adversarial-mle-agents
```

负责：

- `EmbodiedTaskSpec` 公共任务协议；
- `BobTaskScore`；
- Alice 的能力边界 reward；
- novelty/repetition 课程选择；
- 与其它 adversarial curriculum 实验保持一致的数据边界。

两个仓库通过 JSON artifact 和小型 Python contract 交接，不共享 simulator 对象。

## 3. 环境来自哪里

当前环境来自官方 `ManiSkill 3.0.1`：

```text
Python package:
  /opt/py312/lib/python3.12/site-packages/mani_skill

Task implementation:
  mani_skill/envs/tasks/tabletop/pick_cube.py
```

底层物理与渲染使用 `SAPIEN 3.0.3`。OpenETA 使用 Gymnasium 注册名
`PickCube-v1` 创建环境。

当前任务配置：

| 配置 | 值 |
|---|---|
| robot | Franka Panda |
| task | `PickCube-v1` |
| cube | 边长 4 cm 的红色方块 |
| goal tolerance | 2.5 cm |
| simulator frequency | 100 Hz |
| control frequency | 20 Hz |
| adapter episode horizon | 120 actions（覆盖上游默认 50） |
| observation | RGB-D + qpos/qvel + TCP + object/goal state |
| control mode | `pd_ee_delta_pose` |
| reward mode | `normalized_dense` |
| default sensor camera | `base_camera` |
| human render camera | `render_camera`，512×512 |

ManiSkill 的原始 dense reward 包含：接近方块、抓住方块、把方块移向目标、到达目标后
保持静止；成功时原始 reward 为 5，`normalized_dense` 将其除以 5。

## 4. OpenETA 环境适配层

主要文件：

```text
adapter/protocol.py
adapter/sim.py
adapter/maniskill_sim.py
agent/runtime/embodied_snapshot.py
agent/runtime/embodied_goal.py
```

### Observation

`EnvObservation` 包含：

- `cameras`：RGB 和 depth；
- `robot`：关节位置、速度、末端位姿和夹爪状态；
- `objects`：方块与目标的可验证位置；
- `metadata`：环境 ID 和可信 success 字段。

模型当前使用 RGB、任务文本、step index 和 TCP xyz。环境仍保存 depth 和完整机器人状态，
后续可以在不改变 simulator contract 的情况下加入模型 prompt。

### Atomic action

动作 vocabulary 有粗、细两档：

```text
MOVE_X_POS / MOVE_X_NEG
MOVE_Y_POS / MOVE_Y_NEG
MOVE_Z_POS / MOVE_Z_NEG
MOVE_X_POS_FINE / MOVE_X_NEG_FINE
MOVE_Y_POS_FINE / MOVE_Y_NEG_FINE
MOVE_Z_POS_FINE / MOVE_Z_NEG_FINE
GRASP / RELEASE / DONE
```

粗/细 action 向 `pd_ee_delta_pose` controller 分别提交 5 cm / 1 cm 的 target delta；它们不是
“单步一定移动这么远”的位置瞬移，真实位移由 20 Hz controller 和动力学决定。靠近方块和
目标时必须使用 `*_FINE`。

夹爪使用 latch。Panda controller 的真实约定是 `+1=open`、`-1=closed`；因此 `GRASP`
写入 `-1`，后续 MOVE 继续保持闭合，直到 `RELEASE` 写入 `+1`。这是这次实现中修掉的
关键 bug：旧代码将正负号写反，导致看似执行了 GRASP、实际上一直张开。

### Snapshot

snapshot 保存：

- ManiSkill 完整 `state_dict`；
- elapsed steps；
- host 侧夹爪 latch；
- SHA-256 内容摘要；
- `.pt` 持久化路径。

Bob 必须恢复这个 snapshot。只使用相同 seed 不构成严格等价证明。

## 5. 任务如何由 Alice 产生

MVP 使用 `cube_at_position` predicate：

```json
{
  "type": "cube_at_position",
  "position": [0.03, -0.04, 0.12],
  "tolerance": 0.025
}
```

流程：

1. 保存初始 snapshot 和初始 cube position；
2. Alice 接收“创造一个新的可达方块目标状态”的角色 prompt；
3. Alice 在环境中真实执行动作；
4. host 读取最终 cube position；
5. 如果位移小于阈值、越界、没有真实操作证据或 replay 不一致，proposal 无效；
6. 有效最终位置被量化并编译成 predicate、task ID 和 instruction；
7. 恢复初始 snapshot，将绿色 goal marker 设置到 Alice 的最终位置；
8. Bob 只看到目标 marker、instruction 和自己的 observation。

Alice trajectory 是 feasibility proof，不进入 Bob prompt。

## 6. Bob 如何评测

每个有效 Alice task 使用同一个初始 snapshot 独立采样 `K` 条 Bob trajectory。随机性来自
策略 sampling；物理初态保持相同。

```text
success_rate = successful_episodes / K
```

单条 Bob trajectory 的 reward 来自 simulator receipt：

- `cube_at_position` / ManiSkill success；
- grasp progress；
- normalized dense progress；
- 非法 action penalty；
- 可选 step penalty。

模型自报的 `success`、reasoning 或自然语言不参与真值判断。

## 7. Alice reward

课程 reward 实现在：

```text
adversarial-mle-agents/src/mle_agent_rl/embodied_curriculum.py
```

公式：

```text
R_alice = validity_gate × (
    boundary_score(Bob success rate, target=0.45)
    + 0.25 × novelty
) - repetition_penalty
```

- 无效或不可完成任务：`-1`；
- Bob 总失败和总成功都不是最优；
- Bob 成功率接近 0.45 的任务得到最高 boundary reward；
- 近期重复目标扣 0.15。

`K=4` 时 Bob success rate 为 `0/0.25/0.5/0.75/1.0`，其中 0.5 最接近目标边界。

## 8. GRPO 实现

主要文件：

```text
agent/training/embodied_grpo.py
scripts/train_embodied_grpo.py
scripts/run_embodied_grpo_smoke.sh
```

每个 prompt group 从同一个 snapshot 采样多条 trajectory：

```text
rewards
  → group mean/std
  → standardized advantages
  → current/old policy token log-prob ratio
  → clipped policy objective
  → frozen base/reference KL
  → LoRA optimizer step
```

loss 只覆盖模型生成的 completion token。图像 prompt、host observation 和环境 receipt 不计入
policy loss。reference policy 通过临时关闭 LoRA 获得，因此不需要在显存中复制一份 4B 模型。

策略 completion 只输出一个裸 action code，例如 `MOVE_X_NEG_FINE`，不输出 JSON 或
reason。此前 JSON 公共前缀主导了 SFT loss，模型学会格式却没有学会动作分类；裸 code 让
completion loss 直接监督动作选择。parser 为兼容旧 artifact 仍能读取旧 JSON。

Alice 冷启动使用显式 DAgger 退火。`teacher_forced=true` 的 step 只进入监督 action loss，
不进入 GRPO on-policy loss；模型自己采样的 step 才进入 GRPO。Bob 从始至终不使用 teacher，
始终是独立 rollout + GRPO。

Alice 和 Bob 使用相同的 `Qwen3.5-4B` base，各自维护：

```text
runs/selfplay/round-XXXX/alice_adapter/
runs/selfplay/round-XXXX/bob_adapter/
```

## 9. 推荐正式配置

### 配置文件

代码内有两套配置：

```text
configs/embodied_selfplay_smoke.json  # 1 round，验证链路
configs/embodied_selfplay_pilot.json  # 3 rounds，正式小规模 pilot
configs/alice_dagger_curriculum.json  # Alice-only 在线 DAgger 与固定 seed 评估
```

正式 pilot 的核心值：

```yaml
environment:
  env_id: PickCube-v1
  camera_resolution: 128
  translation_step_m: 0.05
  fine_translation_step_m: 0.01
  episode_horizon: 120
  train_seed_start: 0
  holdout_seeds: [10000, 10001, 10002, 10003]

model:
  base: /inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B
  dtype: bfloat16
  lora_rank: 8
  lora_alpha: 16

grpo:
  group_size: 4
  learning_rate: 5.0e-6
  clip_epsilon: 0.2
  beta: 0.01
  alice_temperature: 0.3
  alice_top_p: 0.9
  bob_temperature: 1.0
  bob_top_p: 0.95

selfplay:
  rounds: 3
  proposal_groups_per_round: 2
  bob_evaluations_per_task: 4
  selected_tasks_per_round: 4
  alice_max_steps: 60
  bob_max_steps: 40
  holdout_max_steps: 40
  alice_oracle_mix_schedule: [1.0, 0.5, 0.0]
  target_bob_success_rate: 0.45
```

`group_size=4` 表示每个 Alice initial snapshot 采 4 条 proposal；每个有效 proposal 再让 Bob
独立尝试 4 次。因而这不是便宜的配置，应先过下一节的 smoke 与 warm-start。

## 10. 从零启动的推荐顺序

### A. 先验证环境、任务编译、Bob GRPO

使用脚本 Alice，真实 Bob 模型仍会 rollout 和更新：

```bash
cd /inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA
python scripts/run_embodied_selfplay.py \
  --config configs/embodied_selfplay_smoke.json \
  --oracle-alice \
  --round-limit 1
```

本机已完成的结果位于 `runs/embodied_selfplay_smoke/round-0000`：

- 2/2 Alice proposals 有效，轨迹 replay 一致；
- Alice 在 seed 0 用 44 个 action 完成真实抓取搬运；
- 最终位置 replay 误差约 `6.9e-7 m`；
- Bob reward std `0.004286`，GRPO advantage 非零；
- Bob update loss `-0.13960`，gradient norm `1.14097`；
- LoRA 的 104 个 `lora_B` tensor 全部非零；
- 冷启动 Bob holdout success 仍为 0，这个 smoke 不声称策略已经学会任务。

### B. 分别 warm-start Alice 和 Bob

纯随机 Alice 几乎不能生成有效 goal，所有 reward 都为 `-1` 时没有 GRPO 学习信号。因此先用
可信 scripted controller 做少量 SFT；脚本动作仍在真实 ManiSkill 中执行，每条 demo 都要求
simulator success。

两张卡可以并行：

```bash
python scripts/bootstrap_embodied_sft.py \
  --role alice --device 0 --demonstrations 4 --epochs 2 \
  --output-dir runs/bootstrap_alice_relative &

python scripts/bootstrap_embodied_sft.py \
  --role bob --device 1 --demonstrations 4 --epochs 2 \
  --output-dir runs/bootstrap_bob_relative &

wait
```

最小实现验证已经用 1 条 Alice demo 完成 43 个 completion sample 的反传与 adapter 保存；正式
pilot 推荐 4 条不同 seed、2 epochs。只用这 1 条 demo 做过一次真实双模型检查，Alice 的
有效 proposal 是 0/2，所以该最小 checkpoint 只证明 SFT 链路可用，不应拿来启动正式训练。
scripted controller 已另外验证 seeds 0–3 均能在 60 steps 内得到 simulator success。

### C. 跑真实双模型 + DAgger self-play smoke

```bash
python scripts/run_embodied_selfplay.py \
  --config configs/embodied_selfplay_smoke.json \
  --output-dir runs/embodied_selfplay_dagger_smoke \
  --alice-adapter runs/bootstrap_alice_relative/adapter \
  --bob-adapter runs/bootstrap_bob_relative/adapter \
  --round-limit 1
```

smoke 配置的 `alice_oracle_mix_schedule=[1.0]`，因此这一轮 Alice 全 teacher-forced，但 Alice
模型确实加载、在访问到的 state 上做 DAgger 更新并保存；Bob 始终是纯模型 GRPO。已完成的
本机结果：

- Alice proposals 2/2 有效，replay 一致，每条 43 steps；
- Alice DAgger 86 samples，loss `1.3166`，gradient norm `10.8124`；
- Alice GRPO 明确记录 `skipped_no_on_policy_steps=1`，没有把 teacher action 冒充 on-policy；
- Bob GRPO 16 samples，reward std `0.003789`，gradient norm `1.1592`；
- Alice/Bob 两个新 adapter 和 optimizer state 均已保存。

纯模型 Alice 的检查也做过：少量离线 SFT 后仍是 0/2 valid proposal，主要表现为动作模式
坍缩且没有 GRASP。因此当前的绿色状态是“DAgger 退火 self-play 可运行”，不是“model-only
Alice 已学会”。这也是正式配置必须从 1.0 → 0.5 → 0.0 退火的原因。

### D. 先把 Alice 提升到 model-only 可产生有效任务

Alice-only 入口是 `scripts/train_alice_dagger.py`。它与离线 SFT 的区别是：Alice 在真实环境
访问状态，oracle 在每个访问到的状态提供动作标签；即使该步实际执行的是模型动作，偏离后的
状态也会进入 DAgger 数据。监督样本按 `control_phase` 均衡，避免 `GRASP`、抬升和目标对准
阶段被大量移动 token 淹没。固定的 seeds 100–103 只做 `oracle_mix=0` 评估，不进入训练。

首次启动：

```bash
python scripts/train_alice_dagger.py \
  --initial-adapter runs/embodied_selfplay_dagger_smoke/round-0000/alice_adapter \
  --initial-optimizer runs/embodied_selfplay_dagger_smoke/alice_optimizer.pt \
  --round-limit 1
```

同一命令可以恢复；当 `runs/alice_dagger_curriculum/state.json` 已存在时，以其中的 adapter、
optimizer 和 `next_round` 为准。每次只加一轮便于审计。promotion 同时要求 model-only
`valid_rate >= 0.25` 和 `grasp_rate >= 0.25`，否则不降低 teacher 比例；仅靠推方块产生位移
不会触发退火。配置保留到 12 rounds；建议始终带 `--round-limit 1`，逐轮检查后再继续。

本机排查并修复了两个会破坏训练语义的环境边界问题：

- ManiSkill 单环境 flag 原为 `[False]`，`bool([False])` 会被 Python 判为真。adapter 现在将
  `success/is_grasped/is_obj_placed` 统一归一成标量 `bool`；
- SAPIEN contact impulse 不在 `get_state_dict()` 中。成功抓取后直接 restore 会让第一帧
  `is_grasping()` 沿用旧接触力；restore 现在用隐藏 no-op 刷新接触缓存，再写回精确 state 和
  elapsed step。

修复前 round 0–4 的 prompt 抓取状态语义不可信，不能用来证明 Alice 能力。修复后的结果：

- round 5：阶段标签首次完整覆盖抓取前后；model-only `grasp_rate=0.25`、`valid_rate=0`；
- round 6：252 个阶段均衡样本、32 optimizer steps；model-only `grasp_rate=0.75`、
  `valid_rate=0.25`，平均位移 `0.0295 m`；
- seed 103 在第 49 步自主 `GRASP`，之后 20 步持续持有并抬升，最终位移 `0.0743 m`，replay
  一致；这是首个纯模型有效 Alice proposal；
- promotion 条件满足，round 7 将 `oracle_mix` 从 `0.9` 降到 `0.8` 后，model-only 仍保持
  `grasp_rate=0.75`、`valid_rate=0.25`；下一轮 mix 已自动推进到 `0.7`。

当前可恢复 checkpoint：

```text
runs/alice_dagger_curriculum/round-0007/alice_adapter
runs/alice_dagger_curriculum/alice_optimizer.pt
runs/alice_dagger_curriculum/state.json  # next_round=8, oracle_mix=0.7
```

详细证据分别在 round 目录的 `metrics.json`、`train_trajectories.json` 和
`model_only_eval.json`。当前结论是 Alice 已能产生非零有效任务，并通过了 0.8 teacher 的一轮
稳定性验证，但 1/4 有效率仍偏低；可以用这个 adapter 启动小规模双 agent pilot，同时继续
Alice 0.7 teacher 的课程训练。

### E. 启动正式 3-round pilot

```bash
python scripts/run_embodied_selfplay.py \
  --config configs/embodied_selfplay_pilot.json \
  --output-dir runs/embodied_selfplay_pilot \
  --alice-adapter runs/bootstrap_alice_relative/adapter \
  --bob-adapter runs/bootstrap_bob_relative/adapter
```

脚本在每个完整 round 后原子更新 `state.json`。同一命令重启时会从 `next_round` 和最近的
Alice/Bob adapter、optimizer state 继续；不要删除 `state.json` 后复用相同输出目录。

每轮查看 `alice_oracle_mix`：round 0 是 DAgger bootstrap，round 1 同时含 DAgger 与合法的
on-policy GRPO，round 2 才是 model-only Alice。若 round 1 的 `valid_proposal_count` 已降为 0，
应把 schedule 放慢，例如 `[1.0, 0.75, 0.5]`，而不是继续到 0。

### F. 用训练后 Alice 参数启动 n=16 双 agent GRPO

当前正式入口使用 `configs/embodied_selfplay_grpo16.json`。Alice 和 Bob 都从同一个已训练
Alice checkpoint 初始化，但随后各自保存 adapter 和 optimizer，参数从 round 0 更新后开始
分叉：

```text
runs/alice_dagger_curriculum/round-0007/alice_adapter
  ├── initialize Alice
  └── initialize Bob
```

这里的 `n=16` 同时落在两个实际采样组：

- `grpo.group_size=16`：同一 initial snapshot 下采 16 条 Alice proposal，用 16 个 Alice
  boundary reward 计算组内 advantage；
- `selfplay.bob_evaluations_per_task=16`：每个有效 Alice task 从同一 snapshot 独立采 16 条
  Bob trajectory，用成功率评价任务，并用 16 个 Bob reward 做 Bob GRPO。

Alice 已有非零 model-only 有效率，因此本配置使用 `alice_oracle_mix_schedule=[0,0,0]`，只有
真实模型 action 进入 Alice GRPO，不再混入 teacher。视觉 GRPO update 已改为逐 sample 反传
mean-loss contribution；这与整组 loss 一次反传数学等价，但不会为 n=16 的全部轨迹同时保留
计算图。rollout 使用 action-trie 约束解码：模型仍按自身 logits 采样，但每个 completion 只能是
16 个合法 action code 之一。这样可以提高温度获得组内探索，而不会产生任意文本或越过环境
action allowlist。当前 Alice/Bob temperature 分别为 `2.0/1.5`，`top_p=1.0`。

启动或从完整 round 恢复：

```bash
python scripts/run_embodied_selfplay.py \
  --config configs/embodied_selfplay_grpo16.json \
  --output-dir runs/embodied_selfplay_grpo16 \
  --alice-adapter runs/alice_dagger_curriculum/round-0007/alice_adapter \
  --bob-adapter runs/alice_dagger_curriculum/round-0007/alice_adapter \
  --round-limit 1
```

首轮使用 1 个 initial snapshot、最多选择 2 个 task 更新 Bob；Alice/Bob 最大 80 steps，
相机 64×64。n=16 的完整 round 成本较高，仍建议每次只运行一轮并检查 reward std、有效任务
数、Bob success rate 和 holdout 后再继续。

## 11. 两张 GPU 如何使用

当前节点为 2×48GB RTX 4090：

- GPU 0：Alice policy rollout/update；
- GPU 1：Bob policy evaluation/update；
- ManiSkill physics 当前运行在 CPU，渲染通过 SAPIEN/Vulkan；
- 每个 4B base + LoRA 实测峰值约 10GB，因此显存足够；
- round 内存在依赖关系，先追求正确恢复和可审计 artifact，再加入异步流水线。

## 12. Artifact 与恢复

正式目录：

```text
runs/embodied_selfplay/
├── config.json
├── state.json
├── alice_optimizer.pt
├── bob_optimizer.pt
├── snapshots/
├── round-0000/
│   ├── proposals/
│   │   └── group-00-gen-00/
│   │       ├── alice_trajectory.json
│   │       ├── task.json
│   │       ├── bob_scores.json
│   │       ├── bob_trajectories.json
│   │       └── curriculum_feedback.json
│   ├── alice_adapter/
│   ├── bob_adapter/
│   ├── metrics.json
│   └── videos/
```

`state.json` 至少记录：当前 round/phase、Alice/Bob adapter 路径、optimizer step、训练和
holdout seed、已见 goal cells。当前恢复粒度是“完整 round”：round 内中断会保留已有文件，
但重启会重做该 round；只有 `state.json` 已提交的 round 才算 checkpoint。

## 13. 如何看轨迹

已有 GRPO rollout：

```bash
python3 scripts/render_embodied_grpo_rollouts.py \
  runs/embodied_bob_grpo_multistep/rollouts.jsonl
```

默认生成第三人称 ManiSkill MP4、GIF 和 contact sheet。使用 `--view agent` 可以导出模型
实际看到的 RGB observation。

正式 self-play round 用：

```bash
python scripts/render_embodied_selfplay_round.py \
  runs/embodied_selfplay_dagger_smoke/round-0000
```

Alice-only DAgger round 用：

```bash
python scripts/render_alice_dagger_round.py \
  runs/alice_dagger_curriculum/round-0007 --seed 103
```

不指定 `--seed` 时优先选择该 round 的第一条有效轨迹，其次选择第一条抓取轨迹。当前纯模型
有效轨迹已生成到：

```text
runs/alice_dagger_curriculum/round-0007/videos/model_only_eval-seed-103.mp4
runs/alice_dagger_curriculum/round-0007/videos/model_only_eval-seed-103.gif
```

它默认找到该 round 被选中的 proposal，并从持久化 snapshot 分别重放 Alice 和全部 Bob
trajectory。当前已生成的 DAgger self-play 真实视频：

```text
runs/embodied_selfplay_dagger_smoke/round-0000/videos/group-00-gen-00/alice.mp4
runs/embodied_selfplay_dagger_smoke/round-0000/videos/group-00-gen-00/bob-00.mp4
runs/embodied_selfplay_dagger_smoke/round-0000/videos/group-00-gen-00/bob-01.mp4
```

阅读单个 round 时按以下顺序：

```text
metrics.json
  → task.json
  → alice_trajectory.json
  → bob_scores.json
  → bob_trajectories.json
  → curriculum_feedback.json
  → videos/*.mp4
```

## 14. 推荐代码阅读顺序

第一遍建立主流程：

1. `scripts/run_embodied_selfplay.py`：round coordinator；
2. `adapter/maniskill_sim.py`：环境、粗/细动作、夹爪、snapshot 和 goal；
3. `scripts/bootstrap_embodied_sft.py`：可信轨迹 warm-start；
4. `scripts/train_embodied_grpo.py`：rollout、completion log-prob 与 update；
5. `agent/training/embodied_grpo.py`：reward、advantage 和 clipped loss；
6. `agent/runtime/embodied_task_compiler.py`：Alice final state 到 goal；
7. `agent/runtime/embodied_goal.py`：trusted checker；
8. `adversarial-mle-agents/src/mle_agent_rl/embodied_task.py`：任务协议；
9. `adversarial-mle-agents/src/mle_agent_rl/embodied_curriculum.py`：Alice reward；
10. `scripts/render_embodied_selfplay_round.py`：snapshot replay 与视频。

第二遍沿一个 artifact 反向追踪：

```text
curriculum_feedback.json
  → BobTaskScore
  → Bob trajectories
  → restored initial snapshot
  → Alice compiled goal
  → Alice trajectory
  → original reset seed
```

## 15. 启动长期训练前的验收门

必须先满足：

- `GRASP → MOVE` 能持续持有物体；
- Alice 和 Bob 的初始 snapshot hash 完全一致；
- Alice proposal 有可测的初始/目标差异；
- Alice trajectory replay 能再次达到 compiled goal；
- Bob 不读取 Alice trajectory；
- 每组 reward 有方差，GRPO advantage 不是长期为零；
- `teacher_forced` step 不进入 GRPO，oracle mix 能逐轮下降；
- model-only round 的 Alice proposal 有效率不为零；
- 固定 holdout seeds 不参与训练；
- Alice/Bob adapter 和 optimizer state 可恢复；
- 每个 round 都能导出视频和完整 JSON evidence。

满足这些条件后，才将 `rounds`、proposal 数量和训练 iterations 扩大。
