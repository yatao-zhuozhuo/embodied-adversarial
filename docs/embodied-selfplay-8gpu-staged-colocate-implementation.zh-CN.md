# OpenETA 具身 Self-Play：8 卡分阶段 Colocate 实现方案

> 状态：第一版已实现；Gate 1 与 Gate 3 无效-proposal 分支已通过，待用有效 Alice proposal 完成 Bob receipt 实跑及 Gate 4/5
> 更新时间：2026-09-27
> 适用范围：`PickCube-v1`、`cube_at_position`、Qwen3.5-4B、LoRA、ms-swift GRPO
> 目标：在 8 张 RTX 4090 上依次执行“8 卡 Alice 出题 → 8 卡冻结 Bob 评测 → 8 卡 Alice GRPO 训练”，让 rollout 和训练复用同一组 GPU，同时保持现有 reward 语义不变。

当前完整外层循环入口：

```bash
OPENETA_STAGED_COLOCATE=true \
VLLM_MODE=colocate \
TRAIN_GPU=0,1,2,3,4,5,6,7 \
./scripts/run_embodied_swift_formal10.sh
```

该模式在 Alice/Bob 训练阶段不启动外部 rollout server；每轮训练完成后的固定 Bob holdout 仍使用现有独立 server 评测路径。

## 1. 角色和算法语义

本方案不改变 Alice/Bob 的任务定义。

Alice 是出题者。她从给定初始 snapshot 出发，在 ManiSkill 中执行真实动作，把方块移动到一个与初始位置明显不同、物理上可达的新位置。最终方块位置经过位移、边界、真实操作证据和 replay 一致性检查后，编译为 `cube_at_position` 任务。

Bob 是解题者。他恢复到与 Alice 相同的初始 snapshot，只看到目标 marker、instruction 和自己的 observation，看不到 Alice 的动作轨迹；他的目标是独立把方块移动到 Alice 创造的目标位置。

Alice 的最终 reward 仍由以下部分组成：

```text
Alice proposal 有效性
+ 冻结 Bob 在该任务上的 competence-boundary reward
+ 新颖性奖励
- 重复题目惩罚
- 动作输出格式惩罚
```

“分阶段”只改变执行顺序，不改变任务、采样和 reward 定义。

## 2. 当前实现及其瓶颈

当前正式实验使用：

```text
ms-swift  4.4.2
vLLM     0.19.1
模型      Qwen3.5-4B
训练      8 卡 DDP
rollout   vLLM server mode
```

当前 Alice scheduler 在一条 Alice 轨迹结束时同步调用 Bob：

```text
Alice rollout
  → 编译 proposal
  → _evaluate_alice_task_with_bob()
  → 等待外部 Bob server 完成 K 次真实环境 rollout
  → 计算最终 reward
  → 把完整样本返回 Swift
```

对应代码在：

- `plugins/embodied_swift_grpo.py::_finalize()`；
- `plugins/embodied_swift_grpo.py::_evaluate_alice_task_with_bob()`；
- `plugins/embodied_swift_grpo.py::EmbodiedAliceReward`。

这使 Alice rollout 和 Bob 评测嵌套在同一个请求里。当前外部 Alice/Bob vLLM 服务各占一张卡，而 8 卡训练进程在等待 rollout 时不能充分利用 GPU。Bob 做真实环境交互也很耗时，只给 Bob 一张卡会继续形成串行瓶颈。

当前 Swift colocate 已经具备下面的基础能力：

- 每个 DDP rank 创建本地 vLLM engine；
- rollout 前唤醒 engine，并同步最新训练权重或 LoRA；
- rollout 后让 engine sleep，释放训练阶段需要的显存；
- `vllm_tensor_parallel_size=1` 时，8 个 rank 对应 8 个独立推理 engine；
- 支持多轮 scheduler、prefix caching、LoRA 和 rollout logprobs。

缺少的是 OpenETA 特有的阶段协调：先收齐 Alice proposals，再统一切换到冻结 Bob，最后把 Bob receipts 回填到 Alice 样本。

## 3. 目标执行流程

一次 Alice GRPO generation group 按以下顺序执行：

```text
┌──────────────────────────────────────────────────────────────┐
│ Phase A：Alice rollout                                      │
│ GPU0...GPU7 各使用本地 vLLM engine 和当前 Alice LoRA         │
│ 同一个 snapshot 独立采样 8 条 Alice 轨迹                     │
└───────────────────────────┬──────────────────────────────────┘
                            │ proposals + Alice token/logprob/mask
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase B：冻结 Bob evaluation                                │
│ 收集有效 proposals；每个 proposal 展开 K 个 Bob 请求         │
│ 8 个 rank 切换到同一个冻结 Bob LoRA，并行执行真实环境轨迹      │
└───────────────────────────┬──────────────────────────────────┘
                            │ trusted Bob receipts
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Phase C：Alice GRPO train                                   │
│ receipt 按 proposal_id 回填，生成最终 Alice reward            │
│ vLLM sleep/offload；8 卡共同 forward/backward/optimizer step │
└──────────────────────────────────────────────────────────────┘
```

外层 self-play round 保持：

```text
训练 Alice（Bob adapter 冻结）
→ 从有效 Alice artifacts 构造 Bob 数据集
→ 训练 Bob
→ 保存 Alice/Bob 新 checkpoint
→ 下一 round
```

Bob 自己的 GRPO 训练不需要 Alice 做 evaluator，因此是两阶段：

```text
8 卡 Bob rollout → 8 卡 Bob GRPO train
```

## 4. Batch 含义

下面这组配置：

```text
num_generations=8
generation_batch_size=8
per_device_train_batch_size=1
world_size=8
```

表示一个 prompt/snapshot 对应一个 GRPO group，组内共有 8 条独立 Alice 轨迹。8 卡运行时，每个 rank 负责其中 1 条：

```text
1 个 snapshot × 8 条 completion = 8 条 Alice trajectory
```

它不是 `8 × 8 = 64`。若希望一次处理 8 个 snapshot、每个 snapshot 采样 8 条，generation batch 中要有 64 条 completion；这会显著增加环境实例、上下文 token、Bob 评测数量和训练显存，第一版不采用。

假设 8 条 Alice 轨迹中有 `V` 条有效，每个有效题目由 Bob 评测 `K=2` 次，则 Bob 工作量是：

```text
Bob episode 数 = V × 2
```

若 `V=4`，一共 8 条 Bob episode，可由 8 张卡各执行 1 条。若 `V=8`，一共 16 条；每个 rank 可以顺序执行 2 条，或在显存允许时用 `vllm_max_num_seqs=2` 交错执行。

## 5. 必须保持的等价条件

阶段化实现只有满足以下条件，才能与当前同步 Bob evaluator 语义等价：

1. Bob adapter 在整个 Alice optimizer update 期间保持冻结；
2. Bob 使用相同的初始 snapshot、goal predicate、目标容差和最大步数；
3. 每个有效 proposal 仍执行相同数量的 Bob evaluations；
4. 每条 Bob receipt 通过稳定 ID 回填到正确的 Alice proposal；
5. 收齐 Bob receipts 前不允许执行 Alice optimizer step；
6. 无效 Alice proposal 不启动 Bob；
7. Bob 基础设施错误保持中性 reward，不能伪装成 Bob 策略失败；
8. Alice 的所有真实 assistant action/thinking token、rollout logprobs 和 loss mask 保持对齐；
9. 训练覆盖完整 Alice 轨迹，不能只训练最后若干步；
10. Bob 只用于给 Alice 出题结果评分，Bob 的 token 不能进入 Alice GRPO loss。

## 6. 阶段间数据契约

### 6.1 AliceProposal

Alice scheduler 完成环境执行和 replay 后，先生成 proposal，不再同步调用外部 Bob：

```json
{
  "schema_version": "openeta.alice_proposal.v1",
  "proposal_id": "round-0002-prompt-0003-sample-05",
  "alice_policy_version": ".../alice/checkpoint-12",
  "bob_policy_version": ".../bob/checkpoint-9",
  "snapshot_sha256": "...",
  "snapshot": {},
  "goal_predicate": {
    "type": "cube_at_position",
    "position": [0.03, -0.04, 0.12],
    "tolerance": 0.025
  },
  "compiled_valid": true,
  "replay_equal": true,
  "repeated": false,
  "target_success_rate": 0.45,
  "alice_rollout_info": {},
  "alice_response_token_ids": [],
  "alice_response_loss_mask": [],
  "alice_rollout_logprobs": []
}
```

`proposal_id` 必须由 round、prompt ID 和 group 内 sample index 构成，不能使用进程局部自增编号。

### 6.2 BobEvalRequest

每个有效 proposal 展开 `K` 条 Bob 请求：

```json
{
  "schema_version": "openeta.bob_eval_request.v1",
  "evaluation_id": "<proposal_id>-bob-eval-00",
  "proposal_id": "<proposal_id>",
  "evaluation_index": 0,
  "bob_policy_version": ".../bob/checkpoint-9",
  "snapshot_sha256": "...",
  "snapshot": {},
  "goal_predicate": {},
  "max_steps": 32,
  "seed": 31003
}
```

### 6.3 BobEvalReceipt

Bob scheduler 返回可信 receipt：

```json
{
  "schema_version": "openeta.bob_eval_receipt.v1",
  "evaluation_id": "<proposal_id>-bob-eval-00",
  "proposal_id": "<proposal_id>",
  "bob_policy_version": ".../bob/checkpoint-9",
  "snapshot_sha256": "...",
  "goal_predicate": {},
  "success": false,
  "reward": 0.12,
  "step_count": 32,
  "infrastructure_error": null
}
```

### 6.4 FinalizedAliceRollout

协调器校验 receipt 后，计算：

```text
success_rate = successes / K
difficulty = boundary_score(success_rate, target_success_rate)
novelty = 0.0 if repeated else 1.0
repetition_penalty = 0.15 if repeated else 0.0
reward = clamp(difficulty + 0.25 * novelty - repetition_penalty - format_penalty)
```

最终 reward 写回 Alice 原始 `rollout_infos`。Swift 后续仍通过 `EmbodiedAliceReward` 读取 trusted reward，GRPO loss 路径不需要理解 Bob 的环境细节。

## 7. 代码改造

### 7.1 拆分 Alice finalize

修改 `plugins/embodied_swift_grpo.py`：

1. 把 `_finalize()` 中 Alice 分支拆成：
   - `_finalize_alice_proposal()`：编译任务、replay、有效性与格式检查；
   - `_finalize_alice_with_bob_receipts()`：验证 Bob receipts 并计算最终 reward；
2. staged 模式下 `_finalize()` 返回 `reward_pending=true` 的 proposal；
3. 保留当前 `_evaluate_alice_task_with_bob()`，作为 server mode 回归基线；
4. 无效 proposal 仍可立即计算 `invalid_proposal_shaping`，无需进入 Bob 阶段；
5. 统一复用当前 `_boundary_score()`、`_alice_feasibility_reward()` 和格式惩罚函数，避免产生两套 reward 公式。

建议增加纯函数：

```python
def build_bob_eval_requests(proposal: AliceProposal, evaluations: int) -> list[BobEvalRequest]: ...

def finalize_alice_reward(
    proposal: AliceProposal,
    receipts: list[BobEvalReceipt],
) -> dict[str, Any]: ...
```

这两个函数不持有 ManiSkill 或 vLLM 对象，便于用固定 fixture 验证同步模式与 staged 模式 reward 一致。

### 7.2 增加 staged trainer

新增：

```text
agent/training/embodied_staged_colocate.py
```

实现 `StagedEmbodiedGRPOTrainer`，继承当前版本的 `swift.rlhf_trainers.GRPOTrainer`，只重写 rollout 阶段，不复制 Swift 的 GRPO loss。

核心入口建议放在 `_fast_infer()` 周围：

```python
def _fast_infer(self, samples):
    wake_rollout_engine()
    sync_current_alice_lora()

    alice_outputs = run_alice_phase(samples)
    local_proposals = extract_proposals(alice_outputs)

    all_proposals = gather_and_sort(local_proposals)
    local_bob_requests = shard_bob_requests(all_proposals, rank, world_size)
    local_receipts = run_frozen_bob_phase(local_bob_requests)
    all_receipts = gather_and_index(local_receipts)

    finalized_outputs = attach_rewards(alice_outputs, all_receipts)
    sleep_rollout_engine()
    return postprocess_alice_outputs(samples, finalized_outputs)
```

不要重写 Swift 后续的：

- reward normalization；
- GRPO advantage；
- reference policy/KL；
- importance ratio；
- completion mask；
- backward、gradient accumulation 和 optimizer step。

这样可以把定制范围限制在“如何得到带最终 reward 的 rollout”。

### 7.3 分布式收集与分片

所有 rank 必须按同一个确定性顺序进入 collective：

```text
local Alice outputs
→ gather_object(proposals)
→ 按 proposal_id 排序
→ 展开 Bob requests
→ requests[rank::world_size]
→ 本 rank 执行 Bob
→ gather_object(receipts)
→ 按 evaluation_id 建索引
→ 各 rank 回填自己的 Alice outputs
```

即使某个 rank 没有有效 proposal，也必须参加 gather/barrier，避免其他 rank 永久等待。

每个 proposal 必须恰好得到 `K` 个不同 `evaluation_index` 的 receipt。缺失、重复、snapshot hash 不一致、goal 不一致或 Bob policy version 不一致都视为 evaluator 基础设施错误。

### 7.4 同一个 vLLM engine 切换 Alice/Bob LoRA

Qwen3.5-4B base model 在每张卡只加载一次。Phase A 使用当前训练中的 Alice LoRA，Phase B 使用冻结 Bob LoRA。

当前 Swift 4.4.2 的 colocate 初始化在启用 LoRA 时硬编码：

```python
max_loras = 1
```

目标实现需要在自定义 trainer 的 engine 初始化中设置 `max_loras=2`，并为两个 adapter 使用稳定且不同的 ID：

```text
Alice LoRA  id=1  当前训练 adapter，每个 optimizer step 后重新同步
Bob LoRA    id=2  从本 round 固定 checkpoint 加载，只读
```

Phase A 的所有请求显式路由到 Alice adapter；Phase B 的所有请求显式路由到 Bob adapter。不能依赖“最后一次加载的 adapter”作为隐式状态。

推荐使用 vLLM/Swift 已有的 `adapter_request` 路由能力。Bob 多轮循环的每一次 model call 都要携带同一个 Bob adapter request。

若当前 vLLM 对同一 VLM engine 的双 LoRA 常驻不稳定，兼容方案是在阶段边界卸载/加载 adapter；仍然只保留一个 base model，不重启 engine。只有这条路径也不稳定时，才退回独立 Bob server。

### 7.5 Scheduler

保留两个角色 scheduler：

```text
EmbodiedAliceScheduler  负责 Alice 环境生命周期与 proposal
EmbodiedBobScheduler    负责 Bob 环境生命周期与 trusted receipt
```

staged trainer 负责选择当前阶段使用哪个 scheduler。不要让一个 scheduler 根据请求内容静默切换角色；显式阶段更容易验证 adapter、prompt 和环境目标是否匹配。

### 7.6 启动入口

新增：

```text
scripts/run_embodied_staged_colocate.py
scripts/run_embodied_swift_staged_colocate.sh
```

Python 入口负责在启动 Swift pipeline 前注册 `StagedEmbodiedGRPOTrainer`，避免直接修改 `/opt/openeta-swift` 中已安装的包。注册行为必须受 `OPENETA_STAGED_COLOCATE=true` 控制，普通 server mode 继续使用原 trainer。

Shell 入口负责：

- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`；
- `NPROC_PER_NODE=8`；
- 传入冻结 Bob adapter 路径和 policy version；
- 设置 `VLLM_MODE=colocate`；
- 不再启动外部 Alice/Bob rollout server；
- Alice 阶段完成后继续用现有脚本生成 Bob 训练数据；
- Bob 训练也使用 8 卡 colocate。

## 8. 第一版配置

建议第一版以语义正确和不 OOM 为目标：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPROC_PER_NODE=8

VLLM_MODE=colocate
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_NUM_SEQS=1
VLLM_ENABLE_PREFIX_CACHING=true
VLLM_ENABLE_LORA=true
VLLM_SLEEP_LEVEL=2

NUM_GENERATIONS=8
GENERATION_BATCH_SIZE=8
PER_DEVICE_TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
TRAIN_EPOCHS=3

OPENETA_BOB_EVALUATIONS=2
OPENETA_STAGED_COLOCATE=true
OPENETA_FROZEN_BOB_ADAPTER=<本轮冻结 Bob checkpoint>
```

`vllm_tensor_parallel_size=1` 很关键：它让 8 个 DDP rank 各自拥有一个本地 engine，从而获得 8 路环境轨迹并行。若设置为 8，会变成 8 卡共同服务一个推理 engine，不符合当前 4B 模型和短动作输出的并行目标。

第一版保留 `vllm_enforce_eager=true` 做语义对照。跑通后再测试 `false`，评估 CUDA Graph 是否能稳定降低推理开销。

`sleep_level=2` 释放显存更多，但每次 rollout 需要重新恢复/同步更多权重。若 8 卡实测在 `sleep_level=1` 下训练不 OOM，应优先使用 level 1。最终配置由显存峰值和 phase wall time 决定。

`vllm_max_num_seqs` 先设为 1。确认单轨迹稳定后，在 Bob 阶段单独允许每个 rank 同时处理 2 条请求；不要一开始同时扩大序列长度、并发数和 batch。

## 9. vLLM 与 SGLang 的选择

当前实现选择 vLLM。

ms-swift 4.4.2 的普通 `swift infer/deploy` 支持 SGLang，但 GRPO 专用 `swift rollout`、colocate、自动 LoRA/权重同步和多轮训练集成目前围绕 vLLM 实现。本机环境也只安装了 vLLM 0.19.1，没有安装 SGLang。

| 能力 | vLLM | SGLang |
|---|---|---|
| Swift 普通推理/部署 | 支持 | 支持 |
| Swift GRPO rollout | 原生支持 | 当前未接入 |
| Swift colocate | 原生支持 | 当前未接入 |
| LoRA/训练权重自动同步 | 原生支持 | 需要自行实现 |
| 当前 OpenETA 多轮 scheduler | 已接入 | 需要适配 |
| 多轮前缀复用 | Automatic Prefix Caching | Radix Cache |

SGLang 的 Radix Cache 对长前缀、多轮和高并发场景可能更快，但 vLLM 也支持多轮 Automatic Prefix Caching。当前总耗时还包含 ManiSkill step、Bob 真实执行和 8 卡 forward/backward；只替换推理 kernel 不会消除这些成本。

因此实施顺序是：

1. 先完成 vLLM 8 卡 staged colocate；
2. 记录 Alice rollout、Bob evaluation、adapter switch、weight sync 和 train 各阶段耗时；
3. 如果 rollout 仍是主要瓶颈，再用保存的相同请求做 vLLM/SGLang 离线 A/B；
4. 只有 SGLang 在 Qwen3.5-4B 图像多轮 workload 上有稳定、显著收益时，才评估 GRPO bridge。

SGLang A/B 必须固定模型、adapter、图片、prompt token、采样参数和请求并发，至少比较：

- 每个环境 turn 的 TTFT；
- prompt prefill tokens/s；
- 整条 32-turn trajectory wall time；
- prefix cache 命中率；
- GPU 峰值显存；
- 输出格式有效率；
- 同一 seed 下的任务成功率分布。

## 10. 失败处理

### Bob 单条评测失败

receipt 写入 `infrastructure_error`。同一 `evaluation_id` 可按固定上限重试；超过上限后，该 Alice proposal 使用 `bob_evaluator_error` 中性 reward，不能把错误计为 Bob failure。

### 某个 rank 进程失败

本 generation group 不执行 optimizer step，由外层脚本从最近 checkpoint 恢复。不能用其余 rank 的部分 receipts 继续更新。

### 显存不足

按以下顺序收缩：

1. `vllm_max_num_seqs=1`；
2. 降低固定 KV cache；
3. `sleep_level=2`；
4. 开启 `offload_model`/`offload_optimizer`；
5. 降低每 turn thinking token 上限；
6. 最后才减少环境最大步数或 generation group 大小。

不能通过只训练最后几步解决显存问题，因为那会改变训练目标。

### Adapter 路由错误

每条 artifact 同时记录：

```text
role
phase
adapter_name
policy_version
rank
proposal_id/evaluation_id
```

Bob receipt 的 adapter 或 policy version 与本 round 冻结版本不一致时直接拒绝。

## 11. 日志和性能指标

每个 generation group 至少记录：

```text
alice_phase_seconds
alice_trajectories
alice_valid_proposals
bob_phase_seconds
bob_episodes
bob_success_rate_mean
adapter_switch_seconds
weight_sync_seconds
vllm_wake_seconds
vllm_sleep_seconds
train_forward_seconds
train_backward_seconds
optimizer_step_seconds
gpu_peak_memory_by_rank
prefix_cache_hit_rate
```

artifact 目录建议：

```text
round-XXXX/
  alice_rollout/proposals/
  bob_evaluation/receipts/
  alice_train/
  bob_train/
  phase_timing.jsonl
  staged_group_manifest.jsonl
```

`staged_group_manifest.jsonl` 记录 proposal 与 receipts 的完整映射，便于确认 reward 来源和断点恢复。

## 12. 验收步骤

### Gate 1：纯函数 reward parity

使用已有有效/无效 Alice artifacts 构造固定 fixtures，同时经过：

```text
现有同步 _evaluate_alice_task_with_bob 路径
新 proposal + receipts + finalize 路径
```

两条路径的 reward、stage、success rate、format penalty 和 novelty 必须一致。

### Gate 2：单卡 staged canary

单卡依次跑 1 条 Alice、2 条 Bob、1 个 GRPO step，确认：

- 同一个 base model 可以切换 Alice/Bob LoRA；
- Bob 看不到 Alice trajectory；
- Alice token/logprob/mask 没被 Bob 输出污染；
- optimizer step 发生在 Bob receipts 完成之后。

### Gate 3：8 卡一组

运行一个 `num_generations=8` group，确认：

- 每个 rank 执行 1 条 Alice trajectory；
- proposal ID 无重复；
- 有效 proposal 恰好得到 2 条 Bob receipts；
- 无 collective hang；
- 8 卡都参加训练；
- 保存可恢复 checkpoint。

### Gate 4：与 server mode 对照

用相同 snapshot 集分别跑 server mode 和 staged colocate，比较：

- proposal valid rate；
- Bob success rate；
- reward 分布；
- action 格式有效率；
- group wall time；
- GPU 峰值显存。

允许随机采样导致逐轨迹不同，但统计分布和 reward 公式必须一致。

### Gate 5：正式 5 round

通过前四个 gate 后再跑 5 round，监控：

- Alice 是否逐渐提出更接近 Bob 能力边界的题；
- Bob holdout success 是否提升；
- GRPO zero-variance group 比例；
- 无效 proposal 比例；
- evaluator error 比例；
- 每 round wall time 是否稳定。

## 13. 实施顺序

建议按以下提交边界实施：

1. `AliceProposal`、`BobEvalRequest`、`BobEvalReceipt` 数据类和 reward 纯函数；
2. Alice finalize 拆分，同时保持现有 server mode 行为不变；
3. 单进程 staged coordinator 和 reward parity canary；
4. 8-rank proposal gather、Bob request 分片和 receipt 回填；
5. 双 LoRA vLLM adapter 路由；
6. 自定义 Swift trainer 注册和新启动脚本；
7. 单卡、8 卡、server/colocate 对照；
8. 显存与吞吐调优；
9. 5-round 正式实验；
10. rollout 仍占主要耗时时，再做 SGLang A/B。

## 14. 完成标准

本迁移完成需要同时满足：

- 8 张卡在 Alice rollout、Bob evaluation 和 GRPO train 三个阶段都实际工作；
- 同一时刻每张卡只保留一份 Qwen3.5-4B base rollout model；
- Alice/Bob adapter 路由可从 artifact 审计；
- Alice reward 与当前同步 evaluator 定义一致；
- Bob 使用严格冻结的 policy version；
- 完整轨迹参与训练；
- 无效 proposal、Bob 策略失败和基础设施错误三者可区分；
- 8 卡运行不 OOM、不死锁、可从 checkpoint 恢复；
- 相比当前单 Alice server + 单 Bob server，完整 generation group wall time 有可重复的下降；
- 在决定是否接入 SGLang 前，已有同 workload 的可复现实测数据。

## 15. 相关文件与参考

仓库内文件：

- `plugins/embodied_swift_grpo.py`
- `scripts/run_embodied_swift_grpo.sh`
- `scripts/run_embodied_swift_formal10.sh`
- `docs/embodied-selfplay-vllm-swift-grpo-migration.zh-CN.md`
- `docs/embodied-asymmetric-selfplay-implementation.zh-CN.md`

当前安装的 Swift 关键实现：

- `/opt/openeta-swift/lib/python3.12/site-packages/swift/rlhf_trainers/rollout_mixin.py`
- `/opt/openeta-swift/lib/python3.12/site-packages/swift/infer_engine/grpo_vllm_engine.py`
- `/opt/openeta-swift/lib/python3.12/site-packages/swift/rollout/multi_turn.py`

官方资料：

- ms-swift GRPO：<https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/GetStarted/GRPO.md>
- ms-swift 多轮 GRPO：<https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/multi_turn.md>
- vLLM Prefix Caching：<https://docs.vllm.ai/en/latest/design/prefix_caching/>
- SGLang Session Radix Cache：<https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/session_radix_cache.mdx>
