# OpenETA × ManiSkill：双模型各 500 条（共 1000 条）工具调用 SFT 轨迹采集计划

更新日期：2026-09-28

## 1. 结论与范围

本计划的目标是用 OpenETA 在约 20 个 ManiSkill 任务上，让 **GLM-5.3-w8a8c8 和 Qwen3.8-27B 分别独立采集 500 条通过可信环境判定的完整成功轨迹**，总计 1000 条，再导出为适合 Qwen 小模型进行工具调用 SFT 的多轮数据。两个 API 模型都是 teacher，不互相打分，也不把其中一个降级为 reviewer。

建议的技术路径是：

```text
ManiSkill 3（本机 GPU 仿真/渲染）
  -> OpenETA simulator MCP server
  -> GLM-5.3 API teacher  -> 独立 500 条 shard
  -> Qwen3.8-27B teacher  -> 独立 500 条 shard
  -> lossless rollout bundle
  -> success/完整性/去泄漏过滤
  -> 每模型独立全量训练 shard
  -> 两套及可选合并版 ms-swift SFT JSONL
  -> Qwen 小模型 SFT
```

不要直接使用 `scripts/run_embodied_swift_staged_formal5.sh` 采集这批数据。该脚本是 8 卡、5 轮 Alice/Bob GRPO 自博弈训练入口，动作、状态和奖励语义目前主要围绕 `PickCube-v1`，不等同于通用的 20 任务 SFT 采集管线。

本文中的“每模型 500 条轨迹”默认指最终通过验收、可进入该模型数据 shard 的 episode，而不是 API 请求数、模型 turn 数或原始尝试数。初始尝试上限仍保留为每个 teacher 2500 次，以覆盖不同任务成功率差异；失败轨迹按 teacher 单独保留，不计入任何一方的 500 条成功轨迹。

## 2. 当前仓库能力与缺口

### 已有能力

- `sim/mcp_server` 已提供 ManiSkill 环境生命周期、观察、绝对位姿运动、夹爪和底层 step 等 MCP 工具。
- 当前安装的 ManiSkill 是 `3.0.1`；OpenETA registry 实际可发现 52 个 ManiSkill env。
- `openeta-batch`/`openeta-eval` 已支持并发 episode、失败隔离、续跑、provider 并发限制和强制 cleanup。
- OpenETA rollout bundle 已无损保存：
  - `model_calls.jsonl`
  - `tool_calls.jsonl`
  - `transitions.jsonl`
  - `episodes.jsonl`
  - content-addressed RGB/depth/artifacts
- OpenAI-compatible planner 已能发送 `chat_template_kwargs.enable_thinking`，并记录 provider exchange 和模型/endpoint provenance。

### 正式放量前必须补齐

1. **不能依赖 `adapter/maniskill_sim.py` 做 20 任务采集。** 这个轻量 adapter 的 `task_state()`、对象名和成功语义仍硬编码为 cube/goal/PickCube。正式采集应走通用 `sim/mcp_server`。
2. **补一个 SFT exporter。** `scripts/create_embodied_swift_dataset.py` 生成的是 GRPO rollout 请求行，不是从 OpenETA lossless rollout 导出的 SFT 标签。
3. **统一成功判定。** ManiSkill 默认可能使用 dense/normalized-dense reward，不能用“reward > 0”代表完成。最终接收必须检查同一 episode 的 `info.success == true` 或任务专用 checker。
4. **防止 agent 自报成功。** manifest 显式设置 `metadata.require_official_reward=true`；同时 exporter 再次核验 `success`，不接受只有 `response::task_complete` 的样本。
5. **暴露采集配置。** 建议增加：
   - `OPENETA_LLM_ENABLE_THINKING=true|false|unset`
   - `OPENETA_LLM_MAX_TOKENS`
   - collection run id / dataset version / teacher tag
6. **为不同任务补任务描述和最小 task skill。** 不能只给模型 `ManiSkill task: <env_id>`；每个任务要有明确目标、完成条件和禁止使用的 privileged state。
7. **补 manifest generator、bundle audit 和 exporter 测试。** 任何一个缺失都不进入双模型各 500 条的正式采集。

## 3. Teacher API 设计

### 3.1 GLM-5.3 teacher shard

- Provider：OpenAI-compatible
- Model：`GLM-5.3-w8a8c8`
- API base：`https://hhapbb8eb5cocm8kmma8okkogmea5pde.openapi-sj.sii.edu.cn/v1`
- thinking：第一版保留服务默认值，即请求中不发送 `chat_template_kwargs`；另做一个显式关闭 thinking 的小规模 A/B。
- temperature：正式标签建议 `0.0`～`0.2`。
- provider 并发：从 2 开始，确认无 429/过载后再升到 4；不要一开始按环境并发数同时打满 API。

安全配置示例（不要把 key 写入仓库）：

```bash
read -rsp 'INF_API_KEY: ' INF_API_KEY
export INF_API_KEY
export OPENETA_LLM_PROVIDER='openai-compatible'
export OPENETA_LLM_MODEL='GLM-5.3-w8a8c8'
export OPENETA_LLM_API_BASE='https://hhapbb8eb5cocm8kmma8okkogmea5pde.openapi-sj.sii.edu.cn/v1'
export OPENETA_LLM_API_KEY="$INF_API_KEY"
export OPENETA_LLM_TIMEOUT_S='180'
export OPENETA_LLM_MAX_ATTEMPTS='3'
export OPENETA_LLM_RETRY_BACKOFF_S='1.0'
export OPENETA_LLM_CONTEXT_WINDOW_TOKENS='1000000'
export OPENETA_WEB_SEARCH_ENABLED='false'
export OPENETA_WEB_FETCH_ENABLED='false'
```

用户提供的 key 只放在作业环境变量或受控 secret 中，不进入 `.env`、manifest、日志、计划文件或数据集。正式运行前建议轮换一次已经在聊天中出现过的 key。

### 3.2 Qwen3.8-27B teacher shard

- Endpoint：`https://cqhbod8bjjjbcoakk8pmeebgkaq9akcq.openapi-sj.sii.edu.cn/v1`
- Model：`Qwen3.8-27B`
- 目标：与 GLM-5.3 一样，独立完成 20 个任务上的 500 条成功轨迹，不负责审核或打分 GLM 数据。
- thinking：第一版同样保留服务默认值；pilot 中测试显式 `false`，再为正式 shard 固定一个配置。
- 使用与 GLM 相同的任务目录、seed pool 和接收标准，但使用独立 run id、独立 bundle 目录和独立配额计数。
- 不把两个 endpoint 配成彼此的自动 fallback；某个模型请求失败时只重试该模型，避免一条 shard 中混入另一 teacher 的标签。

Qwen shard 的环境变量应在独立采集进程中设置：

```bash
export OPENETA_LLM_PROVIDER='openai-compatible'
export OPENETA_LLM_MODEL='Qwen3.8-27B'
export OPENETA_LLM_API_BASE='https://cqhbod8bjjjbcoakk8pmeebgkaq9akcq.openapi-sj.sii.edu.cn/v1'
export OPENETA_LLM_API_KEY="$INF_API_KEY"
```

### 3.3 API capability gate

正式采集前分别对两个 endpoint 做以下 probe，并把结果写入 `reports/api_capability.json`：

1. `/v1/models` 或最小文本请求可用。
2. 能稳定返回 OpenETA 要求的 `<decision>...</decision>` XML。
3. 能处理 1 张当前 RGB 图像，同时将 simulator 结构化对象状态明确标记为工具状态输入；本版本训练 planner/tool use，不作为纯视觉策略数据。
4. thinking 默认开、显式 `false` 两种请求都能正常返回。
5. 返回 token usage；超时、429、5xx 的行为可被 OpenETA 正确分类和重试。
6. 并发 2、4 的小压测没有持续 capacity error。
7. 用 20 个刻意构造的合法/非法工具参数测试工具 schema 服从率和 validation-repair 成功率。

如果任一 endpoint 不支持图片，该模型仍必须独立采集自己的 500 条，不能用另一个 teacher 代采或代评。该模型的首版有两个处理方向，不能混用：

- `vision-v1`：先等待或切换到该模型自身支持图像的服务版本；保留 RGB 作为模型输入。
- `state-tool-v1`：明确标记为结构化状态工具调用数据，只训练 planner/tool use，不宣称是视觉策略数据。

## 4. 任务与配额

下面是第一版 20 任务候选。表中配额是 **每个 teacher 的配额**：GLM 按表采 500 条，Qwen 也按同一张表采 500 条。任务必须先通过 MCP lifecycle、控制能力和官方成功字段 smoke；未通过的任务用候补替换，不能为了凑数放宽成功标准。

### Tier A：核心任务，10 × 35 = 350

| 任务 | OpenETA env_id | 配额 | 数据用途 | 主要工具链 |
| --- | --- | ---: | ---: | --- |
| PickCube | `openeta/maniskill_PickCube-v1-v0` | 35 | 全部训练 | observe, move, grasp, place |
| StackCube | `openeta/maniskill_StackCube-v1-v0` | 35 | 全部训练 | detect, grasp, stack, verify |
| PlaceSphere | `openeta/maniskill_PlaceSphere-v1-v0` | 35 | 全部训练 | grasp, transport, place |
| PushCube | `openeta/maniskill_PushCube-v1-v0` | 35 | 全部训练 | observe, approach, push |
| PullCube | `openeta/maniskill_PullCube-v1-v0` | 35 | 全部训练 | approach, contact, pull |
| PokeCube | `openeta/maniskill_PokeCube-v1-v0` | 35 | 全部训练 | align, poke, verify |
| LiftPegUpright | `openeta/maniskill_LiftPegUpright-v1-v0` | 35 | 全部训练 | grasp, reorient/lift, verify |
| RollBall | `openeta/maniskill_RollBall-v1-v0` | 35 | 全部训练 | contact planning, roll |
| PickSingleYCB | `openeta/maniskill_PickSingleYCB-v1-v0` | 35 | 全部训练 | object selection, grasp, place |
| TurnFaucet | `openeta/maniskill_TurnFaucet-v1-v0` | 35 | 全部训练 | handle localization, contact, rotate |

### Tier B：组合与精细任务，6 × 20 = 120

| 任务 | OpenETA env_id | 配额 | 数据用途 | 主要难点 |
| --- | --- | ---: | ---: | --- |
| PullCubeTool | `openeta/maniskill_PullCubeTool-v1-v0` | 20 | 全部训练 | 先取工具再操作目标 |
| PickClutterYCB | `openeta/maniskill_PickClutterYCB-v1-v0` | 20 | 全部训练 | clutter 中目标选择 |
| PegInsertionSide | `openeta/maniskill_PegInsertionSide-v1-v0` | 20 | 全部训练 | 方向与精细插入 |
| PlugCharger | `openeta/maniskill_PlugCharger-v1-v0` | 20 | 全部训练 | 6D 对齐与插入 |
| AssemblingKits | `openeta/maniskill_AssemblingKits-v1-v0` | 20 | 全部训练 | 形状匹配与放置 |
| StackPyramid | `openeta/maniskill_StackPyramid-v1-v0` | 20 | 全部训练 | 多对象长时序 |

### Tier C：困难/异构任务，8 + 8 + 7 + 7 = 30

| 任务 | OpenETA env_id | 配额 | 数据用途 | 备注 |
| --- | --- | ---: | ---: | --- |
| FMBAssembly1Easy | `openeta/maniskill_FMBAssembly1Easy-v1-v0` | 8 | 全部训练 | 精密装配 |
| OpenCabinetDoor | `openeta/maniskill_OpenCabinetDoor-v1-v0` | 8 | 全部训练 | Fetch/关节物体 |
| OpenCabinetDrawer | `openeta/maniskill_OpenCabinetDrawer-v1-v0` | 7 | 全部训练 | Fetch/关节物体 |
| PushT | `openeta/maniskill_PushT-v1-v0` | 7 | 全部训练 | panda_stick，无夹爪流程 |

每模型总计：`500 train`。

双模型总计：`1000 train`。默认同时保留 `glm/`、`qwen/` 两套独立导出；需要联合训练时再生成带 `teacher_model` 字段的 merged 版本。两个模型使用相同 seed pool 便于复现，但各自独立执行和验收，不要求一个模型给另一个模型评分。

候补池：`InsertFlower-v1`、`DrawTriangle-v1`、`DrawSVG-v1`、`SceneManipulation-v1`。候补也必须先确认任务参数、默认机器人、action layout、资产和官方 success 语义。

任务选择注意事项：

- Tier C 包含 Fetch 与 panda_stick，目的是增加工具规划多样性。如果最终小模型只服务 Panda，应把这些任务替换为 Panda 任务，避免机器人形态成为无关变量。
- YCB、cabinet、scene manipulation 等任务可能需要额外资产；资产下载和 hash 必须在放量前固定。
- 每个任务 seed 唯一；若因基础资产/variation 机制导致场景重复，按初始状态 hash 再去重。

## 5. 采集分阶段执行

### Phase 0：实现和离线验证

交付物：

- `scripts/generate_maniskill_sft_manifest.py`
- `scripts/export_openeta_sft.py`
- `scripts/audit_openeta_sft.py`
- 每任务任务说明/checker 配置
- provider thinking/max_tokens 配置入口
- 对应单元测试与 1 个 deterministic fixture

验收：

- manifest 数量、seed、env_id、配额可重复生成。
- exporter 能 join model call、tool result、transition、episode result。
- 任何缺文件、seq 断裂、artifact hash 错误、provider 字段缺失都 fail closed。
- API key 和 Authorization 不出现在任何 bundle/JSONL 中。

### Phase 1：每模型 5 条端到端 canary（共 10 条）

- 两个模型分别只跑 `PickCube-v1` 的 5 个 seed。
- environment concurrency = 1，provider concurrency = 1。
- 分模型人工逐条检查其看到了什么、发了什么工具、工具回执是什么、成功字段是否可信。
- 用 rollout validator 验证每个 bundle，再导出 SFT，最后用 Qwen tokenizer 做一遍 parse/tokenize smoke。

Go/No-Go：每模型 5/5 bundle 完整；每模型至少 3 条真实成功；零伪成功；零 secret 泄漏。

### Phase 2：每模型 50 条 pilot（共 100 条）

- 每个模型分别运行 5 个代表任务 × 10 条：PickCube、PushCube、StackCube、PegInsertionSide、TurnFaucet。
- environment concurrency = 2，provider concurrency = 2。
- 每个模型都做 thinking 默认开与显式关闭的小 A/B，再分别为 GLM shard 和 Qwen shard 固定正式配置；这只是配置选择，不是模型互评。
- 测量：单 episode 时长、token、API 错误、工具 schema 合法率、任务成功率、每轨迹磁盘占用。

Go/No-Go：

- planner 首次输出格式合法率 ≥ 95%。
- validation repair 后的工具调用合法率 ≥ 99%。
- simulator lifecycle/cleanup 成功率 = 100%。
- 伪成功率 = 0%。
- 可接受成功轨迹比例建议 ≥ 40%；低于该值先修 prompt/skill/control，不直接扩大重试数。

### Phase 3：每模型 200 条全任务预生产（共 400 条）

- 每个模型在 20 个任务上各收 10 条可接受成功轨迹。
- environment concurrency 从 2 升到 4；provider concurrency 保持 2，确认稳定后最多 4。
- 逐任务建立成功率、p50/p95 时延、平均 turn、工具分布和失败 taxonomy。
- 成功率过低的任务先修复；仍无法达到最低门槛则用候补任务替换。

Go/No-Go：两个模型的 20 个任务均有通过样本；跨任务 exporter 无特判崩溃；相同 teacher 内无重复 seed/初始状态。

### Phase 4：每模型补齐 500 条（总计 1000 条）

- GLM、Qwen 分别按自己的任务剩余 quota 动态补采，不按固定尝试数盲跑。
- raw attempt 上限先设为每模型 2500、总计 5000；任一模型达到上限仍未满足 quota 时，停止该 shard 并出独立 gap report。
- 每个模型每累计 100 条成功轨迹产生一次 immutable checkpoint、统计报告和 fingerprint。
- API 推理完全在远端，不占本机 GPU。本机只运行 ManiSkill/SAPIEN simulator、MCP server 和采集控制进程；不要同时运行本地 vLLM 或 GRPO 训练。

### Phase 5：冻结数据版本

- 分别冻结 GLM 500 条和 Qwen 500 条 episode 清单、bundle hashes、导出脚本 git commit 和 teacher provenance。
- 分别生成两套全量 train JSONL、数据卡和审计报告，并可额外生成 merged train JSONL。
- 原始失败轨迹保留在 `rejected/`，可用于后续 DPO/错误分析，但不混入第一版 SFT。

## 6. 单条轨迹的接收规则

一条轨迹只有同时满足以下条件才计入对应 teacher 的 500 条配额：

1. `validate_rollout_bundle()` 通过。
2. 环境创建、reset、所有工具回执和 close 均可追溯。
3. 同一 episode 的 ManiSkill `info.success == true`，或该任务的明确 checker 返回 true。
4. 不以 dense reward、agent 自报 `task_complete` 或仅“工具调用成功”替代任务成功。
5. `assistance.assisted == false`；无人类回答、guidance agent 代答或 teacher forcing。
6. 所有训练标签对应 host validator 接受的最终 decision；非法 raw completion 不作为正标签。
7. world-mutating 工具后存在新的 observation/receipt，不能连续基于陈旧画面标注动作。
8. 图像、深度、相机参数和工具引用能解析，SHA256 全部匹配。
9. provider/model/thinking 配置、prompt hash、tool contract hash、git commit、env/version/seed 齐全。
10. 同一 teacher 内通过 seed 和初始状态 hash 去重。

额外规则：

- validation retry 中的错误输出进入 `rejected_candidates.jsonl`；修复后的合法输出可进入 SFT。
- 失败 episode 原样保留，但默认不切成“看起来合理”的成功 turn；否则会把最终失败策略教给小模型。
- 对特别长的成功 episode，可以按 turn 前缀导出多个监督样本，但所有样本必须继承同一 `episode_id` 和 teacher shard。

## 7. SFT 导出格式

第一版训练目标应与 OpenETA 实际 planner wire contract 对齐，即 assistant 输出 canonical XML，而不是另外发明 OpenAI `tool_calls` JSON 格式。

建议每个 turn 导出一行：

```json
{
  "schema_version": "openeta.sft.tool_call.v1",
  "sample_id": "<episode_id>:turn-0007",
  "episode_id": "...",
  "split": "train",
  "messages": [
    {"role": "system", "content": "<OpenETA planner prompt + stable tool contracts>"},
    {"role": "user", "content": "<current task/history/observation projection>"},
    {"role": "assistant", "content": "<decision>...</decision>"}
  ],
  "images": ["artifacts/...png"],
  "metadata": {
    "env_id": "openeta/maniskill_PickCube-v1-v0",
    "seed": 17,
    "teacher_model": "GLM-5.3-w8a8c8",
    "thinking_enabled": true,
    "tool_name": "move_to",
    "episode_success": true,
    "source_bundle_sha256": "..."
  }
}
```

导出原则：

- 训练 assistant target 只保留 host 接受的 canonical `<decision>`。
- teacher 的 hidden reasoning 或 `<think>` 默认不作为监督目标；`<decision><reasoning>` 只保留短、可审计、与工具参数一致的理由。
- user 侧保留模型在采集时真正看到的状态，不加入采集后才知道的未来 reward/success。
- tool result 作为下一轮可见上下文，而不是拼进当前 assistant label。
- 图片路径在数据版本内部使用相对路径，发布前校验 hash。
- 每个 teacher shard 单独输出，并生成样本数、episode 数、token 数、tool 分布与 task 分布统计。

双模型共 1000 条成功 episode 预计会产生远多于 1000 条 turn-level SFT 样本。最终训练时同时做两种权重：

- episode-balanced：每个 episode 总权重相近，避免长轨迹支配 loss。
- tool-balanced：对稀有但关键的 `gripper_close`、精细 `move_to`、re-observe、task completion 适当上采样。

## 8. 目录与版本规范

建议最终目录：

```text
SFT_data/
  ManiSkill_500_Trajectory_SFT_Plan.zh-CN.md
  configs/
    provider.example.env
    task_catalog.v1.json
  manifests/
    canary.json
    pilot.json
    glm.train.500.v1.json
    qwen.train.500.v1.json
  raw/
    glm/<collection-run-id>/
    qwen/<collection-run-id>/
  accepted/
    glm/episodes.v1.jsonl
    qwen/episodes.v1.jsonl
  rejected/
    glm/episodes.v1.jsonl
    glm/rejected_candidates.v1.jsonl
    qwen/episodes.v1.jsonl
    qwen/rejected_candidates.v1.jsonl
  exported/
    glm/train.v1.jsonl
    qwen/train.v1.jsonl
    merged/train.v1.jsonl
  reports/
    api_capability.json
    collection_metrics.json
    data_audit.json
    DATA_CARD.md
  fingerprints/
    dataset.v1.sha256
```

`raw/` 可以保存实际 rollout 的索引和相对引用，或保存 bundle 的不可变副本；不要复制后丢失来源关系。先用 50 条 pilot 实测磁盘占用，再给正式任务预留至少 2 倍余量。RGB-D 多 turn 轨迹可能达到数十 GB，不应只按 JSONL 大小估算。

## 9. 监控指标与停止条件

每个 stage 至少按 GLM/Qwen 分开报告，并给出合计：

- raw attempts / accepted successes / success rate，按任务和 seed 分组。
- 首次 XML 合法率、repair 后合法率、非法工具名/参数计数。
- 每 episode 的 turns、tool calls、prompt/completion/total tokens。
- API p50/p95、429/5xx/timeout、retry 次数、provider queue wait。
- simulator create/reset/step/close 失败和 GPU/Vulkan 错误。
- false completion、重复动作、陈旧观测后动作、碰撞/不可达失败。
- teacher model、thinking 开关、prompt/tool-contract hash 的分布。
- 磁盘增长、bundle validation 和 artifact hash 错误。

立即停止并修复的条件：

- 发现 API key、Authorization 或 cookie 落盘。
- 出现伪成功或跨 episode reward/receipt 对错样本。
- simulator close 失败导致资源持续泄漏。
- 连续 20 次 capacity error/429，或 provider 错误率持续超过 10%。
- 任务成功字段不明确，却仍被 exporter 接收。
- 同一 teacher shard 内 seed 或 initial-state hash 重复。

## 10. 当前机器执行建议

- 直接使用当前机器的 GPU 运行 ManiSkill/SAPIEN 仿真与渲染；GLM-5.3、Qwen3.8-27B 都通过远端 API 推理，不在本机加载模型权重。
- 当前检查可见 1 张 RTX 4090（约 49 GB 显存）；检查时 GPU 利用率为 100%，正式采集前先用 `nvidia-smi` 确认现有负载已经释放或安排错峰。
- 本机只启动一套 simulator MCP server。GLM 和 Qwen 可以顺序跑，也可以用两个采集进程共享服务；第一版优先顺序跑，避免单卡多环境渲染和 worker 争抢造成不稳定。
- Phase 1 从 environment concurrency = 1 开始，Phase 2 升到 2；只有 GPU/Vulkan、显存和 cleanup 都稳定后，Phase 3/4 才升到 4。provider concurrency 独立控制。
- 代码、manifest、raw bundle 和 exported dataset 都放在当前 `/inspire/.../OpenETA/SFT_data` 路径，便于断点续跑和后续训练读取。
- 每个模型、每个阶段使用唯一 collection run id；同一 manifest 可以 resume，但不能覆盖已有 raw bundle。
- 监控本机日志、GPU/CPU/内存、API 请求统计、每模型成功轨迹增长和 bundle 完整性，而不只看采集进程是否仍在运行。

## 11. 预计实施顺序

1. 修成功判定、thinking/max_tokens 配置入口。
2. 写 manifest generator、20 任务说明和 capability smoke。
3. 写 lossless rollout -> SFT exporter 与 audit。
4. 两个模型各跑 5 条 canary。
5. 两个模型各跑 50 条 pilot，分别决定 thinking 与 API 并发配置。
6. 两个模型各跑 200 条全任务预生产，淘汰不稳定任务。
7. GLM 补齐 500 条，Qwen 补齐 500 条。
8. 分别冻结两套 500 条全量训练数据，并生成可选 merged 1000 条版本。
9. 先用约 5% 数据做训练加载与 loss smoke，确认工具名、参数 schema 和闭环上下文正常后再全量训练。

## 12. 最终验收清单

- [ ] GLM-5.3 有 500 个唯一、可信成功 episode。
- [ ] Qwen3.8-27B 有 500 个唯一、可信成功 episode。
- [ ] 两个模型都在 20 个任务上达到各自配额，或有书面记录的 smoke 合格替换任务。
- [ ] 每个 teacher shard 内 episode/seed/initial-state 均唯一。
- [ ] 所有 accepted bundle 通过结构和 SHA256 校验。
- [ ] 零 secret 泄漏、零伪成功、零 assisted 样本进入 SFT。
- [ ] 每条 turn label 是 validator 接受的 canonical OpenETA decision。
- [ ] GLM/Qwen、thinking 开关和 endpoint provenance 可追踪。
- [ ] 导出数据可被目标 Qwen tokenizer/ms-swift loader 全量读取。
- [ ] 数据卡包含任务、版本、许可、失败过滤、已知偏差和复现信息。
- [ ] GLM 500 条和 Qwen 500 条均全部进入各自训练 shard。
