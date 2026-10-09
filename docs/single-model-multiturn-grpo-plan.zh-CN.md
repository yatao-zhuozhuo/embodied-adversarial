# 单模型多轮 GRPO 训练实施计划

日期：2026-10-06。目标分支：`feat/rl`。本次交付为代码调研和实施计划，尚未创建分支、修改训练代码或启动训练。

基于 `feat/SFT-data` 的输入对齐修复，复用现有 ms-swift 和 vLLM GRPO 链路，在 SFT 采集使用的 20 个任务上训练一个策略模型。模型通过多轮 XML 决策与真实仿真环境交互，整条轨迹结束后只获得任务成功或失败的二值奖励。先完成可复现的训练闭环，再验证是否产生真实的策略学习信号。

## 分支基础与工作区依赖

当前本地 `feat/SFT-data` 为 `a97439f`，提交标题为 `Unify GRPO XML trajectories and bounded training inputs`，包含统一输入、逐轮训练条件校验和 XML 接入修复。查询远端 `fork` 得到 `feat/SFT-data` 仍为 `6f8daca`；本地领先一个提交。因此当前应以包含 `a97439f` 的本地分支为基础，实施时再检查是否有新增提交。

工作区有大量未提交修改和未跟踪文件。20 题清单 `SFT_data/configs/deepseek_xml_debug5.v1.json`、采集器 `scripts/collect_two_tool_sft.py` 等尚未跟踪；RoboCasa 渲染修复还涉及修改过的 worker、环境文件和未跟踪的 `sim/envs/robocasa/rendering.py`。仅从 Git 提交创建干净工作树，不会自动获得这些内容。

实施第一步是建立 `feat/rl` 的独立工作树，记录基础提交，并按文件复制、核对确实需要的工作区依赖。记录源码哈希和依赖清单，形成可提交、可复现的实现；不整体提交现有脏工作区，不移动或覆盖原始 SFT 数据。本计划先写在用户指定的当前仓库 `docs` 目录，后续一并带入新分支。

## 这 20 个任务的准确范围

任务来源是 [DeepSeek 20 题诊断清单](../SFT_data/configs/deepseek_xml_debug5.v1.json)。[Gemini 采集入口](../scripts/run_gemini_sft_comparison.py)也使用这份清单；不是更早的 ManiSkill 专用 `task_catalog.v1.json`。

| Benchmark | 五个任务 ID | 清单预期动作维度 |
| --- | --- | --- |
| MetaWorld | `reach-v3`、`bin-picking-v3`、`assembly-v3`、`lever-pull-v3`、`disassemble-v3` | 4 |
| LIBERO | `libero_spatial_task6`、`libero_10_task5`、`libero_90_task70`、`libero_10_task0`、`libero_90_task35` | 7 |
| RoboCasa | `CoffeeSetupMug`、`CloseBlenderLid`、`MoveToFreezerDrawer`、`ArrangeDrinkware`、`ScrubCuttingBoard` | 12 |
| ManiSkill | `PullCube-v1`、`PokeCube-v1`、`LiftPegUpright-v1`、`FMBAssembly1Easy-v1`、`PickSingleYCB-v1` | 7 |

实际动作维度、上下界和控制含义以 `create_env` 返回的实时 contract 为准，并与清单交叉检查。任务文字沿用 reset 返回的目标及 `two_tool_input.task_description` 的已审核描述，不注入教师动作流程。

这里复用的是 **20 个环境任务**。RL 需要当前策略重新与环境交互，不能把教师成功轨迹直接当作 on-policy GRPO rollout，也不能只留下成功轨迹参与训练。

已有证据及其边界：

- `SFT_data/reports/input_v1_preflight/` 保存了四个 benchmark 各五题的 `passed` 报告，覆盖相机、控制契约、reset、observe 和单步控制；这是宿主预检，不是四环境 GRPO 验收。
- 同目录 `swift_encoding.json` 保存了五图输入的 Swift 编码记录；仍须验证新策略真实采样时的 tokenizer 条件和 loss 对齐。
- 调研时 `deepseek_input_v1_nothink_20261006_065047` 有 60 条记录，其中 1 条 `official_success`，来自 `PokeCube-v1/attempt-0001`；其余停止原因为 44 次 `invalid_xml`、8 次 `model_finish`、6 次 `turn_limit`、1 次 `control_step_limit`。这反映纯成功奖励的探索难度，不能推断待训练模型的成功率。

## 现有代码能复用什么

| 代码 | 已有能力 | 本次使用方式 |
| --- | --- | --- |
| [统一输入构造器](../agent/runtime/two_tool_input.py) | 四环境相机 profile、原生控制契约、四组历史、图像去重及有界输入 | 直接复用 |
| [XML 解析器](../agent/runtime/two_tool_xml.py) | `observe_env`、`step_env`、`task_complete`，严格参数校验 | 直接复用，不修复生成答案后沿用旧 logprob |
| [GRPO 输入适配](../agent/training/grpo_input.py) | Swift 消息、图像顺序、实际 tokenizer 预算 | 复用，补齐通用 MCP 图像载荷适配 |
| [逐轮训练 mixin](../agent/training/conditional_turns.py) | 完整 episode 后拆逐轮样本、prompt/token/mask 核验、变长 rank 优势分发 | 保留为训练核心 |
| [InputGRPOTrainer](../agent/training/input_grpo_trainer.py) | 普通 GRPO 与逐轮训练 mixin 的组合 | 作为单策略 trainer，无需双角色 staged trainer |
| [当前 Swift 插件](../plugins/embodied_swift_grpo.py) | 多轮 scheduler、token/logprob 留档、trainer 注册 | 提取或复用通用逻辑，新增独立单策略 scheduler 和 reward 注册名 |
| [SFT 采集器](../scripts/collect_two_tool_sft.py) | 四环境 MCP 生命周期、原生动作、官方成功字段、图片留存 | 提取公共环境逻辑；不依赖教师 API |
| [Swift 启动脚本](../scripts/run_embodied_swift_grpo.sh) | LoRA、GRPO 参数和 checkpoint 加载 | 参考并新增单策略启动入口，显式设置输入 v1 必需参数 |

仓库还存在 [旧单模型 smoke trainer](../scripts/train_embodied_grpo.py)，所以并非完全没有单模型代码。但它使用旧动作码、ManiSkill 专用环境及 shaping reward，不符合本次四环境、XML、输入 v1 和纯成功奖励要求。

现有 Bob 路径也不能直接用一个 `ROLE=bob` 开关完成本次任务：

1. `_EmbodiedScheduler.on_trajectory_start` 明确限制 two-tool 为 `PickCube-v1`，绑定 `McpPickCubeAdapter`、snapshot 和 `cube_at_position` predicate。
2. 统一输入 profile 写死 ManiSkill/PickCube，任务指令会替换为 `XML_BOB_TASK`；它不是任意 benchmark 原生目标。
3. `_bob_reward` 包含抓取、环境 dense reward、目标进度和格式惩罚，空轨迹还可能返回负奖励。
4. worker 的 `host_state_env` 同样只支持 ManiSkill PickCube，不能假定四环境已有通用状态恢复接口。
5. `ConditionalTurnsMixin` 要求 `vllm_mode=colocate`、`steps_per_generation=1` 和 `loss_scale=last_round`；普通启动脚本的部分默认参数不满足这些条件。

## 单模型训练设计

### 一个策略与一套原生任务

使用一个可训练策略和一个 LoRA adapter，连续训练全部 20 个任务。取消 Alice 出题、冻结 Bob 评估服务、双方轮换和对抗奖励；原生环境负责提供目标和官方成功检查。

建议第一版延用现有链路的 `Qwen3.5-4B`，关闭 thinking，并使用 `InputGRPOTrainer + vLLM colocate`。模型初始化作为显式配置：本计划暂按原始 base 加新 LoRA 设计；使用这 20 题不代表已经指定从某个 SFT checkpoint 初始化。如果后续采用 SFT checkpoint，应记录其确切路径、基础模型和输入协议版本，不能自动选择未知历史 adapter。

GRPO 的旧策略 logprob 和可选冻结 KL reference 不属于第二个对抗训练模型。保留 KL 时记录 reference 来源；报告策略梯度与 KL 项，避免把仅由正则项引起的变化当作任务学习。

### 多轮采样与环境契约

```text
任务与初态 → G 条当前策略轨迹
               每轮：真实图像与有界历史 → XML → 原生环境动作 → 回执
           → 每条轨迹最终成功奖励
           → 同组 episode 优势
           → 按实际决策输入拆样本并校验
           → 单个策略更新 → checkpoint → 重新评测
```

每个 request 使用独立 session/handle，宿主执行 create/reset/close；模型只决策两种工具和结束响应。`step_env` 使用严格模式，按回执的 `executed_control_steps` 扣预算。环境结束后停止动作；动作发送后发生通信中断时不盲目重发，以免重复执行。

成功即停止；其他终止条件包括模型结束、环境 terminated/truncated、控制步或轮数或 token 预算耗尽，以及连续无效输出达到上限。沿用三次连续无效和最多三次连续 observe 的约束。截断 XML 不执行，原始生成和结束原因照实保存。

输入沿用 SFT 的相机集合和控制描述：MetaWorld 主图、LIBERO 主图与 wrist、RoboCasa 左右主图与 wrist、ManiSkill 对应子 profile；RoboCasa 的历史与当前组合最多五图。不得将旧脚本的低 `max_pixels`、图数上限或固定七维动作静默带入新入口。MCP 的 `rgb_base64` 与现有 `InputBuilder.observe` 的 RGB 数组接口需要显式转换，并核对像素和图片顺序。

### 组内初态必须可比较

一个 group 由同一任务、同一初始条件的 G 条独立 rollout 构成；组内改变采样随机性，不能给每条 generation 换一个环境 seed。`prompt_id` 绑定任务与初态，`request_id` 标识独立 episode，逐轮展开保持原 episode 标识。

先验证各环境在相同 seed、reset options、场景/对象选择下的重置一致性。记录任务、场景、可取得的宿主状态摘要和初始图像哈希；同 seed 或首帧一致本身不等价于完整物理状态证明。能够恢复完整状态时优先使用状态恢复；若 reset 不可复现，则补该 backend 的最小宿主状态恢复能力，再进入组内训练。不要把 PickCube 的 snapshot 接口宣称为四环境通用实现。

### 奖励只看任务成功

正常完成的 episode 使用 `R = 1`（官方任务检查成功）或 `R = 0`（其他策略失败）。成功依据来自宿主拿到的 `step_env.info.success`，复用并测试采集器 `_official_success` 的字段归一化规则，包含 ManiSkill 单元素列表；每个 backend 验证该字段确实来自官方 checker。

不使用环境 dense reward、抓取状态、距离变化、格式分或模型自报成功计算训练奖励。`task_complete` 只表示模型结束，不能直接获得 1；terminated/truncated 也不能代替成功标记。错误 XML 可以导致零奖励，但不附加负奖励。

基础设施错误单独记录为 `infrastructure_error`，不伪装成失败奖励。第一版采用有界的整组重采或显式失败退出，保证进入更新的 group 完整；不通过悄悄删除某条 generation 改变组内归一化或 DDP 样本数。重采保持策略版本和任务初态不变，不以是否成功为重采条件。

### 保留输入对齐修复

先完成整条 rollout 的最终奖励，再沿用 `ConditionalTurnsMixin` 拆成真实决策条件下的样本。组内优势按独立 episode 计算，同一 episode 各轮继承相同优势，不把重复的逐轮奖励相加。

每轮必须通过采样 prompt IDs 与训练 prefix 相等、response IDs 与实际 loss targets 相等、token 数与 logprob 数一致的校验。只监督当前轮真实生成 token，环境反馈、图像占位、任务输入和历史回答不重复计入当前轮 loss。episode 内逐轮权重保持现有 token 加权语义，覆盖各 rank 轮数不等和滑窗历史变化。

## 实施步骤与验收

| 阶段 | 工作与拟新增入口 | 完成标准 |
| --- | --- | --- |
| 0 分支与依赖 | 从包含输入修复的最新 `feat/SFT-data` 建立 `feat/rl` 独立工作树；核对未提交依赖 | 基础 SHA、依赖文件与哈希明确；当前 SFT 工作区保留 |
| 1 数据与单策略接口 | 新增 `scripts/create_single_model_rl_dataset.py`、`configs/single_model_rl_20tasks.json`；从原清单生成任务与 seed 数据，不读教师答案 | 精确 20 题、每 bench 5 题；唯一 task key；训练与验证 seed 分开；排除现有 holdout |
| 2 通用环境与奖励 | 新增 `agent/training/single_model_rollout.py`、`plugins/single_model_swift_grpo.py`；实现生命周期、官方成功奖励、组内初态审计与逐轮留档 | 四个 backend 各至少一题可完成两轮以上真实交互；20 题逐一完成初始化、相机/动作/初态预检；奖励严格为 0 或 1 |
| 3 单任务训练 canary | 新增 `scripts/run_single_model_grpo.sh`；接入普通 `InputGRPOTrainer`，优先用这 20 题中的 `reach-v3` 或 `PokeCube-v1` | 当前模型完成分组多轮 rollout、输入校验、优化步骤、checkpoint 保存和重载；分别报告零方差与非零方差组 |
| 4 四环境训练 canary | 各选一题，继续更新同一个模型；先串行限制 worker 并发 | 四环境均通过实际训练编码、reward、loss 和权重同步检查，保存可用 checkpoint |
| 5 全 20 题短跑 | 均匀轮转 20 题，完成一个有限训练周期；保存覆盖统计和逐任务结果 | 每题至少一个完整 GRPO group，全部进入统计；无静默跳题；训练前后同配置评测；中断恢复验证通过 |

表中新增文件名是实施约定，目前尚未创建。公共成功判定和图像载荷处理应放入可复用模块，避免新训练插件直接依赖 SFT 教师请求代码。

第一阶段保持小并发，按需启动四环境 worker，不直接复用会自动启动 Alice/Bob 的 staged coordinator。所需 GPU 数量与模拟器显存由 canary 实测决定；八卡是旧验证规模，不是单模型训练的必需条件。

## 首轮建议配置

以下为拟采用的起点，正式数值写入运行 manifest；预算按真实多模态 tokenizer 和资源预检调整。

| 项目 | 单任务训练 canary | 全 20 题短跑 |
| --- | --- | --- |
| 策略 | Qwen3.5-4B，单 LoRA，nothink | 同一策略连续更新 |
| rollout 后端 | vLLM colocate | 同左 |
| group size | 4，资源受限可先用 2 | 8，资源不足时保留 4 |
| `steps_per_generation` | 1 | 1 |
| 单条轨迹轮数与控制步 | 8 轮、最多 64 控制步，验证多轮机制 | 沿用采集预算：60 轮、400 控制步 |
| 单轮输出预算 | 初始 768 tokens，显式配置 | 初始 768；根据真实截断率调整并留档 |
| 完整轨迹生成预算 | 至少覆盖计划轮数乘单轮上限 | 60 × 768 = 46080；调整轮预算时同步调整 |
| 输入上下文上限 | 按最复杂五图输入和四组历史实测，预留本轮输出空间 | 同左，不能把整条轨迹 token 总数误当单轮上下文需求 |
| LoRA 与优化 | rank 8、alpha 16、学习率 `5e-6` | 暂沿用 canary 参数 |
| GRPO | `loss_type=grpo`、`loss_scale=last_round`、token 级 importance sampling | 同左 |
| KL | 暂沿用 `beta=0.01`，记录 reference | 分别记录策略项和 KL 项 |
| 策略更新次数 | 1 至 3 步用于集成验证 | 至少完整覆盖 20 题一个周期，再根据实测决定扩量 |

启动前校验 world size、每卡 batch、generation batch 与 G 的整除关系，显式启用 `OPENETA_INPUT_PROTOCOL=openeta.two_tool_input.v1`、XML 协议和 `OPENETA_COMPLETION_SCOPE=per_round`。不要沿用旧脚本 `MAX_COMPLETION_LENGTH=128` 或 total-scope 等默认值。

少轮数 canary 用于检查程序机制，不据此判断任务难度或正式成功率。checkpoint 恢复需要同步恢复 optimizer、scheduler、RNG、数据进度，并在下一批采样前将策略更新同步至 vLLM。

## 验证与结果判定

先复用现有 `test_two_tool_xml.py`、`test_grpo_input_context.py`、`test_conditional_turn_samples.py`、`test_grpo_xml_boundary.py` 和相关输入校验脚本，再补充与新增行为直接相关的测试：

- 二值 reward 真值表：模型宣称完成、只有 dense reward、错误成功字段、截断和基础设施错误均不能冒充成功。
- 多环境动作维度、实时 contract、图像顺序、缺相机报错，以及 session 隔离和异常关闭。
- 同题同初态分组、不同 episode 标识、可变轮数与跨 rank 优势归属；混合成功/失败组的优势符号正确。
- 真实多轮滑窗发生后，采样和训练输入仍逐 token 一致；初始图片来自当前环境。
- 保存后重新加载，恢复步数与下一批 rollout 策略版本一致。

验收分为两个层次：

1. **训练链路通过**：真实环境多轮采样、官方 reward、GRPO loss、优化器步骤、checkpoint 和恢复全部通过。零方差组也能验证流程，但必须如实记录其优势为零。
2. **有效 RL 更新通过**：当前策略在真实同题 group 内产生成功/失败差异，任务奖励带来非零优势和策略梯度；参数变化有限且可验证。仅出现 KL 导致的参数变化、伪造奖励或教师强制动作不算通过。

纯二值奖励下，全失败组和全成功组都没有相对奖励信号。先检查格式截断、动作执行和终止原因，再在同一套 20 题内调整任务顺序、G、采样温度或轨迹预算。如果仍无真实混合奖励，只报告链路通过与学习信号未通过，不改成 dense reward，也不宣称已经学会任务。额外 SFT 或更换初始化模型属于另行确定的实验变更。

训练前后使用同一评测配置和独立 seed，对每题报告成功数/总数、宏平均成功率、XML 有效率、执行控制步数、平均轮数、零方差 group 比例、reward 标准差、梯度范数、KL、基础设施错误数和耗时。20 题训练内结果用于验证闭环，不代表未见任务泛化；少量评测不预设必须提升。

遵守 [现有 holdout 约束](../SFT_data/configs/two_tool_test_holdout.v1.json)，尤其不要为了省事把旧 GRPO 的 PickCube 当成本轮额外训练题；该任务已列在现有 holdout 中。训练 seed 不使用保留范围 `20261002` 至 `20261011`。

## 交付产物

运行使用独立目录 `runs/single_model_rl/<run_id>/`，至少包含：

- `run_manifest.json`：基础提交、源码与任务清单哈希、初始化模型、协议、训练参数、版本和 seed 划分。
- `tasks.jsonl`：20 题及实际训练初态信息，不含教师答案。
- `rollouts/`：episode/turn 映射、任务/初态、原始 XML、图像、真实输入、token/logprob/mask、动作回执和停止原因。
- `metrics.jsonl`、`conditional_input_validation.jsonl`：episode 奖励、优势、梯度与输入对齐证据。
- `checkpoint-*`、恢复验证记录，以及 `eval_before.json`、`eval_after.json`。
- `acceptance.json`：20 题覆盖、四环境验收、训练链路和有效 RL 更新分别是否通过，附证据路径。

实现完成后提供准确启动命令和运行结果。本文中的 canary、测试和训练步骤均为待执行计划；已有预检或旧 PickCube 训练证据不替代本轮验收。
