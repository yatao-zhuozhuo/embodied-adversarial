# OpenETA 四条链路代码阅读指南

更新时间：2026-10-08。根目录：`/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA`。

本指南区分原 OpenETA 能力和后续新增的评测/数据/训练工程。文件和目录均以当前工作区为准；运行参数最终以实际启动配置、args、run manifest 和输入审计记录为准，不能只看保留历史说明的 README。

## 1. 工作目录与推荐顺序

| 部分 | 相对 OpenETA 根目录的位置 | 当前分支 |
|---|---|---|
| Baseline | `.worktrees/baseline` | `feat/baseline` |
| 数据构造 | `SFT_data`、`scripts` | 主工作区 `feat/SFT-train` |
| SFT | `SFT-train` | 主工作区 `feat/SFT-train` |
| RL | `.worktrees/rl` | `feat/rl` |

推荐顺序：原框架 → 两工具公共协议 → Baseline 一条 episode → 数据构造一条样本 → SFT 编码与监督 → RL rollout、优势和反向传播。

## 2. 哪些是原仓库代码

这里的“原仓库”以本地缓存 `origin/main` 和 Git 加入历史为参照，不表示联网检查了最新远端，也不表示逐一追溯项目最初提交。

原仓库已有：`agent/runtime/planner.py`、`agent/backends/planner.py`、`agent/runtime/runtime_assembly.py`、`agent/runtime/memory*.py`、`agent/runtime/visual_history.py`、`agent/tools/sim_mcp.py`、`sim/mcp_server/*`、`sim/bench_worker.py`、`sim/env_registry.py`、`adapter/protocol.py`。其中一些文件后来有本地修改，不能认为当前内容全部与原版一致。

后续新增：`agent/runtime/two_tool_input.py`、`two_tool_xml.py`、`planner_xml.py`、`embodied_two_tool.py`、`adapter/maniskill_mcp.py`、`agent/training/*`、Swift 插件、`baseline/`、新数据采集与专家合成、`SFT-train/`、`visual/`。

Git 依据：`541d3d4` 加入 portable staged Swift 训练；`a97439f` 加入两工具统一输入/输出、条件逐轮训练等模块。

原 OpenETA 就有 XML `<decision>` 决策及上下文/视觉历史能力。`two_tool_xml.py` 是后加的受限两工具校验层；`two_tool_input.py` 是后加的评测/训练输入构造器，不等于原通用 planner 的完整上下文系统。`planner_xml.py` 是后来把原 planner XML 解析能力抽出的模块。

## 3. 原 OpenETA 框架怎么读

从 `agent/runtime/runtime_assembly.py` 看组件组装，再看 `agent/runtime/planner.py` 的决策、上下文构造和工具执行入口；随后看 `agent/backends/planner.py` 的真实模型请求与图片附件。

再读 `agent/runtime/memory.py`、`memory_store.py`、`visual_history.py`，理解完整证据落盘与有限模型上下文的区别。最后沿 `agent/tools/sim_mcp.py` → `sim/mcp_server/server.py` → `worker_mgr.py` → `sim/bench_worker.py` 跟到实际环境。

## 4. 新链路公共协议怎么读

在对应 checkout 中读 `agent/runtime/two_tool_input.py`：`make_profile` → `snapshot` → `robot_delta` → `execution_feedback` → `build_input`/预算构造函数。

重点：相机与动作契约、前后观测身份、实际本体状态差分、执行结果、四组历史、有序图片和可见字段白名单。

再读 `agent/runtime/two_tool_xml.py`：`parse_decision`、动作契约和参数校验。工具只有 `observe_env`、`step_env`，结束用 `task_complete`；格式正确、执行正常、任务成功是不同概念。

三个 checkout 的同名文件存在实现差异。RL 添加了灵巧手/双臂任务能力，baseline XML 接口也不同。必须记录实际代码 revision 和 profile hash，不把同一个版本字符串当成代码完全相同的证据。

## 5. Baseline 评测

所有以下路径从 `.worktrees/baseline/` 开始。

阅读顺序：

1. `baseline/README.md` 的统一输入 v1 部分。
2. `baseline/manifests/all_40.json`、`baseline/configs/input_profiles.json`、模型配置。
3. `baseline/scripts/run_xml_parallel_campaign.py` 与实际使用的 campaign 启动入口。
4. `baseline/runtime/run.py` 的 `run_episode`。
5. `baseline/runtime/provider.py`、本 checkout 的公共输入和 XML 模块。
6. `baseline/snapshots/two-tool-input-v1/` 的提示词与冻结 lock。

在 `run_episode` 中跟创建/重置 → 统一输入 → provider → XML 校验 → MCP 执行 → 实际步数/原生成功证据 → 历史更新 → 结束和清理。异常、预算用尽、模型主动停止、原生成功分别落盘。

选一条 `result.json` 对照 `inputs/NNN.json`、`exchange.attempts[].request_body`、`validation`、`execution`、`native_success`。`native_success_observed` 不等于成功；汇总应按评分代码和可信环境证据判断。

## 6. 数据构造

### 在线模型采集

`run_flash_expanded_sft.py` 等 campaign 调度 → `collect_two_tool_sft.py:collect_attempt` → `export_two_tool_sft.py:samples_from_episode`/`swift_messages`。

模型调用重点看 `_model_request`、`_provider_wire`、provider 专用适配；重试只重发同一观察的模型请求，不能盲目重发已经可能执行的环境动作。

选一条成功 episode：核对实际输入包/请求、原始 XML、校验参数、工具回执和官方成功证据，再看导出 JSONL 是否保留同一条件上下文与图片顺序。

### Artificial 专家合成

专家来源准备脚本 → `replay_expert_two_tool_sft.py:replay_one` → `compress_expert_replays.py` → 压缩候选的独立重放 → `annotate_expert_sft_reasoning.py`/`annotate_compressed_expert_sft.py` → 审计和导出。

专家动作先在真实环境重放，成功后导出；压缩改变动作执行分组，需重新验证成功，不能继承原成功标签。模型补写只修改 reasoning，必须核对源样本 hash 和补写前后动作等价。补写是事后说明，不是专家原始思维。

读 `audit_expert_replay_success.py` 和 `audit_annotated_expert_sft.py`，理解数据可接受的证据。原始专家、压缩专家、补写专家及在线采集必须有明确来源。

## 7. SFT 训练

早期五轨迹实验见 `SFT-train/README.md`；当前混合主线看 `SFT-train/MIXED_TRAINING.md`。

阅读顺序：`prepare_mixed_data.py` → `audit_mixed.py` → `launch_mixed.py` → `train_mixed.sh` → `export_mixed.py` → `verify_mixed_exports.py`。

数据准备：指定专家 reasoning 和在线模型成功来源，排除长专家轨迹、按 episode/turn 去重、检查成功证据/图片/监督标记。当前落盘统计 74 episode、3521 样本：LIBERO 1855、MetaWorld 1242、ManiSkill 326、RoboCasa 98。

编码审计：用实际 Swift 模板验证当前 XML 被监督、空 think 未监督、图片和占位符匹配。目标是本轮 XML；system/user/历史 assistant 不计算目标 loss。专家 reasoning 的补写来源不因训练而改变。

训练：三个基座分别 LoRA，语言模块训练、视觉编码器/aligner 冻结；读实际 GPU、全局 batch、累积步数、长度策略和 checkpoint 保存参数。没有独立验证集，loss 下降只说明拟合；需要环境评测。

注意：`prepare_mixed_data.py` 的 Flash RoboCasa 排除条件比注释“一条 StackCans”更宽；`export_mixed.py` 硬编码 global_step=1320，换数据/训练步数后要核对，不直接运行。

## 8. RL 训练

所有以下路径从 `.worktrees/rl/` 开始。最新 mixed 模型课程见 `docs/single-model-rl-curriculum-mixed.zh-CN.md`；旧 Alice/Bob、formal_experiment2 及旧 32 任务重启是历史路径。

阅读顺序：

1. `configs/single_model_rl_curriculum_mixed.json`、环境入口脚本。
2. `scripts/run_single_model_curriculum.py`、`agent/training/curriculum.py`：阶段任务、seed、恢复状态、升阶条件与全零奖励停止。
3. `scripts/run_single_model_grpo.sh`：实际 scheduler/reward/trainer 注册、生成/训练预算和环境变量。
4. `plugins/single_model_swift_grpo.py`：`on_trajectory_start` → `_build` → `_turn` → `_save` → `NativeSuccessReward`。
5. `agent/training/single_model_rollout.py:NativeTaskEnvironment`：原生环境创建、指纹、观察、动作和可信成功检查。
6. `agent/training/grpo_input.py:InputBuilder`：各轮统一输入与精确编码预算。
7. `agent/training/conditional_turns.py`、`input_grpo_trainer.py`、`rollout_cpu_sync.py`：条件一致性、逐轮训练和内存/同步。

RL 最难的部分是 `conditional_turns.py`：`split_turn_samples`、`_verify_prompts`、`_prepare_batch_inputs`、`_postprocess_batch`、`episode_token_weights`、`training_step`。理解滑窗生成必须在同一实际上下文下算 logprob，生成前缀与真实生成 token 的区分，变长轨迹的优势路由、episode 权重和 DDP 同步。

原生奖励为任务成功的 0/1；基础设施错误抛出而不是当 R=0。每轨迹交互轮数不是 optimizer update 数；同组样本初态相同、跨组 seed 变化。

最新课程文档采用 mixed SFT 完整模型新建 LoRA，阶段之间恢复完整训练状态。参数可能继续更新，查实际配置/args，不能把之前的 768-token、60轮配置当成所有运行默认。

## 9. 阅读和验证方法

不要一开始完整阅读巨大的 planner 和 Swift 依赖。每条链路先选一个实际产物，反向追查它的形成代码，再读抽象层。

- Baseline：一条 result、实际请求和输入审计。
- 构造：一条 episode、动作成功证据、来源标记、导出样本。
- SFT：一个 JSONL 样本、渲染后的 prompt、input_ids/labels、模型训练 args。
- RL：一条 episode 的逐轮输入、采样 prompt tokens、生成 IDs/logprobs/mask、奖励与训练审计。

相关测试优先读契约而不是盲目执行 GPU 作业：XML/input 测试、专家压缩/补写测试、Swift 编码审计、RL `test_conditional_turn_samples.py`、`test_single_model_rl.py`、`test_single_model_curriculum.py`、`test_grpo_input_context.py`。真实环境/模型验收与纯逻辑测试分开理解。

## 10. 可视化与结果位置

`visual/server.py` 是读取/适配层，`visual/static/app.js` 是列表、播放、模型输入与 reasoning 展示。页面呈现不自动证明原始输入或评分一致。

默认 Baseline 根目录为 baseline worktree 的 `baseline/results`；外部/远程评测必须额外配置结果根目录，并把媒体读取限制在这些已配置目录内。新的存储位置不能靠刷新旧根目录自动发现。

### 2026-10-08 最新 Baseline v2 入口

三模型 mixed 最新 campaign 位于 `OpenETA/.worktrees/baseline/baseline/v2/campaigns/20261008-mixed-112`，不是旧 `baseline/results`。优先读该目录的 `campaign.json`、`models.json`、`start.py`、`evaluate.py`、`backend.py`、`initial_guard.py` 和 `supervise.py`；它们是本次运行的实际编排、模型适配和初态检查入口。正式轨迹在 `formal/<模型>/<任务>/attempt-*/`，canary 单列在 `canary/`，recovery 是历史诊断/恢复证据，不混入正式成绩。查看器已经增加该 v2 campaigns 的发现，按 v2/campaign/formal 或 canary 分组。

`evaluate.py` 实际导入 `baseline.v2.runtime`，并绑定该 campaign 的模型后端和 `verify_accepted_initial`。最新 v2 评测主循环应读 `OpenETA/.worktrees/baseline/baseline/v2/runtime.py`，而 `baseline/runtime/run.py` 用于理解旧输入 v1 链路。两者不能只因字段相似而视为同一执行入口。
