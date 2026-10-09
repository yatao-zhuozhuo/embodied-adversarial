# 四个 benchmark 的专家示范转两工具 SFT 数据方案

日期：2026-10-07。状态：**实施计划，尚未下载数据、重放示范、修改采集/导出代码或启动训练**。

目标是减少让在线大模型反复试错的 API 成本：从可合法使用的成功专家示范出发，在 OpenETA 中重放并重建每轮模型可见输入，把实际执行的原生动作写成 `openeta.two_tool_xml.v1` 的 `step_env` 决策。第一版只监督动作和工具协议，`<reasoning>` 留空；事后生成的简短解释仅作为独立、可选的后续实验。这里的“专家轨迹”不是 GLM/Qwen 在线规划轨迹，也不应伪装成它们的原始回答。

## 1. 范围与不可跨越的边界

- 第一阶段沿用 [20 题诊断清单](../SFT_data/configs/deepseek_xml_debug5.v1.json)：MetaWorld、LIBERO、RoboCasa、ManiSkill 各 5 题。清单是候选范围，不保证每题都有兼容的专家示范或能成功重放；缺口必须报告，不能补造成功样本。
- 模型输出和训练目标仅为当前 [两工具 XML 协议](../agent/runtime/two_tool_xml.py)：`observe_env`、`step_env` 和非工具的 `task_complete`。不同 benchmark 的动作维度、控制器、相机 profile 不强行统一。
- 训练输出为 train-only JSONL；不必从构造的数据中再切一个验证文件。但必须严格排除 [项目测试保留清单](../SFT_data/configs/two_tool_test_holdout.v1.json) 的任务、任务别名及保留 seed，后续独立评测不能使用训练示范的同一初态或轨迹。
- 不修改评测协议来迁就示范，不把高层专家步骤仅改名为低层 `step_env`，不让学生看到宿主隐藏的对象真值、奖励、未来观测或专家内部状态。
- 来源、版本、许可/再分发条件、控制模式及文件哈希必须入账。未经核准再分发的原始示范留在受控本地目录；导出训练集时遵循相应数据许可。

### 1.1 四个 benchmark 的划分如何落地

| Benchmark | 官方数据/评测语义 | 本项目的训练边界 |
| --- | --- | --- |
| MetaWorld | ML10/ML45 有训练任务与未见测试任务；MT50/单任务专家策略不是自动划分。[官方 benchmark](https://metaworld.farama.org/introduction/basic_usage/) | 当前 `metaworld_50` 候选任务仍按项目 holdout 排除；报告时说明是“项目任务 holdout”，不冒称 ML10/ML45 官方测试分数。 |
| LIBERO | 各套件有任务示范，并提供固定初始状态用于评测；训练示范与评测可属于同一任务。[官方 README](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/README.md) | 排除保留的完整任务及其所有示范；非保留任务也避开保留 seed/初态，并按任务和初态记录去重键。 |
| ManiSkill | 官方示范随任务提供；观测/控制模式可以重放转换，个别任务可能因非确定性重放失败。[官方重放文档](https://maniskill.readthedocs.io/en/latest/user_guide/datasets/replay.html) | 按任务键、对象实例/场景和 seed 审核；当前项目 holdout 高于“示范文件可下载”这一事实。 |
| RoboCasa365 | 区分 pretrain/target 数据与场景；Atomic-Seen、Composite-Seen、Composite-Unseen 是相对预训练数据的任务分组，不是统一的“禁止训练测试集”。[官方数据](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html)、[官方评测](https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html) | 第一阶段只考虑清单中的 `pretrain_*` 任务；保留任务及其别名全部排除，并记录实际场景 split，不能把 target 场景混进 pretrain 评测口径。 |

开始前先冻结 holdout 文件哈希，并用规范化任务键、环境 ID、任务描述和 RoboCasa `pretrain_`/`target_` 去前缀键做交叉检查。早先 `SFT_data/accepted/{glm,qwen}` 的 ManiSkill 数据含 PickCube/StackCube，而这两题在当前 holdout 中；若沿用这份 holdout，旧数据必须隔离，不能因为它们已经标了 `train` 就混入新训练集。评测最终使用哪种口径，应在训练前记录：项目内 held-out 任务/seed、官方协议，或两者分别报告，不混写结果。

## 2. 示范来源与可行性预检

| Benchmark | 优先来源 | 首先核对的问题 |
| --- | --- | --- |
| MetaWorld | 官方逐任务脚本专家策略，直接在指定环境生成成功轨迹。[官方专家用法](https://github.com/farama-foundation/metaworld/blob/main/docs/benchmark/expert_trajectories.md) | 当前安装版本的任务 ID、策略与环境是否对应；专家是否依赖学生不可见的状态；动作与 OpenETA 4 维控制契约是否同义。专家可用特权信息产生标签，但特权信息不得进入学生输入或解释文本。 |
| LIBERO | 官方任务示范 HDF5、任务 BDDL 与固定初态。[官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) | 示范原始动作、控制频率、夹爪方向、相机和 OpenETA 7 维动作是否一致；能否从原始状态确定性重放。 |
| ManiSkill | 官方示范或官方运动规划解法。[示范下载/设置](https://maniskill.readthedocs.io/en/latest/user_guide/learning_from_demos/setup.html)、[规划示例](https://maniskill.readthedocs.io/en/latest/user_guide/demos/scripts.html) | 原始示范可能没有观测图像，需重放补录；控制模式转换可能失败，必要时使用环境状态重放，但导出的动作仍须能在目标 OpenETA 控制器下独立执行并成功。 |
| RoboCasa365 | 官方 pretrain 人类/合成示范。[官方数据概览](https://robocasa.ai/docs/build/html/datasets/datasets_overview.html) | 机器人、底盘/手臂控制器、12 维动作布局、场景与相机是否与 OpenETA 一致；若无法无损转换或重放，标记该任务不可用，不数值插值“凑”动作。 |

预检表以每个候选任务为单位，记录 `source_present`、license、源版本、示范数、控制模式、动作维度/界限、reset 可复现性、相机、OpenETA 官方成功回执、失败原因。下载并不等于可用；只有完整重放成功的轨迹才能进入 SFT。优先使用本机已有且许可允许的示范，避免重复下载。

## 3. 构造管线

```text
来源/许可清单 → holdout 与重复项过滤 → 控制/相机/初态预检
  → 在 OpenETA 目标环境重放专家动作并记录真实观测、回执
  → 官方成功验收 → 将动作事件映射到 XML 决策
  → 用统一输入构造器重建每轮真实可见上下文
  → 协议解析、重放一致性、泄漏与 Swift loss 审计
  → 独立 provenance 的 train-only SFT JSONL
```

### 3.1 先重放，再贴标签

1. 保存原始 `benchmark/version/task/source_episode/scene/object/reset state/seed/control mode/action array` 与文件 SHA256。为每个源 episode 分配稳定 ID；重复下载、重复转换不重复计数。
2. 在目标 OpenETA 环境恢复源初态，按源时间顺序执行动作，记录**动作前**相机帧、本体状态、实际工具请求、执行步数、动作后回执与最终官方 `success`。仅源文件自称成功不足以验收。
3. 控制模式不同先做有定义的转换，再在目标控制器重新执行。既要检查数组维度/上下界，也要检查坐标系、归一化尺度、夹爪正负方向和控制频率；不能仅按维度相同认定兼容。转换失败、场景不一致或重放不成功保留审计，不进入训练。
4. 固定环境、源码、资产、相机与输入 profile 的版本/hash。图像必须来自真实重放，按 [统一输入规范](trajectory-input-unification.zh-CN.md) 生成；不得用源示范中不同视角的图片冒充 OpenETA 当前观察。

### 3.2 原生动作到两工具决策

- 对每个实际执行的控制动作 `a_t` 构造 `step_env(action=a_t, num_steps=n)`，用现有 `serialize_decision` 和 `parse_decision` 严格往返校验。`n` 只能是 1–10 且不得超过剩余步数；只有连续、数值相同且控制器语义相同的动作才可合并为重复步数。逐步变化的动作不能任意压缩、抽帧或猜一个代表动作。
- 初始图像已由宿主提供，不凭空插入 `observe_env`。只有源轨迹中确实存在需要独立观察、且目标运行时也能产生相同可见证据的决策，才考虑该标签；否则以 `step_env` 为主。
- `task_complete` 是响应，不是成功证据；只在官方成功回执之后输出，且不把最终成功标签提前放进动作前输入。若运行时成功后本来不会再请求模型，是否需要训练 `task_complete` 单独做协议 canary，不能凭空补一个学生永远不会遇到的决策轮。
- 每条 SFT 样本使用**该动作之前**的真实输入包：任务、当前相机/机器人状态、预算、有限历史及之前执行回执。目标只包含本轮 XML；未来状态、后验成功与完整专家轨迹不进入提示词。长专家轨迹可能产生大量决策轮，这是本方案仍需测量的序列成本，不能把 400 个不同动作伪装成 40 个 `num_steps=10` 调用。

### 3.3 Reasoning 的处理

第一版把 XML `<reasoning>` 留空，遵循现有 [Swift 监督配置](../SFT_data/configs/two_tool_xml_sft.v1.json) 的 current-decision-only / answer-only 原则，不生成 `<think>`，也不要求 GLM/Qwen 再在线执行一遍。能否学会合法工具调用和任务动作，先用独立评测验证。

若动作版确有收益，再小规模尝试“事后动作意图”标注：标注器只看该轮**动作前的学生可见输入**和已固定的动作标签，输出一句可被观察支持的简短理由；不得读取动作后图像、成功/奖励、专家内部目标坐标或整条未来轨迹。该文字标记为 `posthoc_action_rationale`，不是专家原始 reasoning，更不是隐藏 `<think>`。无根据、与动作不符或把未来结果写成既定事实的标注直接丢弃；训练时与空 reasoning 数据做消融，而非默认混合。

## 4. 数据格式、来源隔离与导出

建议新增与现有在线教师数据**物理隔离**的目录，例如 `SFT_data/expert_replay_v1/`，下设 `sources/`（仅清单/引用，不默认复制受限原始数据）、`replays/`、`audit/`、`exports/train.v1.jsonl`、`reports/`。每个 episode 保存至少以下字段：

```text
schema_version, episode_id, bench, task_id, env_id, seed, split=train,
source_kind=expert_replay, source_dataset, source_version, source_episode_id,
source_sha256, source_license_ref, scene_id/object_id/reset_fingerprint,
control_mode_source, control_mode_target, conversion_method,
input_protocol, input_profile_hash, model_protocol,
replay_success, official_success_receipt, action/observation alignment hashes,
reasoning_source=empty | posthoc_action_rationale, posthoc_annotation,
source_episode_path, per-turn messages/images/XML/tool_request/tool_receipt
```

沿用逐轮 Swift `messages`＋有序 `images`，只监督当前 assistant XML，历史回答和环境回执不重复计入 loss。图片文件哈希与训练输入顺序须一致，训练时启用超长样本显式报错，不静默截断。当前 [在线教师导出器](../scripts/export_two_tool_sft.py) 只接收 `output_source=original_model_xml` 等原始在线模型证据，并把导出样本标成 `reasoning_source=original_model_response`；实施时应做**独立的专家重放导出器和 schema**，不能改写来源字段去绕过它的审计。旧 GLM/Qwen 在线数据也不得自动并入新目录。

## 5. 执行阶段、数量与停止条件

| 阶段 | 计划产物 | 继续条件 |
| --- | --- | --- |
| A. 清单与可行性 | 冻结 20 题、holdout、官方来源链接/许可、版本和控制契约；每题一份预检结果 | 可解释每个任务是否可重放；不把缺源任务记作“待模型努力”。 |
| B. 四环境 canary | 每 bench 先选 1 题、每题至少 2 条不同初态的完整成功重放；保存输入、XML、回执和图像 | 四环境各通过协议与 Swift 编码检查；失败环境先修控制/重放，不批量扩张。 |
| C. 小批量 | 若全部 20 题可用，先每题 5 条，目标 100 条**成功 episode**；不可用任务列缺口，不用同题重复轨迹填数 | 统计每题成功率、每条控制步/决策轮、GPU 时间、磁盘、转换失败原因；抽检图像和泄漏边界。 |
| D. 扩量 | 仅在 C 有效且训练可消费时，建议上限每题 25 条，目标 500 条成功 episode；任务配额可根据可用性经人工确认后调整 | 每条均独立官方成功；按 episode 而非 SFT 行数计数，不能用同一示范的多轮样本冒充多条轨迹。 |
| E. 训练对照 | 动作-only SFT 对照未训练模型；可选再测少量事后解释 | 在从未用于训练的任务/seed 上报告成功率、合法 XML 比例、动作执行率和成本；不在 holdout 上调参到过拟合。 |

第一版不预设“两位 API 教师各 500 条”：专家重放数据是**一种**独立来源，与模型身份无关。数量目标应服从任务覆盖和真实成功率；若 20 题无法达到 500，报告实际值和缺口，不降低成功门槛。主要成本从在线教师 API 转移到示范获取、仿真重放、图像存储和审计；可选解释的 API 成本单独预算、单独报告。

## 6. 最终验收清单

1. 训练集中没有 holdout 任务、任务别名、保留 seed、相同 reset fingerprint 或已知测试轨迹；旧 ManiSkill 在线数据不误混。
2. 每个 episode 的源文件与许可可追踪，目标环境官方成功为真；失败、非确定性重放、图像/控制不匹配只进审计。
3. 每轮 `step_env` 参数与实际执行请求相同，XML 严格解析，动作前输入与重放当时的学生可见证据一致，图片哈希和顺序可复核。
4. Swift 模板中只有当前决策是 loss target；来源和 `reasoning_source` 可区分，绝不把事后解释标为在线模型原文。
5. 报告同时给出成功 episode 数、逐轮 SFT 样本数、任务覆盖、转换/重放失败分布及 GPU/API/磁盘成本；四种计数不互相替代。

本文件只规定实施路径与验收门槛；不表示四个 benchmark 的示范已全部下载、许可已逐项核准、所有 20 题均可转换，或新数据已经生成。
