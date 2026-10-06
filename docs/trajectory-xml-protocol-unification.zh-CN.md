# Baseline、SFT 与 Swift GRPO 的轨迹协议统一方案

日期：2026-10-05。状态：**已确定目标方向，待实施和验收**。

本文记录本次讨论确定的方案：以 OpenETA 原主 planner 的 XML `<decision>` 为模型输出协议，统一 baseline 评测、新版两工具 SFT 和 Swift 多轮 GRPO。本文不表示代码已完成迁移，也不改变正在运行的实验、历史数据或 checkpoint。

输入统一的详细目标与三个链路的实施要求见 [多轮输入统一方案](trajectory-input-unification.zh-CN.md)。其按 benchmark 配置相机的规则取代把所有环境图片数量和分辨率一概固定的设想；两份文档都需区分目标设计与实际验收状态。

## 1. 决策及依据

**统一模型输出为 XML `<decision>`；模型可调用工具仍只有 `observe_env` 和 `step_env`。** SFT 文件继续使用 JSONL，训练继续使用 Swift，环境执行继续经过 OpenETA 仿真 MCP。

这是为了与 OpenETA 原实现高度对齐，不是认为 XML 天然比 JSON 或原生 function calling 的学习效果更好。XML 有标签开销和格式错误风险，实际效果需要实验验证。

核查依据：

- [原 planner 解析器](../agent/runtime/planner.py) 中 `_parse_backend_payload` / `_parse_xml_decision` 明确将主模型字符串作为 XML 解析；代码注释明确不静默接受旧 JSON 文本。宿主内部的字典输入属于另一条路径，不代表模型可自由输出 JSON。
- `_agent_owned_tool_planner_system_prompt` 规定 `kind / name / reasoning / parameters`，以及数组的 `<item>` 编码。[后端提示词](../agent/backends/planner.py) 也使用此约定。本地缓存的 `origin/main` 与此一致；未把它表述为联网确认的最新远端状态。
- 本机 `/opt/openeta-swift` 安装的是 `ms-swift 4.4.2`。其 `MultiTurnScheduler` 支持自定义交互逻辑、生成 token IDs 与 loss mask，不强制模型必须生成 JSON 或 API `tool_calls`。当前 [GRPO 插件](../plugins/embodied_swift_grpo.py) 已使用这套扩展机制。
- Swift 官方说明支持标准 `messages` 数据和自定义多轮 rollout：[数据格式](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html)、[多轮训练](https://swift.readthedocs.io/en/latest/Instruction/GRPO/DeveloperGuide/multi_turn.html)。这些链接指向滚动更新的文档；具体参数、模板行为与兼容性以安装的 4.4.2 和本项目验证为准，不假设新版文档中的自动转换能力在本机全部存在。

本方案更新了 [两工具 self-play 设计稿](embodied-selfplay-two-mcp-tools-design.zh-CN.md) 中“采用简化 JSON 命令”的目标选择。该旧文档保留为历史背景；其宿主管理、模型可见信息、MCP 执行边界仍需保留。本文不覆盖其所有训练设计。

## 2. 三个层面必须区分

| 层面 | 统一目标 | 说明 |
|---|---|---|
| 模型输出 | XML `<decision>` | 模型实际学习和生成的文字协议 |
| 宿主调用工具 | 工具名与经过校验的参数，调用 MCP | XML 不直接发送给模拟器；宿主解析后构造工具请求 |
| 数据文件 | JSON/JSONL 中保存原始文本、结构化动作与证据 | 文件是 JSON，不意味着模型需要生成 JSON |

“工具名相同”不等于训练协议相同；“统一模型协议”也不要求所有历史文件立即使用同一个 schema。

统一 XML 仅对齐原 OpenETA 的决策表达方式。当前任务仍是两个低层环境工具的受限策略，不自动获得原通用 planner 的其他工具、技能、记忆或规划能力。

## 3. 当前状态及需要迁移的位置

当前工作目录分支为 `feat/SFT-data`；baseline 位于 `.worktrees/baseline`，分支为 `feat/baseline`。

| 链路 | 当前模型输出 | 当前主要记录格式 | 改动方向 |
|---|---|---|---|
| Baseline | XML decision，但动作数组示例允许 JSON 字符串 | `rgb_native_actions_v1`；`turns`、图像附件、最终观测 | 对齐原生 XML 数组、共享解析/校验、新协议版本与运行目录 |
| 新版 SFT `qwen_live_v1` | API 原生 `tool_calls`，参数为 JSON | `openeta.qwen_live_two_tool.v1`；`turns`、`model_calls`、多相机帧 | 新采集直接生成 XML；历史证据另行导出为 XML 训练样本 |
| `runs/formal_experiment1` 训练 | `{"tool":"step_env",...}` / `{"response":"finish"}` | `openeta.embodied_rollout.v1` + `openeta.embodied_two_tool.v1` | 替换动作文本解析、提示词与结束响应；保留 MCP、奖励与调度 |

相关入口：

- Baseline：`.worktrees/baseline/baseline/runtime/run.py`、`baseline/snapshots/native-actions-v1/system.txt`、快照 `lock.json`。
- SFT：[collect_two_tool_sft.py](../scripts/collect_two_tool_sft.py)、[export_two_tool_sft.py](../scripts/export_two_tool_sft.py)。
- 训练：[embodied_two_tool.py](../agent/runtime/embodied_two_tool.py)、[embodied_swift_grpo.py](../plugins/embodied_swift_grpo.py)、[maniskill_mcp.py](../adapter/maniskill_mcp.py)，以及 staged-colocate 的启动/数据生成流程。
- 展示：[visual/server.py](../visual/server.py) 目前只做读取适配，尚未统一原始模型协议。

旧版 ManiSkill SFT 是独立历史资产，可能包含不同工具和动作语义，**不纳入直接格式转换**。不能把高层工具动作仅改名成 `step_env`。

## 4. 目标模型协议

建议将新协议标识为 `openeta.two_tool_xml.v1`。这是拟采用的新标识，尚未注册或实现；与旧 JSON 协议必须可区分。

### 4.1 观察

```xml
<decision>
  <kind>tool_call</kind>
  <name>observe_env</name>
  <reasoning>Inspect the current scene.</reasoning>
  <parameters/>
</decision>
```

`observe_env` 没有模型可控参数，不移动机器人。

### 4.2 执行动作

下面是 Panda / PickCube 的 7 维动作示例，不适用于所有环境：

```xml
<decision>
  <kind>tool_call</kind>
  <name>step_env</name>
  <reasoning>Move slightly toward the cube.</reasoning>
  <parameters>
    <action type="array">
      <item type="number">0.2</item>
      <item type="number">-0.1</item>
      <item type="number">0</item>
      <item type="number">0</item>
      <item type="number">0</item>
      <item type="number">0</item>
      <item type="number">1</item>
    </action>
    <num_steps type="integer">3</num_steps>
  </parameters>
</decision>
```

统一协议使用 XML 数组；新训练样本不再输出 `<action>[0.2,...]</action>` 这种数组字符串。历史 baseline 的这种编码仅在显式历史转换中处理。

参数约束：

- `action` 为有限数值组成的数组，不接受布尔值、NaN、Infinity。
- 维度和上下界由宿主创建环境所得的动作契约确定；当前配置常见维度为 MetaWorld 4、LIBERO 7、ManiSkill 7、RoboCasa 12。不能把训练侧 PickCube 的 7 维限制复制到全部 benchmark。
- `num_steps` 必填，严格为 1–10 的整数，并受剩余控制步预算限制。这里指环境控制步，不是底层物理子步；环境提前终止时以回执中的实际执行步数为准。
- 不允许宿主静默裁剪、补零、截断、随机补动作或替模型修正参数。
- 不允许模型传入 `handle`、`session_id`、环境 ID、快照路径、`strict` 等宿主管理参数。

### 4.3 主动结束

```xml
<decision>
  <kind>response</kind>
  <name>task_complete</name>
  <reasoning>The attempt is complete.</reasoning>
  <parameters/>
</decision>
```

结束是非工具响应，不调用 MCP，也不等于成功。Baseline/SFT 仍由环境可信回执判定成功；Alice 仍需编译和有效性检查，Bob 仍需目标检查。旧训练 `finish` 在转换中映射到此形式。

### 4.4 解析、推理文本与无效输出

- 每轮只接受一个完整 decision，限定 `kind/name` 组合；不开放 `tool_batch`、其他工具或其他 response 类型。
- 复用原 planner 的 XML 解析能力，并在共享入口补充两工具限制、唯一 decision、重复字段、参数类型与范围校验。原通用解析器能解析某种输入，不等于两工具契约允许执行它。
- 通用解析器已有有限 XML 修复能力。迁移实现必须显式确定是否允许修复，三条链路保持一致，并记录修复标记；不得一侧修复、一侧判错。建议正式严格评测把发生修复的输出归为格式错误，原始记录仍保留供审计。
- `<reasoning>` 可以为空；多行文本和 XML 特殊字符使用规范转义或 CDATA，序列化器必须正确处理 `]]>`。
- 启用 thinking 的模型可以在回答前生成模型模板规定的思考段，但最终回答仍必须是一个 XML decision。保留原始 thinking/answer，不把未闭合思考段、截断输出或嵌在思考内容里的 decision 当成可执行动作。
- `<think>` 与 `<reasoning>` 不要求重复同一段内容。已有原始推理文字可以按明确规则映射；不得事后编造解释。
- 无效输出记录错误，不执行环境动作。无效轮次如何消耗预算、是否重试及如何惩罚必须统一配置并写入运行元数据。

## 5. 共享实现边界

建议把已有 XML 解析器整理为可供三条链路调用的稳定公共接口，而不是分别复制三份或长期跨模块调用私有函数。共享层负责：

1. 分离模型模板中的 thinking 与最终回答，检查完整性。
2. XML → `kind/name/reasoning/parameters` 的结构化 decision。
3. 两工具白名单、结束响应及参数/预算校验。
4. 规范 XML 序列化，供 SFT 导出及往返检查使用。
5. 生成统一的校验结果与执行请求；不承担模型选择动作、奖励计算或环境生命周期管理。

通用 OpenETA planner 的其他工具不能因本次两工具限制而失效；限制放在两工具协议层，通用 XML 解析基础能力保持可复用。

宿主仍拥有 create/reset/close、快照保存恢复、目标设置、成功检查、奖励、审计等能力。模型可见工具表只描述两个工具。生产执行前的严格动作校验不能仅依赖宽松的仿真底层。

## 6. Baseline 如何统一

- 系统提示词、动作数组示例、结束方式切换到本协议，使用共享解析与校验。
- 原始 XML、解析结果、实际 MCP 参数、错误与图像证据分别保存，避免只有文本而无法确认实际执行内容。
- Baseline 当前有输入快照 hash 校验。协议/提示词修改必须建立新的协议快照与 lock，不能直接改被冻结的旧快照导致旧实验不可复现。
- 新建结果目录，并记录模型、协议/提示词版本、动作契约、观测配置和代码 revision。协议改变后的成绩不能无标注地拼接进旧 baseline 总分。
- `success` 仍以评分逻辑为准。当前 baseline 的 `native_success_observed=true` 表示读到了有效 native success 证据字段，**不是“环境成功过”**；不能用它替代 `success` 或回执中的 `ever_success`。

## 7. SFT 如何统一

### 7.1 新采集

模型直接输出 XML assistant 文本；工具目录作为提示词中的可用工具契约提供，避免同时要求 API 原生 `tool_calls` 和 XML 两种输出。宿主解析、校验、调用 MCP，把经过过滤的观测和执行反馈作为下一轮输入。

新采集与训练使用相同的协议示例和错误反馈规则；完整记录模型请求、原始回复、图像、解析结果与真实执行参数。协议升级后使用独立 schema/version 或新目录，不能让旧读取器误判。

### 7.2 既有 `qwen_live_v1` 转换

保留原始 `episode.json`、图片和模型回复；另行生成 XML 训练集。每个样本同时保留来源 episode、turn、原始输出形式、转换器版本和目标协议。

转换步骤：

1. 核实原模型 tool call、记录的工具参数和实际执行回执一致。
2. 从原生 tool-call 的函数名及参数构造规范 decision；只接受语义可精确保留的两工具调用。
3. 真实 `reasoning_content` / `reasoning` / `content` 分别保存。映射规则确定且可追踪；没有解释时留空，不生成新理由。
4. 同时转换目标回复、历史 assistant 动作、工具回执的消息布局和提示词。不能只改最后一条回答，却留下“请使用原生 tool_calls”的原提示词。
5. 对转换后的 XML 做“序列化 → 共享解析/校验 → 比较原始动作”检查；动作、步数、顺序不变。
6. 不补造未发生的结束响应、不变更成功标签、不把失败轨迹转换成成功样本。

转换后的 XML 是**由真实动作证据构造的训练表示**，不是老师原样输出。增加例如 `output_source=converted_tool_call`、`source_protocol`、`conversion_version` 等标记，不能沿用“XML 为原始模型回复”的描述。

### 7.3 Swift SFT 数据与监督范围

使用 Swift 标准 `messages` + `images` JSONL；`assistant.content` 保存 XML。多模态消息按本机 Swift 模板要求转换并核验图片顺序，不假设现有 API 请求 JSON 可以直接用于训练。

对于 XML 模式，不让默认 agent template 再把动作自动格式化成另一种 function-call 文本；保留模型本身所需的聊天/视觉模板，并固定模板版本、thinking 配置与渲染后的实际 prompt。

监督方案二选一并写入配置：

- 完整多轮 episode：对所有真实 assistant 目标训练，环境观测与宿主插入内容不训练。
- 逐决策样本：历史仅作为上下文，只监督当前目标回答，避免重复监督历史回复造成权重偏移。

现有导出中的 `metadata.target_message_index` 只是元数据，**不会自动让 Swift 只监督该消息**。必须通过实际支持的 loss 配置/预处理实现，并检查编码后的 labels/mask。保留现有 holdout seed、任务边界及来源筛选规则。

## 8. Swift GRPO 如何统一

- 将当前 `parse_command` 的简化 JSON 分支替换为共享 XML 入口，更新两工具 prompt 与 `finish` 处理；scheduler 内部仍可使用解析后的字典。
- 继续通过 `McpPickCubeAdapter` 执行，不因输出格式变化绕过 MCP、改变动作控制器或奖励定义。
- Swift 优化的是模型实际生成的 token。必须保留对应的原始 `response_token_ids`、logprobs 与 mask；**不能把 rollout 生成的 JSON 事后转成 XML，再配原 JSON 的 logprobs 做 GRPO**。
- 模型生成的思考/回答是否训练，按固定策略处理；环境回执、下一轮图像提示、宿主错误文字不进入模型生成部分的 loss。
- 格式错误与基础设施错误分开记录。XML 迁移不应暗中改变 reward、成功判定、rollout 预算或同组初态恢复条件。
- 同步修改实际使用的 staged-colocate、数据生成和评测入口，不能只改一个未被正式训练调用的原型脚本。
- 旧 JSON-policy checkpoint 可以作为协议适配的初始化候选，但需先做 XML SFT/格式适配和 canary；不能假定换提示词后直接恢复原训练即可稳定工作。新训练使用独立输出目录并标记初始化来源。

## 9. 观测、终止与运行配置也要对齐

目前三条链路还存在观测与提示词差异，例如 baseline 主图 512 分辨率、新版 SFT 256，以及训练按实验配置采样的小分辨率图像。首次观测也不同：baseline/训练可由宿主预先提供，新版 SFT 要求模型先调用 `observe_env`。

目标默认采用**宿主取得初始观测后开启模型第 1 轮**，模型后续仍可选择观察。这是新的运行配置选择，应显式版本化；转换历史 SFT 时保留原来真实发生的首次 observe，不为迎合新默认而伪造或删除动作。

为可比实验固定并记录：相机/分辨率/预处理、模型可见本体状态、历史窗口、thinking、生成预算、首次观测策略、无效动作处理、控制步预算、结束响应。不同 benchmark 可有不同动作契约，Alice/Bob 可有不同任务文本与目标检查，但不能让模型意外看到宿主专用的 object/goal 坐标、深度、奖励或评分信息。

Alice 生成任务与 Bob/baseline 解题的成功定义不必相同；必须在元数据与界面中说明，不为统一字段而混淆其含义。

## 10. 轨迹存储与可视化

优先统一输出协议与执行语义，不先重写所有历史文件。新记录至少应能找到下列信息；字段位置可由各自外层 schema 承载：

| 信息 | 必须保留的内容 |
|---|---|
| 版本 | 外层 schema、模型协议、prompt/观测配置版本、代码 revision |
| 原始回复 | thinking/answer/provider 回复及来源，不被规范 XML 覆盖 |
| 结构化动作 | decision、校验结果、实际工具请求、实际控制步数 |
| 视觉证据 | 动作前后帧、camera ID、hash/路径及缺帧状态 |
| 执行结果 | MCP 回执、terminated/truncated、错误类型 |
| 评分 | 环境 reward、最终训练 reward、官方/目标成功证据分开保存 |
| 转换来源 | 仅对转换数据记录原协议、源文件、turn、转换器版本 |

[统一可视化](../visual/README.md) 继续兼容旧 baseline、原生 tool-call SFT、JSON-command 训练；新增 XML 协议显示、解析状态和原始/转换来源。它只负责展示，不给缺失字段补成功标签或生成解释。

## 11. 迁移顺序与验收

1. **共享协议与测试**：实现 XML 序列化/解析/两工具校验；以原 planner 样例回归，同时验证 4/7/12 维动作、结束、空参数、重复字段、错误类型、越界、截断 thinking 和多 decision。三入口对同一输入应产生相同决定或相同拒绝。
2. **SFT 转换与编码验证**：少量成功样本逐条确认动作等价、来源保留、图像对齐、目标 XML 在 labels 中、历史监督范围正确。确认没有模板把 XML 改成另一种工具调用文本。
3. **新采集与 baseline canary**：用新协议快照运行少量固定任务/seed，检查工具执行参数、动作前后帧和结束判定。模型声称完成但环境未成功时不能记成功。
4. **短 SFT + GRPO canary**：先验证模型能稳定生成有效 XML，再跑小规模多轮 GRPO，确认 token/logprob/mask 对齐，工具回执被屏蔽，同组初始状态一致。有效率、奖励和资源用量均记录，不能只看进程未报错。
5. **正式迁移**：发布协议/提示词/模板版本和验收结果，使用新输出目录启动新实验。旧实验按旧配置保留；旧 XML/JSON/function-call 数据不混作一个没有协议标签的数据集。

验收完成前，不删除旧解析器或覆盖历史轨迹，不声称三条链路已经统一。允许显式按协议版本选择历史读取器；新的正式训练/采集/评测必须只使用新协议，不靠自动猜测输出格式兜底。

## 12. 完成标准

统一完成应同时满足：**相同模型输出契约、共享执行前校验、可追溯的 SFT 表示、Swift 训练与推理一致、可比较的观测配置、版本明确的 baseline 结果**。仅把三个工具名改成一致或让页面都能播放，不算完成协议迁移。
