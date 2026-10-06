# GRPO XML 接入与验证

本文件只记录 GRPO 链路；SFT 采集/导出与 baseline 由独立会话维护，不以本文声明三条链路已经全部验收。

## 协议与入口

- 模型协议：`openeta.two_tool_xml.v1`；启动开关：`OPENETA_EMBODIED_PROTOCOL=two_tool_xml_v1`。
- 共享入口：`agent.runtime.two_tool_xml.parse_decision`。支持 `contract={action_dim, action_low, action_high}` / `remaining_steps`，或 `action_low`、`action_high` / `remaining_control_steps`；失败抛出 `ValueError`。
- 成功结果包含 `decision`、原始 `thinking`、`answer`、`valid` 和 `repaired=false`。严格拒绝修复、重复字段、多个 decision、非法参数、未闭合 thinking 与截断输出。
- GRPO scheduler 内部可以保留结构化字典，但模型实际生成、用于优化的文本必须是 XML；不把旧 JSON 生成结果替换成 XML 后沿用旧 logprob。
- `observe_env`、`step_env` 使用真实 MCP。`task_complete` 是 response；不发送工具请求，也不自动获得成功标签。结束时的宿主回放/评分/清理仍独立执行。
- 无效输出消耗一轮、没有控制动作；连续三轮无效结束。超出三次连续观察视为无效。奖励公式、目标成功检查与初态回放合同不变。
- 保留全部模型生成 thinking/answer token 的监督；环境观测、宿主回执与错误反馈不进入生成 token 的 loss。

## 新运行身份

`record_embodied_xml_run.py` 写入 `run_manifest.json`，记录协议、prompt、无效输出策略、主要配置、git revision 与关键源码哈希。已有旧轮次而没有 XML manifest 的目录会被拒绝，不能把旧 JSON 训练直接续在同一路径中。

`formal_experiment1` 是旧 JSON 运行，已按用户要求停止并删除。新实验目录为 `runs/formal_experiment2`。

## 验证证据

- 协议、planner 兼容、base 路由及训练合同测试：73 项通过。
- `runs/xml_protocol_validation/mcp_final`：真实 MCP + 确定性 XML scheduler，Alice/Bob 均完成观察、动作和结束；生成 token 与 logprob 数量均为 154；结束不被记为成功。
- `runs/xml_protocol_validation/model_legacy_init`：真实 Qwen3.5-4B + 原始 LoRA，thinking=true，每次最多 4096 tokens。
  - Alice：3/3 XML 有效，执行 2 个环境控制步。
  - Bob：1/3 有效，后两轮输出截断，未执行控制动作。因此旧 Bob 初始化未通过；该初始化方案已按用户要求弃用。
- `validation_summary.json` 保留机器可读的验收状态和证据路径。以上小样本只验证协议行为，不证明任务成功率或长期稳定性。

## 剩余验收

用户已明确要求 Alice/Bob 都从原始 Qwen3.5-4B 开始，不加载历史 LoRA、不做短 SFT。设置 OPENETA_INITIALIZE_FROM_BASE=true，首轮为两个角色分别创建全新的 LoRA，冻结 Bob 评估直接路由 base 权重；后续轮次只使用本实验 checkpoint。

`model_base` 实测：Alice 3/3 输出有效并执行两个控制步；Bob 1/3 有效，两次输出在 4096 token 截断。此小样本显示 base 的格式遵循尚不稳定，不作为引入历史 checkpoint 或额外 SFT 的理由。训练按协议拒绝无效输出并记录其比例。

`runs/xml_protocol_validation/grpo_base` 已完成八卡 Alice 单步 GRPO 和完整 checkpoint 保存。8 条 rollout 未编译出有效任务，进入预设的 Bob bootstrap 回退。用户要求提交正式实验后结束该验证；Bob 梯度步骤与 holdout 未在 canary 中完成，不宣称完整双角色验证通过，将在正式运行中检查。

正式实验重新从 base 初始化，不继承验证权重。旧 Bob checkpoint 不再是新实验的依赖。历史 bootstrap/holdout 环境任务仍保留，只用于数据与评测，不加载其权重。


## 输入 v1 的 GRPO 接入（2026-10-06，PickCube 链路验收通过）

新运行身份为 `runs/formal_experiment2_nothink_input_v1`，两个角色从 base 新建 LoRA，nothink。旧 `formal_experiment2_nothink` 保留，已停止训练及自动重启；不会将不同输入混写到旧目录。

- 使用 `agent.runtime.two_tool_input` 的 `make_profile`、`snapshot`、`build_input`，与 SFT 共用相机、可见状态与文本历史规则。`agent.training.grpo_input` 仅负责落盘图像、Swift 消息编码和实际 tokenizer 预算。
- 当前仅验收 ManiSkill PickCube/Panda；主图原生 256×256，上一次决策前主图＋当前主图，按像素去重。相机/控制 profile 来自 `create_env.input_control_contract`，不复用旧 action_hint。
- 最多四组历史，不重放隐藏 thinking；裁剪顺序为最老文本组、历史图。保留任务、控制契约、当前必要图像、最新执行反馈；最小输入超限报错。
- `ConditionalTurnsMixin` 先按完整 episode rollout 并完成 Alice/Bob 奖励计算，再按实际决策输入拆成训练样本。每轮共享原始 request_id/prompt_id 和 episode reward；Swift 4.4.2 request-aware advantage 对 episode 去重后归一化，不累加重复奖励。
- 按每卡实际决策数分发全局 advantage，避免 Swift 默认的等长切片假设；逐轮 loss 按生成 token 数恢复每 episode 平均，保持原有 GRPO 权重。
- `--loss_scale last_round` 只监督当前轮原始 token；每条样本强制核验采样 prompt IDs、训练前缀、生成 token 与 loss mask、rollout logprob 长度。未验证的输入拒绝训练。
- `OPENETA_COMPLETION_SCOPE=per_round` 避免 Swift 把变化后的滑窗当成累积对话预算；宿主仍限制整条轨迹生成预算，剩余预算不足下一完整轮时结束。
- 每轮 artifact 保存 input profile/hash、共享输入包、实际 Swift messages/images、prompt IDs、响应 token/logprob/mask、episode/turn 映射。验证状态写入 trainer 目录下 `conditional_input_validation.jsonl`。

离线回归 66 项通过；实际 Swift 多模态编码连续 7 轮验证了滑窗、视觉 token 展开、训练前缀与目标 mask 一致；8 个 episode 展开为 36 轮后 advantage 不变。

真实八卡验证目录：`runs/input_protocol_validation/grpo_v6`，最终退出码 0，`input_acceptance.json` 保存验收结果：

- Alice：8 episodes、31 次决策（2 次发生历史滑窗），8 rank 的采样/训练条件与 loss 核验通过，完成梯度步骤及完整 checkpoint。
- Bob：8 episodes、27 次决策（1 次发生历史滑窗），同样通过 8 rank 核验并保存完整 checkpoint。
- Holdout：4 episodes、12 次决策，实际输入与生成 token 核验通过，无基础设施错误；任务成功率 0/4。Alice 未产生有效新任务，Bob 使用既有 bootstrap 环境任务回退；不加载其历史权重。
- 修复了普通 Bob 首轮只同步文字而未同步实时图片的 Swift 兼容问题。另在引擎边界统一应用 scheduler 的单轮上限并通过实际方法回归；本次 Bob canary 使用修复前的较大 ceiling，但全部实际输出自然结束且最长 660 tokens，均小于正式上限 768。

新正式目录从 base 新建双方 LoRA，不继承 canary 权重。验收仅覆盖 GRPO ManiSkill PickCube/Panda 的输入与训练链路，不代表任务成功率改善，也不宣称其他 benchmark 或 baseline 已通过。
