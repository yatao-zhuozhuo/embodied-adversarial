# Embodied Self-Play 改造设计：仅向模型开放 `observe_env` 与 `step_env`

> 状态：设计稿，尚未实施。更新时间：2026-10-03。
> 目标入口：`scripts/run_embodied_swift_staged_formal5.sh` 所走的正式 staged-colocate 流程；`scripts/run_embodied_selfplay.py` 是单进程原型，需要共享协议但不作为唯一改造对象。
> 范围：第一阶段只支持 ManiSkill `PickCube-v1`、Panda、`pd_ee_delta_pose`。不在本改造中扩大任务集。

## 1. 目标与边界

模型在每轮交互只能选择以下两个**真正由 OpenETA 模拟器 MCP 执行**的工具：

| 模型可调用工具 | 参数 | 语义 |
|---|---|---|
| `observe_env` | 无模型可控参数 | 获取当前 RGB 图像和机器人自身状态；不推进物理时间。 |
| `step_env` | `action: float[7]`，`num_steps: int`，范围 1–10 | 将同一原生动作向量重复执行指定环境控制步数；环境提前终止时立即停止。 |

模型还可输出一种**非工具的终止响应** `finish`，用于主动结束 Alice proposal 或 Bob 尝试。`finish` 不对应 MCP 工具，不推进模拟器，也不意味着任务成功；成功只能由宿主的可信状态检查得出。首轮观察由宿主自动取得，模型可在中途显式调用 `observe_env`。模型调用时不接触 `handle`、`session_id` 等路由标识，由宿主填入。

宿主继续拥有环境创建、重置、关闭、快照保存/恢复、Alice goal 隐藏、Bob goal 设置、任务编译、reward、轨迹落盘和训练调度。这些操作**不可加入模型的 available_tools**，也不可通过 `python_exec` 或通用 OpenETA planner 间接调用。模型不能调用 `ik_preview_check`、`move_to`、`follow_eef_trajectory`、`gripper_control` 或离散动作码。这里使用的是 simulator MCP 的原始 `observe_env`/`step_env`，不是现有 OpenETA AgentTool 的 `observe`（映射到 `render_env`）/`move_to` 门面。

本设计训练的是“视觉和本体感知 → 连续低层动作 → 工具反馈”的策略，**并不等同于现有 Qwen SFT 的高层工具调用格式**。若以后要混训，必须显式区分任务/工具协议，不能把两种轨迹直接拼成同一格式。

## 2. 已有实现与差距

- MCP 服务器已提供 `observe_env(handle, session_id)` 与 `step_env(handle, action, num_steps, session_id)`，见 `sim/mcp_server/server.py`。`step_env` 返回最终 observation、reward、terminated、truncated、info；`observe_env` 返回含相机、机器人、objects、metadata 的完整观测。
- 当前正式 self-play 在 `plugins/embodied_swift_grpo.py` 使用 16 个动作码的正则约束、`parse_action`、`EnvAction(code=...)`，并通过 `ManiSkillSimulatorAdapter`/隔离进程适配器执行。`agent/runtime/embodied_selfplay_prompt.py` 还把 cube/goal 精确坐标、相对误差及“先对齐 x/y/z”等手写策略给模型。
- `scripts/run_embodied_selfplay.py` 与 `scripts/train_embodied_grpo.py` 的原型也共享动作码协议；只改原型不会改变正式 staged 实验。
- MCP 的公开工具目前没有与正式 self-play 的 `capture_snapshot`、`restore_snapshot`、`set_goal_position` 等价的宿主管理接口。不能将“相同 seed 的独立 reset”假装为“相同完整物理 snapshot”：Alice/Bob 和 GRPO 同组比较需要**确实恢复相同初始状态**。
- 现有 `step_env` 并未强制 `num_steps <= 10`；worker 对错误维度动作会截断/补零，并可能裁剪越界数值；`action=None` 还会采样随机动作。新入口必须在发 MCP 请求之前严格拒绝这些输入，而不是依赖底层宽松行为。
- 此处的 `num_steps` 在 worker 中是重复调用 `env.step` 的次数，即**环境控制步**；ManiSkill 的一个控制步内部可能包含多个物理仿真子步。不能把 `num_steps=10` 写成“仅 10 个物理仿真子步”。

## 3. 模型可见的观测和请求

模型可见输入仅包含角色/任务说明、轮次与剩余预算、当前 RGB 图像、机器人本体状态（关节位置与速度、TCP 位姿、夹爪状态）、上一次工具调用的执行状态。初始 `observe_env` 的完整返回先由宿主过滤，再送给模型：不展示 `objects` 中的精确方块/目标位置、depth、success/reward、`metadata` 中的可信检查字段、内部 handle/路径或 Alice 的轨迹。Bob 的目标通过可见 marker 和任务文字表达；Alice 看不到原 PickCube 目标 marker。可信几何信息仍保留给宿主编译任务和计算奖励。视觉可用性是验收条件：若现有 MCP 的主 RGB 帧或目标 marker 不可见，不能靠重新暴露真实 object 坐标来让模型通过。

建议模型输出采用一个紧凑、可严格解析的 JSON 命令（不是动作码）：

```json
{"tool":"step_env","action":[0.2,-0.1,0,0,0,0,1],"num_steps":3}
```

```json
{"tool":"observe_env"}
```

```json
{"response":"finish"}
```

这是 self-play 的模型输出协议；宿主将合法命令翻译为实际 MCP 调用。`step_env.action` 长度必须等于本环境创建时声明的 `action_dim`（本阶段预期为 7），所有元素必须是有限数，且在服务端真实 `action_space` 声明的上下界内；`num_steps` 必须是整数且在 `[1, 10]`。禁止额外字段、布尔值冒充数字、缺省 action、NaN/Infinity、自动裁剪或自动补零。环境或控制模式与预期不一致时 fail closed。向模型解释向量槽位与归一化控制语义，但不要在宿主 prompt 中提供按坐标误差选动作的阶段策略。

返回给模型的 `step_env` 反馈只包含是否执行、实际执行控制步数、是否环境终止、过滤后的最新 RGB/机器人状态及错误信息；原始 reward、success 和对象真值只进可信训练记录。现有 MCP 未显式回传实际执行控制步数，而且只保留最后一个 `env.step` 的 reward；需要扩展 worker receipt，训练记录注明 `reward_scope=last_control_step`，不可把它声称为 1–10 步的累积奖励。需要逐控制步 reward 时再扩展宿主日志/worker receipt。图像以真实帧作为下一轮多模态输入，不把 base64 字符串放进文本 prompt。

## 4. 一次 rollout 的执行契约

```text
宿主恢复同一初始 snapshot、设置角色目标
  → 宿主取得首帧并过滤可见字段
  → 模型输出 observe_env / step_env / finish
  → 宿主严格验证；合法工具调用才经 MCP 执行
  → MCP 回执 + 新帧 → 下一模型轮次
  → finish、可信成功、环境终止、轮次/控制步预算耗尽之一结束
  → 宿主用可信状态编译/评分，记录完整原始证据
```

Alice/Bob 的初始 snapshot、Alice 产生的新目标、Bob 的 goal predicate、冻结 Bob 的评测次数及 staged Phase A/B/C 顺序保持不变。`observe_env` 不消耗环境控制步，但消耗模型 turn 和调用预算；应对重复无进展观察设置上限。`step_env(num_steps=n)` 消耗一个模型 turn、至多 n 个环境控制步；因此同时设置模型 turn 上限和累计控制步上限，不能把旧的“40 个动作”直接解释为“40 个控制步”。Alice replay 必须逐条重放**原始向量及 num_steps**，并比较终态（现有 cube 位置容差可沿用）；只重放模型文字或动作码不够。

错误处理：格式/维度/范围不合法时绝不执行物理动作，给模型结构化错误并计入格式惩罚；连续无效输出达到阈值则结束为失败。MCP 超时或连接断开且无法确认 `step_env` 是否执行时，标记 `outcome_unknown` 并中止该轨迹，不得盲目重试同一动作。`finish` 之后仍运行可信 Alice 有效性门槛或 Bob 成功判定；Bob 的成功可在 `finish` 前由宿主检测到。

## 5. 训练与奖励改造

现有 `ACTION_REGEX`、12-token 非 thinking 输出上限和单动作解析需要换成结构化命令约束与足够的单轮 token 预算。ms-swift 多轮 scheduler 仍保留每轮 assistant completion 的 token IDs、loss mask 与 rollout logprobs；用户图像/观测、宿主工具回执都不能混入 assistant 的训练 mask。对合法 `observe_env`、`step_env` 和 `finish` 的模型生成 token 计算 GRPO；对纯宿主自动操作不计算策略梯度。

Alice 的 proposal 有效性、replay、novelty 与冻结 Bob boundary reward 不变；Bob 的 goal success 仍由可信 `PositionGoalChecker` 判定。可以保留现有格式惩罚，但须按新命令有效率计算。若旧 DAgger/oracle teacher 仍要使用，必须改成同协议的连续向量调用；不能把动作码 teacher 当作新模型的合法 SFT 标签。原有 Alice/Bob adapter/checkpoint **不能被视为新协议已训练模型**；初始化迁移应单独做兼容性实验，并在 artifact 里标注协议版本。

建议新轨迹 schema 至少保存：协议版本、role、snapshot hash、goal predicate（Bob）、每轮模型可见输入摘要与图像引用、原始生成文本、解析后的 tool/参数、验证结果、MCP request/receipt、实际控制步数、终止原因、可信终态/replay 结果、reward、policy version。公开给模型的观察和仅供宿主评分的特权字段分开存放，以便审计是否泄漏答案。原数据、旧 checkpoint 和现有实验配置保持不动，使用新 run name 与协议版本。

## 6. 前置实现选择：真正的 MCP 与可重放快照

要满足“模型调用的是真实 MCP 工具”，正式流程应新增一个**宿主专用**的模拟器管理通道：让创建的 MCP 环境能够保存/恢复完整 ManiSkill state、设置 Alice 隐藏目标/Bob 目标，并附带 state hash。它可作为内部 RPC 或只向受信宿主暴露的管理 API，但不能出现在模型工具目录中。服务端/worker 必须串行化同一 handle 的 restore 与 step，确保恢复后的首帧与 physics state 一致。随后再接 self-play scheduler 的两工具 dispatcher。

使用本地适配器实现同名函数可以做早期协议 smoke test，但它不是“经 MCP 调用”，**不能作为最终验收**。如果暂时没有可验证的 MCP snapshot restore，先停在 smoke test，不启动正式 GRPO：否则同组 rollout 与 Bob 评测可能不在相同初始状态上，奖励语义失效。

## 7. 分阶段实施与验收

1. **工具及观测契约**：增加严格请求校验和可见字段投影；确认 `observe_env` 不 step、`step_env` 恰好执行 1–10 步且提前终止；不向模型展示其它工具或特权状态。
2. **宿主快照能力**：完成 MCP 管理通道；同一 snapshot 恢复多次的 hash/初始观测一致，向量轨迹 replay 终态一致；Alice/Bob 共享 snapshot hash。
3. **单轨迹闭环**：先不训练，跑 Alice、Bob 各一个人工固定命令序列及模型序列；检查 RGB 更新、7 维动作、夹爪方向、终止与错误路径，完整保存可视化帧和 MCP receipts。
4. **正式 scheduler 切换**：优先改 `plugins/embodied_swift_grpo.py`、新 prompt/解析器和 launcher 的独立配置；保持 staged Phase A/B/C、Bob 冻结和可信 reward。之后让原型脚本复用同一协议，避免两套含义不同的 `step_env`。
5. **训练 canary**：少量同 snapshot GRPO group，验证每个合法模型轮次都有准确 token/logprob/mask、无效动作不改物理状态、reward 方差/梯度正常、显存与总 token/控制步预算可控。再评估多轮任务成功率，不能只看模型能否输出 JSON。

最小通过标准：模型侧工具白名单恰为两个；公开观测没有对象真值泄漏；错误动作零副作用；控制步数严格在预算内；Alice replay 与 Bob 起点 hash 校验通过；官方成功与 `finish` 不混淆；训练样本可复现、可审计。

## 8. 主要代码落点

| 位置 | 预期改动 |
|---|---|
| `sim/mcp_server/server.py`、`sim/mcp_server/worker_mgr.py`、`sim/bench_worker.py` | 加强 `step_env` 参数边界；增加宿主专用 snapshot/goal 管理与准确执行回执。 |
| `agent/runtime/embodied_selfplay_prompt.py` | 两工具协议、视觉/本体状态投影；去掉动作码、真值坐标及手写控制阶段。 |
| `plugins/embodied_swift_grpo.py` | 正式多轮 rollout 的解析、MCP dispatch、向量 replay、终止与 artifact；保留 staged reward/训练链路。 |
| `agent/training/embodied_grpo.py`、`scripts/run_embodied_selfplay.py`、`scripts/train_embodied_grpo.py` | 原型侧共享新协议与训练记录；旧动作码实现保留为 legacy，避免破坏既有 run。 |
| `scripts/run_embodied_swift_staged_formal5.sh` 及新配置/测试 | 通过显式协议开关启用新路径，不静默改变旧实验的恢复行为。 |

以上是修改设计，不代表当前代码已经具备上述能力或已有数据可直接用于该协议训练。
