# Alice / Bob 共用 BoundedEmbodiedGRPOTrainer 的修改计划

日期：2026-10-02。状态：设计方案，尚未实施。本次仅新增本文档。

目标是让 Alice 训练、Alice 阶段的冻结 Bob 评估、Bob 独立训练，共用同一套多轮生成预算、请求/history 管理和 vLLM 生命周期管理；GRPO loss、advantage、KL、反向传播、优化器及 LoRA 更新继续使用 ms-swift。

## 1. 现状、证据与问题边界

当前 `scripts/run_embodied_swift_staged_colocate.sh` 根据 ROLE 选择不同入口：

| 路径 | 入口 | Trainer | rollout 特点 |
| --- | --- | --- | --- |
| Alice 训练 | `scripts/run_embodied_staged_colocate.py` | `StagedEmbodiedGRPOTrainer(GRPOTrainer)` | Alice 出题 → 冻结 Bob 评估 → 补算 Alice reward |
| Alice 内部的 Bob 评估 | 同上 | 由 Alice Trainer 编排，无 Bob 参数更新 | 共用 `_run_scheduler_phase()` |
| Bob 独立训练 | `swift rlhf` | 原生 ms-swift GRPOTrainer | Bob rollout → Bob reward → 训练 |

主要代码位置：

- `agent/training/embodied_staged_colocate.py`：Trainer 注册、`multi_turn_completion_length_context()`、`_run_scheduler_phase()`、`_fast_infer()`、adapter 切换、proposal/receipt 持久化。
- `plugins/embodied_swift_grpo.py`：共享 `_EmbodiedScheduler`、`prepare_request_config()`、`run()`、环境步进、请求图像、动作解析、原始轨迹、reward。
- `scripts/run_embodied_swift_formal10.sh`：分别设置 Alice/Bob 的数据集、checkpoint、环境变量并启动训练。

round-0000 的历史证据：

| 指标 | Alice | Bob 独立训练 |
| --- | --- | --- |
| 已落盘轨迹数 | 96 | 40（含失败训练步生成的样本） |
| 原始轨迹平均 turns | 29.72 | 26.35 |
| 原始轨迹单 turn 最大输出字符数 | 3,970 | 67,306 |
| Swift completion mean_length | 各训练步约 10k | 首个完成训练步约 26.7k |
| Swift `memory(GiB)` 记录 | 最高 61.46 | 首步 126.18，之后 backward OOM |

证据文件：

- `runs/staged_colocate_5round/round-0000/alice_train/v2-20260930-100650/logging.jsonl`
- `runs/staged_colocate_5round/round-0000/bob_train/v0-20260930-190141/logging.jsonl`
- `runs/staged_colocate_5round/logs/round-0000-bob-trainer.log`：存在 `max_tokens(24576)` 被剩余上下文裁到约 24k 的告警，以及 backward 申请 33.02 GiB 失败的记录。

这些证据说明 Bob 的环境 turn 数没有增长，单 turn 文本存在明显长尾，实际生成参数没有落实预期的 1024-token 上限。Alice 的自定义路径显式调用 `scheduler.prepare_request_config()`，Bob 的历史运行没有获得相同行为。

需要在实施时继续核实的事项：

1. 日志 `completion_length` 的统计是否包含反馈、是否因多轮拼接重复计数。不能直接把日志中的 46.7k 当成单次 vLLM 请求长度；需要原始生成 token IDs、实际模板编码长度和 loss mask 三者对账。
2. `sleep_level=2` 是配置，不是权重和 KV cache 已释放的运行证据。历史日志缺少可对齐的 sleep 前后进程显存采样，不能宣称已经排除权重残留。
3. 33.02 GiB 的申请大小与大 logits 张量相容，但只有张量形状/allocator trace 才能确定具体分配来源。
4. 前面对 256 分辨率额外显存及修复后显存的区间属于粗估，不作为验收承诺。此次方案不以“Bob 必须降到 Alice 同样显存”为标准。
5. 现有 worktree 已有未提交修改，实施必须基于当前工作树审阅增量，不能覆盖这些变更。

## 2. 目标结构及职责

```text
ms-swift GRPOTrainer
└── BoundedEmbodiedGRPOTrainer
    ├── 公共请求/预算/图像-history/资源生命周期/观测记录
    ├── Bob：默认 _collect_role_rollouts() → Bob reward
    └── StagedEmbodiedGRPOTrainer（保留现有类名）
        └── Alice：Alice rollout → frozen Bob evaluation → Alice reward
```

无需新增只有空实现的 Bob 子类。基础类本身承担 Bob 的单策略流程；Alice 子类只覆盖角色编排。注册时明确检查角色，避免将 Alice 请求误送入单策略基础实现。

公共层拥有：

- 每个请求的 turn budget、累计生成预算和实际上下文预算。
- 初始化后的请求传递、图像与消息对应关系、输出 token/logprob/loss mask 校验。
- 每次 engine 调用的最终采样上限，首 turn 和后续 turn 使用同一路径。
- 一次完整 rollout 编排的 wake → 同步当前权重 → 采样 → sleep；异常路径也收尾。
- 公共阶段耗时、显存和预算统计；保留 scheduler 的原始轨迹写入职责。

Alice 子类拥有：proposal ID、任务编译结果收集、冻结 Bob adapter/基础模型选择、Bob 请求分片与重试、receipt 校验、最终 reward 和 proposal/manifest 持久化。

Scheduler 继续拥有：环境初始化/恢复、真实动作执行、终止判定、prompt 构造、观测采集和原始 artifact。公共 Trainer 协调这些能力，不再复制一套仿真逻辑。

## 3. 文件修改清单

| 文件 | 操作 | 具体内容 |
| --- | --- | --- |
| `agent/training/embodied_bounded_grpo.py` | 新增 | 公共 Trainer、角色注册入口、统一 engine 调用和生命周期 |
| `agent/training/embodied_rollout_budget.py` | 新增 | 纯 Python 请求预算状态与允许生成长度计算，便于 CPU 测试 |
| `agent/training/embodied_staged_colocate.py` | 修改 | 改为继承公共 Trainer；保留 Alice 三阶段业务逻辑；移除重复公共方法 |
| `plugins/embodied_swift_grpo.py` | 修改 | 幂等输出对齐 helper；预算终止/清理 hook；token 和 history 记录；保持 reward 公式 |
| `scripts/run_embodied_bounded_colocate.py` | 新增 | 公共 Python 启动入口，按显式角色注册对应 Trainer |
| `scripts/run_embodied_staged_colocate.py` | 修改 | 保留旧入口，作为 Alice 兼容 wrapper，委托公共入口 |
| `scripts/run_embodied_swift_staged_colocate.sh` | 修改 | Alice/Bob 都从公共 Python 入口启动；按角色选 scheduler/reward |
| `scripts/run_embodied_swift_staged_formal5.sh` | 修改 | 暴露/打印公共预算和 history 参数，默认值明确 |
| `scripts/run_embodied_swift_formal10.sh` | 修改 | 两角色一致传递新参数；保留数据集和 checkpoint 编排 |
| `configs/embodied-swift-staged.example.env` | 修改 | 增加公共配置和预算含义说明 |
| `scripts/summarize_embodied_formal_round.py` | 修改 | 兼容读取新预算、显存、耗时摘要；旧 artifact 缺失字段允许为空 |
| `scripts/probe_embodied_bounded_grpo.py` | 新增 | 隔离 run 下的短程 GPU canary，输出对比报告 |
| `tests/test_embodied_rollout_budget.py` | 新增 | 纯预算规则及终止语义测试 |
| `tests/test_embodied_bounded_grpo.py` | 新增 | fake engine / scheduler 验证角色路由、请求传递、资源收尾和 token 对齐 |
| `tests/test_embodied_staged_contracts.py` | 扩展 | Alice receipts/reward、冻结 Bob 预算终止语义回归 |
| `docs/embodied-swift-staged-portability.zh-CN.md` | 修改 | 更新调用链、配置和恢复运行说明 |

实施第一阶段保持模型、64×64 分辨率、reward 公式、32 turns、LoRA 设置、Bob 训练任务选取规则不变。256 分辨率、任务采样改进、loss/logits 优化分别做后续实验，避免混淆此次改造收益。

## 4. 公共 Trainer 具体如何写

### 4.1 先确认运行环境接口

在实际训练机器读取已安装 ms-swift/vLLM/Transformers 版本和源码，记录版本及相关文件校验值。文档中的固定版本不能代替运行环境检查。

重点核对 `GRPOTrainer._fast_infer()`、`samples2requests()`、`_postprocess_rollout_outputs()`、`run_multi_turn()`、engine `set_default_max_tokens()`、sleep/wake 和生成 logprobs 的调用协议。确认普通 Bob 路径何处绕过 scheduler `run()`；只在仓库中实现兼容层，不修改 site-packages。

### 4.2 方法拆分草图

下面是职责草图，具体签名在第 4.1 步确认后适配：

```python
class BoundedEmbodiedGRPOTrainer(GRPOTrainer):
    def _prepare_scheduler(self):
        # 公共配置校验、预算/观测状态；子类补充 Alice 专属状态
        ...

    def _fast_infer(self, samples):
        with self._rollout_engine_session():
            return self._collect_role_rollouts(samples)

    def _collect_role_rollouts(self, samples):
        # 基础类用于 Bob 单策略训练
        return self._run_bounded_scheduler_phase(
            samples, self.multi_turn_scheduler, self._current_adapter_request()
        )

    def _run_bounded_scheduler_phase(self, samples, scheduler, adapter_request):
        # 初始化一次；传递同一批初始化请求；每次调用前计算预算
        # 多轮调度后统一校验 token / mask / logprob / messages
        ...

class StagedEmbodiedGRPOTrainer(BoundedEmbodiedGRPOTrainer):
    def _collect_role_rollouts(self, samples):
        # Alice → proposal → frozen Bob → finalize reward
        # 两个角色调用同一 _run_bounded_scheduler_phase()
        ...
```

将已有 `_engine_infer_with_adapter()`、`_rollout_with_adapter()`、请求初始化、通用计时迁到公共层；Alice `_fast_infer()` 改成 `_collect_role_rollouts()`，不再自行 sleep/wake。保留 GRPO 的 `super().compute_loss()`、`super().training_step()` 和 optimizer 行为，避免创建第二套训练算法。

当前的 `multi_turn_completion_length_context()` 不能原样作为唯一预算机制搬过去：它使用 phase 级 engine 窗口，且预算耗尽时会增加 crash-guard 空间。改造后以请求级账本为准；与 Swift context 的交互只保留兼容适配，不允许两个机制相互覆盖上限。

### 4.3 三个长度上限分别定义

| 限制 | 默认 | 计数内容 |
| --- | --- | --- |
| 每 turn 生成上限 | `OPENETA_THINKING_MAX_TOKENS=1024` | 本轮全部生成 token，包含 thinking 和最终动作 |
| 每条轨迹累计生成上限 | `MAX_COMPLETION_LENGTH=24576` | 所有 turn 实际生成 token IDs 之和；不重复累计历史 |
| 单次请求/训练样本上下文上限 | `min(MAX_LENGTH, VLLM_MAX_MODEL_LEN)=32768` | 模板编码后的文字、图像 token、特殊 token 与当前生成空间 |

为每个 request_id 保存 `generated_tokens_total`、`turn_index`、budget 终止原因。每次 engine 调用前，在最终多模态模板编码后计算：

```python
allowed = min(
    per_turn_limit,
    trajectory_limit - generated_tokens_total,
    context_limit - encoded_prompt_tokens,
)
```

- 首轮和后续每轮都执行；每个请求独立计算，使用 request config 副本。
- 若底层 batch 只接受一个 RequestConfig，按 `allowed` 分组或以逐请求配置接口下发，保持输出原顺序；不能让第一个请求的预算代表整批。
- engine 后续的默认长度修正只能进一步缩小 allowed，不能将其提升回 24576。
- `allowed <= 0` 时通过 scheduler 终止钩子收尾，记录 `token_budget_exhausted` 或 `context_budget_exhausted`，不再调用模型。
- 不伪造 `DONE` token、动作或 logprob。累计预算耗尽与真实 goal success 分别记录；Alice 即使耗尽预算，仍按已有物理轨迹编译验证任务。
- Bob 预算耗尽属于策略在给定预算内未完成任务，不是 infrastructure error；环境故障才走 evaluator error 语义。
- 未闭合 thinking 的既有安全动作解析保留，并记录发生次数；不能无限续加 crash-guard token。
- 初始化就超出上下文、没有任何 assistant 输出的样本，明确报配置错误，不能送入 GRPO 后静默丢弃，改变 group 大小。
- 请求级停止必须配合多 rank 的全局完成判定。某 rank 没有活跃请求时仍进入必要的 collective，不能独自退出导致死锁。

### 4.4 图像、history 与训练一致性

第一阶段统一默认 `OPENETA_HISTORY_MODE=full`，保留当前的完整多轮图片和文本历史。

具体约束：

1. scheduler `on_trajectory_start()` 创建了第一张图、prompt 和模板参数后，向 engine 传同一请求；不能从旧 sample 重建并丢失初始观测。
2. 每个新 user 观测只追加一次，消息中的 image placeholder 与实际图片列表一一对应。
3. 每轮记录源图片尺寸、hash、`image_grid_thw` 或实际视觉 token 数；区分源图 64×64 与 processor 重采样后的尺寸。
4. 原始每 turn 生成 token IDs 只累计一次。最终 turn、特殊 token、response loss mask 和 rollout logprobs 必须对齐，拒绝重复拼接。
5. 将现有 scheduler `run()` 内的最终 turn 修复/校验抽成幂等 helper，让 server 路径和公共 colocate 路径都可调用。避免重复补最后一轮，也避免用事后重新 tokenize 猜测原始采样 token。
6. 没有可靠 rollout logprobs 时显式记录兼容策略；不能在未报告的情况下悄悄清空并声称 importance correction 正常。
7. 送入训练的 messages、images、token IDs 必须复现采样时条件；训练编码后再检查上下文长度。

“只保留最近 1–4 张图”不在第一次重构中启用。删掉 rollout 里的旧图却仍按完整历史重算 GRPO logprob，会改变策略条件分布。若后续加入窗口 history，需要单独设计每 turn 条件重放及 loss/importance 对齐，再做实验。

### 4.5 vLLM sleep/wake 与异常收尾

公共 `_rollout_engine_session()` 包含完整资源生命周期：

1. 记录阶段开始的每 rank 显存。
2. 按安装版本支持的接口唤醒权重，按既有机制同步当前训练策略；level 2 wake 后不能错误跳过同步。
3. 唤醒 KV cache，执行所有该阶段 rollout。Alice 的两个角色在同一个 session 内切 adapter，不反复重建基础模型。
4. `finally` 中关闭残留环境会话、清理 prefix cache、`sleep(level=2)`，并恢复 allocator/context monkeypatch 状态。
5. 回到训练阶段前记录 `is_sleeping`、allocated/reserved 和进程显存。异常收尾本身失败时保留原始异常并记录清理失败，不吞掉错误。

不能把 `torch.cuda.empty_cache()` 当作活跃权重释放，也不能用训练模型的 `offload_model` 与 vLLM sleep 混为一谈。第一阶段保留原有训练模型/optimizer offload 设置，靠实测检查 vLLM 部分。

DDP 致命异常需要统一失败并交给 launcher 终止其他 rank，避免仅本 rank 执行阻塞 collective。无本地 Bob 评估任务的 rank，也要验证 Alice phase 的集体操作顺序一致。

### 4.6 Adapter 与角色隔离

- 当前被训练策略使用 adapter ID 1；Alice 额外使用 ID 2 表示冻结 Bob。Bob 单策略不初始化冻结 Bob 路径或要求其权重存在。
- 原模型冷启动仍使用 `__base_model__` 语义：冻结 Bob 评估可以不带 adapter；当前 Alice/Bob 的可训练 LoRA 仍按 Swift 流程初始化。
- 保留已有 Alice adapter 名称及 checkpoint 格式；Bob 使用明确的 current-Bob 名称，不能在日志里误标成 current-Alice。
- 当前模块存在 Trainer→plugin 和 plugin→Swift 的导入链；公共模块采用局部导入/明确注册顺序，不让 plugin 反向导入 Trainer 形成环。

## 5. 两条业务路径保留什么

Alice 子类保留 `_extract_local_proposals()`、`_bob_sample()`、receipt 构造、`shard_bob_requests()`、冻结 adapter 选择、`finalize_alice_reward()`、proposal 和 manifest。

Alice 的编译门槛、boundary reward、重复惩罚、格式惩罚、两次 Bob 评估及 evaluator error 处理均保持现有公式。补算 reward 时继续保留 Swift 标量 `num_turns` 元数据。相同 proposal/receipts fixture 下，改造前后 reward 必须一致。

Bob 只用当前策略完成一次多轮 rollout，由共享 scheduler 的 `_bob_reward()` 给分，直接返回给原生 GRPO。Bob 不产生 Alice proposal，不二次评估自己。独立训练与冻结评估共用预算实现，但数据、adapter 是否更新仍不同。

## 6. 启动和配置如何接线

新增入口 `scripts/run_embodied_bounded_colocate.py`：

1. 在导入 Torch/vLLM/ManiSkill 前执行现有单设备映射。
2. 读取并校验 `OPENETA_TRAIN_ROLE=alice|bob`。
3. 导入 plugin，注册 scheduler/reward。
4. 按角色向 TrainerFactory 注册 Alice 子类或公共 Bob 基础类。
5. 调用 `swift.pipelines.rlhf_main()`。

共享 shell 的 Alice/Bob 都设为 `python -m torch.distributed.run ... run_embodied_bounded_colocate.py`；明确 export 角色，不能只依赖未导出的 shell 局部变量 ROLE。

保留 `OPENETA_STAGED_COLOCATE=true` 对 Alice 的现有 proposal 延迟定分语义；Bob 为 false，但两者都使用公共 bounded 入口。这个变量不再承担“是否具有 token 限制”的含义。

启动日志必须输出：实际 Trainer 类、训练角色、每-turn 上限、累计预算、上下文上限、history 模式、源图分辨率、processor 像素配置、sleep_level、KV cache 配额、依赖版本和代码 revision。

旧 `scripts/run_embodied_staged_colocate.py` 保留 Alice wrapper，使已有外部命令兼容。此次统一范围是 TP=1 同步 colocate；server/异步/TP>1 的旧入口继续独立维护，不能默认声称已覆盖。

## 7. 记录哪些信息才能验证效果

公共 timing JSONL 按 role/phase/rank 区分，避免 Alice、冻结 Bob、Bob 独立训练的同名记录冲突。episode artifact 保持原有目录和字段，仅增补版本化元数据。

| 层级 | 新增记录 |
| --- | --- |
| 每 turn | request_id、role、adapter、turn、实际输入 token、视觉 token、allowed max_tokens、真实生成 token、累计生成 token、finish_reason、动作有效性 |
| 每 trajectory | num_turns、累计生成 token、训练编码长度、mask 有效 token 数、logprob 数、预算终止原因 |
| 每阶段/每 rank | wake、sync、rollout、sleep、forward/backward、optimizer 耗时；allocated/reserved/peak；可用时的 NVML 进程显存 |
| 每训练 step | reward mean/std、成功率、有效 proposal 比例、invalid action 比例、thinking 截断率、长度 P50/P95/max |

显存分别在 `before_wake`、`after_wake`、`after_rollout`、`after_sleep`、`before_forward`、`after_forward`、`after_backward` 采样。CUDA 计时同步仅在明确阶段边界或 profiling 模式使用，避免逐 token 同步拖慢训练。

PyTorch peak 只代表相应 allocator/进程；CuMem 和仿真 worker 的显存需独立记录。重置 peak 前保存上一阶段统计，不能将 rollout-only 的约 21.6 GiB 与训练累计 `memory(GiB)` 直接当同一口径比较。

后续新实验可以通过已有 `REPORT_TO=tensorboard` 输出两角色同名指标，JSONL 为必备来源。W&B 作为可选后端，不要求账号或启动外部上报；不修改历史 run 的记录。

## 8. 实施顺序和验收

### 阶段 A：接口核对及基线报告

读取实际训练环境的依赖实现，确认 token/length/logprobs 的真实统计口径；整理历史 Alice/Bob 的同口径长度分布。记录当前工作树差异。产出调用链和基线 JSON，不启动正式训练。

### 阶段 B：提取公共层并接入两角色

先写纯预算模块和公共 Trainer，再让现有 Alice 子类继承，最后将 Bob 入口切到公共类。迁移时保留现有 reward、数据格式、基础模型冷启动、checkpoint 续跑和日志兼容性。

### 阶段 C：CPU / fake-engine 回归

必须覆盖能复现当前问题及边界错误的测试：

- 首轮和第 N 轮输入配置即使为 24576，实际 engine 生成上限仍不超过 1024。
- 两个不同初始 prompt 长度/不同剩余预算的请求互不影响，输出顺序保持。
- 剩余预算为 20 时最多生成 20；为 0 时不调用 engine，无负数 max_tokens，无 crash-guard 无限延长。
- 图片 placeholder、图片数量、messages、token IDs、mask、logprobs 对齐，最终 turn 恰好计入一次。
- 用异常注入验证 rollout、reward finalize 和 sleep 各阶段失败时资源清理与异常保留。
- Bob 注册公共类且不要求 frozen Bob adapter；Alice 注册子类，保留 receipts 校验。
- 原模型冷启动和已存在 LoRA 路径均正确路由。
- 现有 `tests/test_embodied_staged_contracts.py` reward/summary 回归通过。

### 阶段 D：GPU 短程验证

使用独立 `RUN_ROOT`，固定模型、snapshot、goal、采样参数、64 分辨率和历史策略，不覆盖原 run。

1. 单 GPU 诊断 probe 验证每-turn 实际上限、预算结束和 engine sleep；诊断入口可单卡，正式 shell 的 8 卡要求保留。
2. 8 卡 Bob 完整 rollout + old/ref forward + loss backward + optimizer + checkpoint/resume；至少完成两个优化步，覆盖此前第二步 OOM 的阶段。
3. 8 卡 Alice 完整三阶段及一次更新；覆盖有效/无效 proposal、零本地 Bob 请求、评估失败重试。
4. 长轨迹 stress case 覆盖接近 32 turns、接近累计 token 预算；不能只用很早成功的短任务证明显存安全。

验证项包括所有 rank 均完成、无 deadlock、无 OOM、预算不超限、sleep 后状态及显存合理回落、checkpoint 可恢复、frozen Bob 权重不更新。若无法取得训练 GPU，CPU 测试只能作为阶段性结果，不把此计划标记为全部验收通过。

### 阶段 E：对照及是否恢复正式运行

用同一批固定任务分别跑“重构前 Alice 路径”与“公共路径”，对比预算参数、raw token 统计、reward 和显存；随机采样不要求逐 token 相同，但确定性 helper 和 reward fixture 必须一致。旧 Bob 无上限行为仅作历史对照，不需要再次运行到 OOM。

新结果写入独立报告，再决定从 Bob `checkpoint-1` 续跑还是从原基座重新跑对照。修复会改变 rollout 分布；继续旧 checkpoint 属于混合训练历史，不能当成从头一致配置的实验。正式 run 续跑不包含在“只写本计划”的当前操作中。

## 9. 预期效果及不确定性

确定可验收的结果是：Alice、冻结 Bob、独立 Bob 每 turn 都执行同一个 1024-token 上限；每条轨迹生成 token 不超过 24576；上下文不超过实际有效限制；预算耗尽有显式终止；同角色资源生命周期和观测口径一致。

预期 Bob 超长 thinking 长尾减少，推理耗时和训练中间张量显存下降。模型权重、LoRA 参数和优化器配置不变，因此常驻部分不因这次重构显著缩小。总显存下降多少必须通过相同任务下的 forward/backward 实测；即使 token 限制修复，也不能保证全量 logits 的反向峰值不再 OOM。

截断率可能上升：历史 Alice 已有许多 thinking 未闭合。若 1024 不足以稳定输出动作，另做更短推理 prompt 或预算调整实验；不要自动提高到 24k 掩盖问题，也不能保证 reward/成功率一定提高。

若显存仍不足，根据记录定位到 logits、激活、KV cache 或活跃权重，再单独选择优化措施。提高到 256 分辨率的显存评估，应在公共路径验证后进行 64/128/256 的固定任务对照，不从历史异常 Bob 峰值线性外推。

## 10. 完成标准

- 两角色使用公共入口；Alice 只额外编排冻结 Bob 评估。
- 每-turn、累计生成、上下文三个限制都有实际 token 证据和边界测试。
- 训练数据与采样条件对齐；最终 turn 和 logprob 无漏记/重复。
- 两角色均有分阶段显存及时间记录，可判断 vLLM 是否释放。
- GRPO 更新及 reward 公式回归通过，8 卡端到端和恢复测试通过。
- 提交文件清单、依赖版本、基线与改造后测量报告；所有旧 artifact/checkpoint 可追溯。
