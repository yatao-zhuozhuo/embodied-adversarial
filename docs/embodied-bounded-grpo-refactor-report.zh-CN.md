# BoundedEmbodiedGRPOTrainer 实施与验证报告

日期：2026-10-02

分支：`feat/embodied-asymmetric-self-play`
基线 revision：`541d3d4f3195f674e642d2eec8ddb4ae85c19d3c`

## 已实施

- 新增公共 `BoundedEmbodiedGRPOTrainer`：Bob 直接使用公共类；Alice staged Trainer 继承公共类，并保留 proposal、冻结 Bob evaluation、receipt 与 reward finalize。
- Alice、冻结 Bob、独立 Bob 的 engine 调用都经过 `_run_bounded_scheduler_phase()`；每个 request 按剩余累计预算分组下发独立 `RequestConfig`，再恢复原顺序。
- scheduler 记录每 turn token 数、累计 token、实际 allowance 和 `token_budget_exhausted`；最终 token IDs、loss mask、rollout logprobs 通过幂等 helper 对齐并在送入 GRPO 前再次校验。
- 公共 engine session 负责 wake、权重同步、KV wake、prefix cache reset、sleep(level=2) 和异常清理；阶段 JSONL 记录时间、allocated/reserved/peak 和清理结果。
- Alice/Bob 统一从 `run_embodied_bounded_colocate.py` 启动；旧 Alice Python 入口为兼容 wrapper。第一阶段只允许 `OPENETA_HISTORY_MODE=full`。
- 正式 shell 默认使用共享目录旁的 `.venv_selfplay_embodied`，禁用会访问公网的 ModelScope 本地模型更新检查，并保留 TensorBoard/W&B 的 Swift 原生 `report_to` 接口。

## 实际环境核对

指定机器：`ws-3e1a245569b1874e`（主机名 `model-training`）

| 依赖 | 实际版本 |
| --- | --- |
| ms-swift | 4.4.2 |
| vLLM | 0.19.1 |
| PyTorch | 2.10.0+cu128 |
| GPU | 8 × NVIDIA H200，143771 MiB/卡 |

已读取并核对实际 `GRPOTrainer._fast_infer()`、`samples2requests()`、`_postprocess_rollout_outputs()` 和 `run_multi_turn()` 签名。未修改 site-packages。

## 回归结果

在指定机器执行：

```text
20 passed, 5 warnings in 7.60s
```

覆盖纯预算边界、剩余 20/0 token、超额拒绝、token/mask/logprob 对齐、最终 turn 幂等、不同 request allowance 分组及顺序恢复、异常时 engine sleep、staged reward/contracts 和 ManiSkill backend 参数。

单卡真实 canary 已完成训练模型与 vLLM 初始化并进入 `Train: 0/1`。它确认了两个额外事实：

1. 原启动会在 `check_local_model_is_latest()` 阻塞公网 ModelScope 请求；现已用 `--check_model false` 消除。
2. rollout 启动时该工作容器缺失可用的 NVIDIA Vulkan ICD/匹配的 `libnvidia-gpucomp.so.580.159.04`，ManiSkill 报 `vk::createInstanceUnique: ErrorIncompatibleDriver`。公共 finally 路径仍成功执行 vLLM sleep，记录为 `vllm_is_sleeping_after=true`。

失败证据位于 `runs/bounded_refactor_canary_single/bob-canary.log` 和 `phase_timing.rank-00.jsonl`。该错误在独立 `probe_maniskill_isolated.py` 中同样复现，发生在任何模型动作生成之前，因此不是 token-budget 或 Trainer 路由错误。

## 尚未通过的 GPU 验收

由于指定机器当前 Vulkan driver 注入不完整，尚未完成真实 ManiSkill rollout、GRPO backward/optimizer、checkpoint/resume 和 Alice 三阶段 GPU canary。修复/重建带 NVIDIA `graphics,compute,utility` driver capability 的工作容器后，应按计划依次执行：单卡 probe、8 卡 Bob 两个 optimizer steps、8 卡 Alice 一个完整更新、长轨迹压力测试。当前不能把 CPU/fake-engine 回归标记成全部端到端验收通过。
