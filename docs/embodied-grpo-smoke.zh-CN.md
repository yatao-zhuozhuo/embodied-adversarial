# ManiSkill 具身 GRPO Smoke

> 这份文档描述保留的单进程 Transformers/手写 GRPO correctness baseline。新的
> `vLLM + ms-swift GRPO` 路径及实跑结果见
> `docs/embodied-selfplay-vllm-swift-grpo-migration.zh-CN.md`，不要把本页“尚未实现
> vLLM/Alice 对抗奖励”的历史说明套用到新路径。

这条入口用于验证从视觉 rollout 到 LoRA 权重更新的最小训练闭环。它不是最终的
分布式训练器，也不声称一次 smoke 能提高任务成功率。

## 已实现闭环

每个 iteration 执行：

```text
ManiSkill reset(seed) + capture snapshot
  → 从同一 snapshot 采样 G 条图像条件轨迹
  → bounded action code → EnvAction → simulator receipt
  → host-owned success / grasp / dense reward
  → 组内 reward 标准化得到 advantage
  → completion token-only clipped GRPO + reference KL
  → LoRA optimizer step + checkpoint
```

rollout 和训练使用同一个 `Qwen3.5-4B` policy。参考策略通过临时关闭 LoRA
adapter 得到，因此不需要在 GPU 上再放一份 4B 模型。所有候选轨迹都从同一个完整
ManiSkill snapshot 开始，不把 `seed` 当作严格状态等价证明。

## 安装

当前镜像需要：

```bash
python3 -m pip install --index-url https://pypi.org/simple -e '.[grpo,dev]'
```

## 最小运行

```bash
cd /inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA

CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR="$PWD/runs/embodied_bob_grpo_smoke" \
ITERATIONS=1 GROUP_SIZE=2 MAX_STEPS=1 \
bash scripts/run_embodied_grpo_smoke.sh
```

多步 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR="$PWD/runs/embodied_bob_grpo_multistep" \
ITERATIONS=1 GROUP_SIZE=4 MAX_STEPS=4 MAX_NEW_TOKENS=40 \
bash scripts/run_embodied_grpo_smoke.sh
```

扩大实验时建议先逐项增加 `ITERATIONS`、`GROUP_SIZE` 和 `MAX_STEPS`，同时检查
`metrics.jsonl` 中的 `reward_std` 和 `advantage_abs_mean`。如果一个 group 的 reward
完全相同，GRPO 的策略梯度为零，这是正常现象，不应通过伪造随机 reward 规避。

## 产物

```text
OUTPUT_DIR/
├── config.json
├── metrics.jsonl
├── rollouts.jsonl
├── snapshots/*.pt
├── checkpoint-0001/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── final/
```

`rollouts.jsonl` 保存模型 completion、执行动作和可信环境回执，但不重复保存每帧像素；
初始物理状态单独以 content-addressed snapshot 保存。

可从 snapshot 重放动作并导出每条轨迹的 GIF 和一张总览图：

```bash
python3 scripts/render_embodied_grpo_rollouts.py \
  runs/embodied_bob_grpo_multistep/rollouts.jsonl
```

输出位于 `runs/embodied_bob_grpo_multistep/visualizations/`，包括每条轨迹的第三人称
ManiSkill 仿真 MP4、GIF 和一张总览图。画面上方标注当前动作、环境 reward 和
success；默认选择 `rollouts.jsonl` 的最后一个 iteration。传入 `--view agent` 可改为
导出模型实际看到的 RGB observation。

## 当前实跑结果

在 2×48GB RTX 4090 节点上，`Qwen3.5-4B` 单卡多步 smoke 得到：

```text
group_size:             4
最多动作步数:           4
实际训练 completion:   14
reward_std:             0.01766
advantage_abs_mean:     0.70370
loss:                  -0.20106
grad_norm:              0.84642
峰值显存:               9.57 GiB
训练 iteration 耗时:   55.6 秒（不含模型首次加载）
```

保存后的 `lora_B` 共 3,670,016 个参数，全部已从零初始化更新为非零值，证明反向传播、
optimizer step 和 checkpoint 写入均实际发生。

## Alice 与 Bob 的边界

`--role bob` 使用任务完成 reward。`--role alice` 当前只提供 feasibility 预训练：要求
Alice 真实完成目标，并对 grasp/progress 做轻量 shaping。真正的 asymmetric Alice reward
不能只由 Alice 自己的单条轨迹决定；它必须使用 `adversarial-mle-agents` 中聚合后的
Bob success rate、validity gate、novelty 和 repetition penalty。把 Alice 的完整 reward
接入之前，不应把 `--role alice` 的 smoke 称为对抗式 self-play 训练。

当前训练入口也尚未实现：

- Alice 自动生成 predicate 和 instruction；
- 每个 Alice task 的多 snapshot / 多 seed Bob 评测；
- Alice/Bob checkpoint 交替冻结的 round coordinator；
- vLLM 或多 GPU rollout 加速；
- 固定 holdout seeds 上的训练前后能力比较。

相关实现入口：

- `agent/training/embodied_grpo.py`：action parser、trusted reward、advantage 和 loss；
- `scripts/train_embodied_grpo.py`：真实环境 rollout 与单进程 LoRA trainer；
- `scripts/run_embodied_grpo_smoke.sh`：可覆盖参数的一键 smoke；
- `tests/test_embodied_grpo.py`：reward 和 GRPO 数学单测。
