# ManiSkill SFT 数据采集

目标：GLM-5.3-w8a8c8 与 Qwen3.8-27B 各采集 500 条可信成功轨迹；1000 条全部作为 SFT 训练数据，不划分验证集或测试集。

详细设计见 `ManiSkill_500_Trajectory_SFT_Plan.zh-CN.md`。

## 已生成资产

- `configs/task_catalog.v1.json`：20 个任务及每模型配额。
- `manifests/glm.train.500.v1.json`：GLM 500 条目标 manifest。
- `manifests/qwen.train.500.v1.json`：Qwen 500 条目标 manifest。
- `configs/provider.*.example.env`：不含密钥的 provider 示例。

## 启动前

在启动采集的同一个 shell 中注入密钥，不要把密钥写入仓库：

```bash
read -rsp 'INF_API_KEY: ' INF_API_KEY
export INF_API_KEY
```

确认 GPU 0 的既有任务已释放：

```bash
nvidia-smi
```

## Canary

先顺序运行两个 5-episode PickCube canary：

```bash
./scripts/run_maniskill_sft_canary.sh glm
./scripts/run_maniskill_sft_canary.sh qwen
```

调试时可以先只跑 1 条，并保持 thinking 开启、提高输出预算：

```bash
OPENETA_CANARY_COUNT=1 \
OPENETA_LLM_ENABLE_THINKING=true \
OPENETA_LLM_MAX_TOKENS=16384 \
./scripts/run_maniskill_sft_canary.sh glm
```

正式采集默认关闭 thinking：实测 GLM 开启 thinking 后单轮会超过 180 秒，
且 4096 tokens 曾在输出 `<decision>` 前被截断。关闭 thinking 后仍要求模型
输出完整的 `<reasoning>` 与工具调用 XML，并显著提高 1000 条采集的可行性。

脚本默认拒绝在 GPU 利用率高于 25% 时启动。不要用 `OPENETA_ALLOW_BUSY_GPU=true` 绕过，除非已经确认资源不会互相影响。

## 正式采集

Canary 验收后顺序运行两个可续跑采集器：

```bash
./scripts/run_maniskill_sft_collection.sh glm
./scripts/run_maniskill_sft_collection.sh qwen
```

默认参数：

- 单卡 ManiSkill worker pool：1
- environment concurrency：1
- provider concurrency：1
- wave size：5
- 每模型最大尝试数：2500

稳定后可通过环境变量把 environment/provider concurrency 提升到 2，再观察 GPU/Vulkan、429 和 cleanup 指标。

采集器会持续扫描 raw bundle，只把以下 episode 计入配额并导出：

- rollout bundle 完整且 artifact hash 正确；
- ManiSkill 官方 `success` 为 true；
- 无人工或 guidance 介入；
- teacher model 与 shard 一致；
- 主 planner decision 已通过 host validator。

感知通道按服务能力分开配置：Qwen 端点接收 `image_url`，使用原始
`base_camera` 图像，同时读取 ManiSkill MCP 的结构化对象状态；GLM 端点
明确是非多模态服务，因此不发送图像，只使用同一结构化状态。本数据版本
训练 planner/tool use，不宣称是纯视觉策略数据。两个 teacher 都独立规划
和调用动作工具，不互相评分或提供感知结果。

输出位置：

```text
raw/glm/                         GLM 原始 rollout
raw/qwen/                        Qwen 原始 rollout
accepted/glm/episodes.v1.jsonl   GLM 接收清单
accepted/qwen/episodes.v1.jsonl  Qwen 接收清单
exported/glm/train.v1.jsonl      GLM SFT 数据
exported/qwen/train.v1.jsonl     Qwen SFT 数据
reports/                         每波结果、导出报告和 MCP 日志
```

## 最终审计

```bash
uv run python scripts/audit_maniskill_sft.py \
  --dataset SFT_data/exported/glm/train.v1.jsonl \
  --accepted-index SFT_data/accepted/glm/episodes.v1.jsonl \
  --catalog SFT_data/configs/task_catalog.v1.json \
  --teacher-model GLM-5.3-w8a8c8 \
  --require-complete-quota

uv run python scripts/audit_maniskill_sft.py \
  --dataset SFT_data/exported/qwen/train.v1.jsonl \
  --accepted-index SFT_data/accepted/qwen/episodes.v1.jsonl \
  --catalog SFT_data/configs/task_catalog.v1.json \
  --teacher-model Qwen3.8-27B \
  --require-complete-quota
```
