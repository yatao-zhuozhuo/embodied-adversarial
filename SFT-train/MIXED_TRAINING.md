# 三模型混合数据 SFT

机器：Inspire `embodied-selfplay-copy`，分布式训练空间，八张 H200。
基于各模型原始基座做 LoRA，未加载此前五轨迹训练的 adapter。

| 模型 | GPU | 梯度累积 | 全局 batch |
| --- | --- | --- | --- |
| Qwen3.5-4B | 0,1 | 4 | 8 |
| Qwen3.5-9B | 2,3 | 4 | 8 |
| Qwen3.8-27B | 4,5,6,7 | 2 | 8 |

## 数据

`mixed-data/train.jsonl` 共 74 条轨迹、3521 条决策样本。

- 压缩专家 reasoning 数据：排除整条超过 100 turn 的轨迹，保留 64 条、3188 个样本。
- Gemini Flash：保留 8 条、238 个样本。仍排除此前用户指出可疑的 RoboCasa StackCans 单轮轨迹。
- Gemini Pro：保留 2 条、95 个样本。
- 已找到的 GPT 采集报告为 `preflight_failed`、`episodes_launched: 0`，没有成功导出可加入。

按环境：LIBERO 1855、MetaWorld 1242、ManiSkill 326、RoboCasa 98。
来源和过滤统计见 `mixed-data/summary.json`。图像引用共享盘已有文件，不重复复制。
只加入这些来源；未混入旧 GLM、旧 Qwen、DeepSeek 或无 reasoning 的重复专家导出。

## 每轮输入与输出

输入是 Swift 的 `messages` 和有序 `images`：

1. system：任务描述、可用工具、动作维度/边界、控制坐标系、夹爪语义、XML 协议。
2. 最近历史：assistant 的先前 XML 决策，与 user 的执行回执（请求/实际控制步数、错误和终止状态）。
3. 当前 user：轮次、剩余预算、机器人关节和 TCP 状态、实测状态差分、图像相机及时间标签。
4. 当前输入图片：以 `<image>` 占位符按序对应 `images`，通常包含上一决策前主图及当前可用相机视角；
   遵循采集的 profile 和去重规则，不重放全部旧图片。

目标是最后一个 assistant 的原样 XML `<decision>`：工具名、`reasoning`、动作数组、`num_steps`。
不是预测下一张图片或环境 reward。历史 assistant `loss:false`，当前 assistant `loss:true`，
Swift 使用 `last_round+ignore_empty_think`，system/user/图像/历史回答不监督。

专家动作来自独立成功重放，当前 reasoning 是 Gemini 对固定动作的事后说明，
历史专家 XML 中 reasoning 可能为空；不把说明称为专家原始思维。
在线 Gemini 样本复用实际请求和原始最终 XML；不将隐藏 thinking 添加为训练标签。

## 训练与观察

现有 Swift 4.4.2；三个模型 config 均采用 `qwen3_5` 架构/模板。
语言部分 LoRA rank16/alpha32，冻结视觉编码器和 aligner，BF16、SDPA、梯度检查点。
训练 3 epochs，峰值学习率 5e-5、cosine、warmup 3%，每 100 步存 checkpoint，保留最近 5 个。
不切分验证集。max_length=8192；每个模型先全量编码验收，再运行 DDP。
CLI 的 `truncation_strategy=delete` 配合 `strict=true`，编码异常报错，禁止静默丢样本。
关闭 Swift 的 `check_model` 在线版本检查，避免无公网机器在 Trainer 初始化时等待 ModelScope。
三个模型全量编码均通过，长度范围 1817–7935 tokens。
H200 上的 FLA 反向传播要求 Triton >=3.7.1；训练通过 PYTHONPATH 使用
`logs/triton-fixed/` 中独立安装的 Triton 3.7.1，不修改机器原有 Swift 环境。

入口：`prepare_mixed_data.py` → `launch_mixed.py` → `train_mixed.sh`。
每模型日志 `mixed-runs/<模型>/train.log`，编码报告 `encoding.json`，
checkpoint 在 `checkpoints/`，后台 supervisor 退出后写 `exit_code`（0 表示成功）。
`launch_mixed.py` 检查已有 PID 记录以避免重复启动。训练中不修改混合数据文件。
