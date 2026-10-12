# Unified v6 reasoning pilot third SFT

来源：`SFT_data/expert_replay_unified_v6_reasoning_pilot_third_20261011`。
仅使用本批完整导出：77 条成功轨迹、3487 条样本，LIBERO 1402、MetaWorld 1579、ManiSkill 506。
全部数据用于训练；保留真实历史窗口、有序图像以及当前 XML，历史 assistant 不计算 loss。
当前 XML reasoning 是 Gemini 对固定专家动作补写的步骤目的，不是专家原始隐藏思维。

在 `embodied-selfplay-copy` 上从原始 Qwen3.5-4B 和 Qwen3.5-9B 基座训练 LoRA，分别使用 GPU 6 和 7。
Swift 4.4.2，语言部分 all-linear rank16/alpha32，冻结视觉编码器和 aligner，BF16/SDPA。
每卡 batch1、梯度累积8、3 epochs（1308 个优化步），学习率5e-5、cosine、warmup3%。
每100步保存 checkpoint，保留最近5个；不加载此前混合数据 adapter。
模板 qwen3_5，loss_scale=last_round+ignore_empty_think；最大长度8192、strict=true，禁止静默截断/丢样本。
两模型均通过全部3487条编码验收，长度1815–5623 tokens。
使用已有 Triton3.7.1 overlay，并关闭无公网环境的 ModelScope 在线版本检查。

准备的数据及统计：`v6-pilot-third-20261011/train.jsonl`、`summary.json`。
日志：`v6-pilot-third-20261011/<模型名>/train.log`。
Checkpoint：同目录 `checkpoints/`；编码报告 `encoding.json`，后台 PID 为 `pid`，退出码为 `exit_code`。
启动入口 `launch_v6_pilot.py`、训练脚本 `train_v6_pilot.sh`。
图像引用共享盘原始文件，训练期间须保持来源目录可用。
