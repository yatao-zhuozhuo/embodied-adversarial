# Qwen3.5-4B 多轮 SFT

使用 `SFT_data/gemini_flash_expanded_20261006_single100` 已导出的五条成功轨迹：
LIBERO 四条（116 个有效决策），MetaWorld 一条（17 个有效决策）。排除 RoboCasa。
`data/train.jsonl` 共 133 条样本，`data/images/` 保存 249 张去重后的输入图片。

每条样本复用采集时的真实多轮上下文，仅当前 assistant XML 计算 loss。
历史 assistant 的 `loss` 为 false；不监督教师隐藏 thinking，XML 内的 reasoning 保留。
使用 `qwen3_5` 模板及 `last_round+ignore_empty_think`。

```bash
python SFT-train/prepare_data.py
/opt/openeta-swift/bin/python SFT-train/audit_data.py
bash SFT-train/train.sh
```

全量编码验收结果：133 条，长度 1830–5156 tokens，图片数量与占位符匹配，当前 XML 完整监督。
审计详情在 `data/encoding_audit.json`。

训练使用本机 GPU 0、现有 Swift 4.4.2 和本地
`/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B`。
语言部分 LoRA rank 16 / alpha 32，冻结视觉编码器及 aligner，BF16，SDPA，
batch 1、累积 8、学习率 5e-5、3 epochs、最大长度 8192、超长报错。
全部五条用于训练，无自动验证集拆分。启动脚本尾部可追加 Swift 参数覆盖默认值。

后台启动日志为 `logs/train.log`，启动进程 PID 在 `logs/train.pid`。
Swift checkpoint 和训练指标保存在 `outputs/` 下的运行目录，每个 epoch 保存一次。
数据、日志及模型输出不纳入 Git。

## 增加训练步数

首轮已完成 51 步，三个 epoch 的平均 loss 分别为 0.2548、0.1985、0.1743。
`bash SFT-train/continue_train.sh` 从 `checkpoint-51` 的 adapter 权重继续，
新增 300 步，使用新 optimizer 和 cosine 调度，峰值学习率 2e-5、warmup 3%。
不恢复首轮已经衰减至零的 optimizer/scheduler 状态。
每 50 步保存 checkpoint，输出到 `outputs-continued/`；后台日志为
`logs/continue.log`，PID 为 `logs/continue.pid`。
这仍是五条轨迹的训练集拟合，loss 下降不等于独立环境任务成功率提升。
