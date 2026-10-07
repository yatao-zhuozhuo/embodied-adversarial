# 从 Qwen3.5-4B-embodied 重新开始单模型 RL

2026-10-07，按用户指令停止原 `single_model_rl_32g32_20261006` 训练及其环境服务，删除整个旧运行目录（约 5.3 GB，包含 checkpoints、轨迹、训练日志与 TensorBoard event 文件），并停止、删除旧 TensorBoard `openeta-rl-32g32-20261006`。仓库代码、任务清单与用户指定的新模型保留。

新训练于 2026-10-07 11:04 UTC 提交到 `embodied-selfplay-copy` 的 8 张 H200。模型：`/inspire/qb-ilm/project/exploration-topic/ky26060/model/Qwen3.5-4B-embodied`。模型包含完整的两份 safetensors 权重，使用该模型权重初始化新的 LoRA。实际 args 已验证：`model` 为指定路径，`adapters=[]`，`resume_from_checkpoint=null`；从 global step 0 重新开始，没有复用旧 RL adapter、优化器或训练进度。

保持原训练设置：32 道题（原 20 道 + 不在 baseline-40 中的 12 道 ManiSkill）、group 32、每卡 batch 4、5 epoch、总 160 次更新、每轮最多 768 token、最多 60 轮/400 控制步。输入与 XML 协议及原生二值任务成功奖励保持不变。沿用已验证的环境重置确定性、逐轮反向传播、优化器 dtype 与生成前缀对齐修复。

- 配置：`configs/single_model_rl_embodied_32tasks.json`
- 环境入口：`scripts/single_model_rl_embodied_32tasks_env.sh`
- 实现提交：`cdd5d20`
- 新运行根目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_embodied_32g32_20261007`
- 训练输出：上述目录下 `formal/trainer/v0-20261007-110518`
- 提交记录：上述目录下 `train/train.submission.json`
- TensorBoard：`openeta-rl-embodied-32g32-20261007`，位于分布式训练空间，已确认 RUNNING，自动停止时间为启动后 72 小时。

旧运行的诊断证据目录也随旧产物删除。之前仓库文档保留历史过程；其中旧运行的 checkpoint/日志链接不再可用。

启动验证通过：首组 32 个环境均已进入真实采样，第一轮 32 个请求的实际生成 prompt 与保存的输入快照全部一致，已推进到第二轮。训练进程与环境服务运行正常，尚未完成首次 optimizer 更新。
