# RoboCasa 跨进程重置一致性修复

问题：单模型 GRPO 从 checkpoint-5 恢复后，首组 `ScrubCuttingBoard`（seed 20321400）在环境初始化阶段失败。模型恢复已完成，失败发生在采样前的同组初始状态校验。

原因：RoboCasa 1.0.1 的 `Counter.get_reset_regions` 先用 `set` 去重 XML 元素，再按集合遍历顺序给台面区域分配 `geom_N`。XML 元素的对象地址影响顺序，因此同一个 RNG seed 在不同进程里可能选择不同几何区域。原始复现实验中，机器人初始位置差约 2.2 米；固定 PYTHONHASHSEED 仍不能消除差异。

修复位于 RL 工作树 `sim/envs/robocasa/direct_env.py`：在官方 Counter 方法返回后，按区域 offset 和 size 排序，再稳定分配采样索引。保留全部原始区域及其几何数据，沿用原生 RNG 和成功判定。补丁在环境构造前安装，且重复安装无副作用；未知区域标识会显式报错。一致性校验继续执行。

验证输出目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/reset_diagnosis`。

- 修复前四个独立进程显示不同 qpos/qvel/ctrl/求解器状态。
- 修复后四个独立进程（不同 hash seed）导出的完整 MuJoCo 状态逐字段完全一致。
- 相关测试 30 项通过，1 项需要特殊运行环境的条件测试跳过。
- 其余四个 RoboCasa 训练任务各 8 个独立环境、ScrubCuttingBoard 的 32 个独立环境均通过真实 MCP 验证，组内初始状态完全一致，单步执行和关闭正常。完整结果见 `group_acceptance.json`，验证进程退出码为 0。

续训继续使用原来的 32 道题、group 32、每卡 batch 4、5 个 epoch，总共 160 次更新，从 checkpoint-5 恢复。模型输入与 XML 协议沿用原实现；TensorBoard 沿用 `openeta-rl-32g32-20261006`。

本次续训于 2026-10-06 22:39 UTC 提交到 `embodied-selfplay-copy` 的 8 张 H200。提交记录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/resume_ckpt5_fixed_02/train.submission.json`。训练输出：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/formal/trainer/resume_ckpt5_fixed_02/v0-20261006-223923`。恢复模型、优化器、调度器和 RNG 状态（`resume_only_model=false`），总更新数仍为 160。

第一次带环境修复的续训 `resume_ckpt5_fixed_02` 已完成全部 32 条真实轨迹，最长 51 轮，随后在 rank 3 的梯度前向阶段发生 CUDA OOM。Swift 4.4.2 的 Liger fused GRPO loss 直接使用展开后的全轮次批次，绕过条件化分块。新增 `ConditionalTurnsMixin.training_step` 对每轮单独前向、按 episode 总目标 token 数加权、立即 backward 并释放计算图；各 rank 用零权重补齐到相同轮次数，最后一次 backward 同步 DDP 梯度。保留每卡 4 个 episode、group 32 和单次 optimizer 更新。相关 26 项测试通过，8 H200 实测不同 rank 分别 2–9 轮时得到一致的正确梯度 3.25。实现提交 `c91f669`。

正式续训再次于 2026-10-06 23:01 UTC 提交，当前提交记录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/resume_ckpt5_fixed_03/train.submission.json`；输出目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/formal/trainer/resume_ckpt5_fixed_03/v0-20261006-230101`。恢复来源仍是原 checkpoint-5。

`resume_ckpt5_fixed_03` 完成最长 56 轮的采样和逐轮 backward，训练计算实测显存约 11–25 GB，随后 fused Adam 更新报 moment 与 parameter dtype 不一致。checkpoint-5 的原 moment 是 BF16；恢复时 PEFT 参数暂为 FP32，PyTorch 将 moment 转为 FP32，而 colocate 阶段又将参数变为 BF16。提交 `10cd652` 在更新前对齐 Adam moment 的 dtype/device，保留原 step 和 moment 数值。使用原 checkpoint-5 的实际 moment 在 GPU 上验证 fused Adam 成功更新到 step 6。

最新正式续训为 `resume_ckpt5_fixed_04`，2026-10-06 23:23 UTC 提交，恢复来源和训练参数保持不变，TensorBoard 继续监控同一根目录。提交记录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/resume_ckpt5_fixed_04/train.submission.json`。

最终验证通过：`resume_ckpt5_fixed_04` 的首组 32 条真实轨迹最长 60 轮，成功完成 `global_step=6/160`、`epoch=0.1875`。`checkpoint-6` 包含 adapter、optimizer、scheduler、8 个 rank RNG 状态及 trainer_state；恢复后的 optimizer moment 对齐记录为每 rank 992 个 moment，原 step 连续递增。TensorBoard 已显示本次 run 的 `train/reward` step 6。训练继续运行。

本次 update 用时约 21 分钟，记录显存 30.78 GiB。reward=0，表明本组尚未完成任务；此前模型 XML 错误较多的问题仍然存在。修复目标是恢复环境和训练链路，奖励仍保持原生任务完成的二值判定。
