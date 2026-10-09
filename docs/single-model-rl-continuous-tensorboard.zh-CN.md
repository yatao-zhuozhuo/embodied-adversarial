# 续训的连续 TensorBoard 曲线

2026-10-09 已启用 `openeta-rl-curriculum-continuous-20261009`，工作空间为分布式训练空间。新面板只展示一个 Run：`continuous`。已验证全局 step 1–9 的 reward 连续显示，既有恢复运行的历史没有丢失。旧的 `openeta-rl-curriculum-mixed-fast3-20261008` 分散面板已停止，原日志文件保留。

## 查看

进入新 TensorBoard 的 Scalars，选择 `continuous`。主要指标：

- `train/reward`：一组轨迹的原生二值成功奖励均值。
- `train/success_rate`：同一奖励均值，即该组成功率（0–1）。
- `train/success_percent`：成功率百分比（0–100）。
- `train/reward_std`、`train/loss`、`train/kl`、`train/grad_norm` 等已有训练标量同样保留。

已验证 reward：step 1–9 分别为 0.03125、0、0、0.71875、0.03125、0、0、0.5、0.03125。step 8 成功率显示为 50%，step 9 为 3.125%。这些是实际 checkpoint 的记录，未填充未完成 step 的奖励。

## 连续性与重启

固定日志目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_curriculum_mixed_fast3_20261008/tensorboard/continuous`。

CPU 汇总进程每 15 秒检查 `current_run.json` 指向的当前课程运行。读取最新保存 checkpoint 的 `trainer_state.json`，按累计 global step 回填或追加标量。课程的 checkpoint 保留整个训练历史，因此首次启动即可显示早期运行的数据，不需要等待新训练更新。

该进程与训练独立，不修改模型输入、奖励、数据、梯度或 optimizer。此次训练没有停止或重启。未来续训启动脚本会自动启动汇总进程，并更新活动运行指针；所有阶段和恢复运行仍写入同一个 `continuous` 日志目录。汇总进程具有 PID/主机/start_ticks 检查和文件锁，防止同时启动多个写入者；重复采集不会重复写点，checkpoint 回退会隐藏未来失效点。

如果 Notebook 重启，汇总进程会一起消失；下一次通过当前课程脚本恢复训练时自动重启并补齐历史。TensorBoard 平台服务独立于 Notebook，自动停止上限为 72 小时；到期后需重新启动面板，已有 event 文件仍保留。

启动入口：

```bash
/opt/openeta-swift/bin/python scripts/single_model_tensorboard_series.py \
  --source-root /inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_curriculum_mixed_fast3_20261008 \
  --start
```

进程记录和日志位于固定目录的父目录 `tensorboard/collector.submission.json` 与 `tensorboard/collector.log`。

实现提交 `7d26e79`。10 项相关测试通过，包括实际 TensorBoard event 读取、重启连接、回退清理、零 reward 展示、历史运行谱系选择和课程恢复回归。新面板 API 已验证仅存在 Run `continuous`，reward 与成功率均具有 9 个有效标量点。
