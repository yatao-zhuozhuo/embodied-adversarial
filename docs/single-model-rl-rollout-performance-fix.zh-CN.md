# 课程 RL 采样性能修复

## 确认的瓶颈

原 `single_model_rl_curriculum_mixed_20261008` 首组在近 2 小时仍只到约 22 轮，没有产生 optimizer checkpoint。输出平均约 132 token，不能用 1536 token 上限解释分钟级延迟。

最初的栈采样看到模型进程等待环境返回、其他 rank 等待 collective，因此先修交互控制同步和同步环境调用。然而带模型运行验证表明，只修这些还不够，之前把它判断为单纯 NCCL 等待是不完整的。

进程级 GPU 监控进一步确认，高占用来自 **graphics 渲染进程**。隔离掉模型和跨卡通信，只使用 32 个真实 MetaWorld 环境进行压力测试，仍然复现了延迟：三轮 32 环境并发执行分别耗时 78.43、63.32、75.66 秒；平均 RPC 为 52.00、51.18、62.12 秒。该结果排除了模型推理为主要原因。

MuJoCo worker 的 EGL/OpenGL 操作需要稳定的线程所有权；固定线程后，32 个独立 GPU 渲染进程的压测仍然慢，而只改变为 8 个进程之后显著改善，因此主要瓶颈是每张 GPU 多个 graphics 进程争用，线程固定是额外的正确性措施。单独一个环境的原生 step 约 4 毫秒，额外热启动 render 约 4 毫秒，整套 worker 序列化单步约 0.23 秒；说明并发进程调度是严重放大因素。

## 最终实现

- 交互控制改为纯 CPU rendezvous store：只发送布尔状态，不调用任何 tensor/object collective，也不修改默认 NCCL 梯度组。初次和后续推理前，先确保所有环境操作完成。
- 独立环境调用使用 `asyncio.to_thread` 并发执行；状态、输入构建与模型 tokenizer 保持主事件循环顺序，避免共享 tokenizer 的线程竞争。
- MetaWorld、LIBERO 与 RoboCasa 的 create/reset/step/render/observe/close 统一执行在 worker 所属主线程；保留 worker 内 GL 锁。
- MuJoCo benchmarks 最多 8 个 worker，即每张 GPU 一个 graphics 进程。每个进程可以管理多个独立环境，**group 32、32 条独立轨迹不变**。ManiSkill 仍使用原全局池上限。
- 记录环境执行及输入构建耗时作为审计数据，不加入模型输入或奖励。

关键提交：`520fa34`、`f5d7311`、`1b198b2`、`4f36c65`。模型、原生 RGB 相机、协议、48 轮上限、1536 token/轮、73728 总生成预算、32768 上下文、原生二值成功奖励、课程升阶与连续 4 组全零保护保持原设置。

## 已完成的验证

- 原有训练、worker、线程所有权、CPU 控制同步、环境并发与课程测试通过。
- 8 GPU 实测纯 CPU 同步与 NCCL 梯度并存，8 个 rank 获得一致正确梯度 4.5。
- 32 环境压力测试保留相同初始化：32 个 native fingerprint 完全一致，全部正确执行一次控制步并返回有效图像。
- 8 worker 下三轮并发耗时为 **1.43、1.48、1.59 秒**；平均 RPC 为 **0.94、0.97、1.13 秒**，最大 RPC 不超过 1.58 秒。相对原 32 worker 测试，环境阶段约快 40–50 倍；这不是完整训练 update 的加速倍数。

压力测试证据：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_curriculum_mixed_fast3_20261008/diagnostics/environment_stress_8workers_acceptance.json`。失败的 32 worker 压测记录也保留在相同 diagnostics 目录。

## 当前正式运行

旧运行均已停止并保留历史产物，没有 checkpoint 可以续训；从相同 mixed 模型开始重新跑简单阶段。

- 根目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_curriculum_mixed_fast3_20261008`
- 监督进程：根目录 `curriculum.submission.json` / `curriculum_status.json`
- MCP：根目录 `service_8workers/mcp.submission.json`，端口 19775
- 阶段日志：根目录 `stages/01_easy/train.log`
- TensorBoard：`openeta-rl-curriculum-mixed-fast3-20261008`
- 启动时间：2026-10-08 09:29 UTC

完整模型运行已完成 5 次 update 并保存 checkpoint-5：到达目标 2/64 成功、关闭抽屉 23/32 成功，两种按按钮任务均无成功。首个 update 约 71 分钟，说明环境压测改善不代表完整训练耗时问题已经解决。2026-10-08 15:24 UTC，一个环境 worker 以 -11 退出，训练在第 6 组采样时因连接断开停止。

## checkpoint-5 恢复

已重建环境服务、通过四道简单任务 preflight，从原 checkpoint-5 完整恢复模型、optimizer、scheduler、RNG 和 dataloader 位置（resume_only_model=false，ignore_data_skip=false）。当前监督进程根目录为原运行的 `formal/trainer/resume_ckpt5_01`，实际首组初始化为 `metaworld:button-press-v3:20490100`，对应原数据集第 6 组，确认没有从头重跑。旧阶段成功记录保留用于课程升阶判定，后续阶段继续自动接续。TensorBoard 沿用原名称和监控根目录。恢复入口提交 `250cf6f`，5 项课程测试通过。

此次按用户要求确认恢复采样后交回，不持续主动轮询，也不声称第 6 次 update 已完成。原 worker 底层 -11 崩溃原因尚未彻底修复，当前使用重建后的干净环境服务。
