# 单模型 GRPO：32 道任务、group 32、5 epoch

保留原来的 20 道 SFT 任务，增加 12 道 ManiSkill 任务。任务清单为 `configs/single_model_rl_32tasks_manifest.json`，训练配置为 `configs/single_model_rl_32tasks.json`。数据生成同时检查冻结的 `docs/baseline-40-tasks.json` 和原 SFT holdout；32 个环境 ID 均与 baseline 不重叠。

新增任务：

- `DrawTriangle-v1`、`DrawSVG-v1`：Panda Stick，6 维末端位姿增量，无夹爪。
- `TwoRobotPickCube-v1`、`TwoRobotStackCube-v1`：一个策略共同控制两台 Panda，14 维动作，按原生 action space 的机器人顺序拼接；两张当前手腕图加当前/上一主视角，最多 4 图。
- `RotateValveLevel0-v1` 至 `RotateValveLevel4-v1`：DClaw，9 维关节增量。Level 4 的目标旋转方向在 reset 后从官方任务定义生成文字指令。
- `RotateSingleObjectInHandLevel0-v1` 至 `RotateSingleObjectInHandLevel2-v1`：Allegro Hand，16 维关节增量，官方累计旋转成功判定。

所有主视角为原生 256×256；双臂手腕为原生 128×128。沿用共享 InputBuilder、最近四组决策历史、严格 XML 和每轮精确 token/logprob/loss mask 对齐。控制器类型、动作顺序、归一化、增量尺度和来源哈希写入任务 profile，不能将手部关节动作描述成 Panda 末端动作。物理指纹读取状态和任务张量，不调用有副作用的 evaluate；成功奖励继续只取真实 step 返回的官方 success。

训练使用 `embodied-selfplay-copy` 的 8 张 H200。每卡 4 条轨迹、梯度累积 1，全局 batch=32；同一个任务/种子采样 32 条轨迹组成一个 GRPO group。32 道任务 × 5 epoch，共 160 次更新、5120 条轨迹；正式训练由 epoch 驱动（max_steps=-1），不继承旧的 20-step 上限。每条轨迹最多 60 次决策、400 控制步，每轮 768 tokens。

新实验从同一 Qwen3.5-4B base 初始化新的 LoRA，旧 group 8 实验的 checkpoint 和日志保留。启动前 source `scripts/single_model_rl_32tasks_env.sh`，再调用原有提交脚本。训练默认 `report_to=tensorboard`；TensorBoard 服务读取正式 trainer 目录，自动停止时间为 72 小时。

输出根目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006`。

验证：32/32 真实任务 preflight 通过（`preflight_v3/results/summary.json`）；47 项单元/回归测试通过（`unit-tests.log`）。group 32 实际训练、编码验收、正式提交和 TensorBoard 曲线证据在提交后补充。
