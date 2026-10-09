# 第 22 次更新的条件输入对齐修复

2026-10-07 02:20 UTC，`resume_ckpt5_fixed_04` 在完成第 21 次更新后停止。下一组 `MoveToFreezerDrawer` 的 rank 4 报错 `training condition differs from sampled prompt`。checkpoint-20 和 checkpoint-21 均完整保留。

## 原因与修复

失败组的一条轨迹首次生成的原始 token 以 `[248068, 271, 248069, 271]` 开头，即 `<think>\n\n</think>\n\n`。这一段是模型在已包含相同非思考前缀的 prompt 之后实际生成的输出。Swift 4.4.2 的 `replace_assistant_response_with_ids` 根据 token 内容判断前缀是否已存在，因此没有补入属于 prompt 的前缀，训练 prompt 比实际采样 prompt 少了 4 个 token。

`agent/training/conditional_turns.py` 现在为训练表示显式添加零损失模板前缀，随后完整保留原始生成 token 和 loss mask。采样审计中的原始目标 token 仍是严格比较依据，所有原始 logprob 保留。模型自己生成的相同前缀仍然是有损失的生成目标，不做输出修复。一致性检查继续严格执行。

多模态分块也使用 `encode_sample` 编码原始 token 与 mask，沿用初次批编码的条件输入；避免 Swift 默认分块路径重新编码展示文本。逐轮反向传播、episode 权重、原生二值成功奖励保持原实现。

实现提交：`d485dc0`。

## 验证

- 相关测试 30 项通过，包括生成前缀与模板前缀相同、普通生成内容、多模态分块、梯度权重与优化器状态恢复。
- 对失败组 32 条真实轨迹、103 个决策回放：旧实现复现一个 prompt 少 4 token 的失败；新实现全部通过实际采样 prompt、训练目标与 logprob 数量一致性校验。
- 证据：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/reset_diagnosis/prefix_replay_acceptance.json`。
- 可复现脚本：`scripts/validate_conditional_prefix_replay.py`。

## 续训

2026-10-07 03:36 UTC 在 `embodied-selfplay-copy` 的 8 H200 上提交 `resume_ckpt21_fixed_05`。从 `resume_ckpt5_fixed_04/v0-20261006-232325/checkpoint-21` 恢复模型、优化器、调度器和各 rank RNG 状态。训练继续使用 32 道题、group 32、每卡 batch 4、5 epoch、总 160 次更新。TensorBoard：`openeta-rl-32g32-20261006`。

输出目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/formal/trainer/resume_ckpt21_fixed_05/v0-20261007-033619`。

提交记录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/resume_ckpt21_fixed_05/train.submission.json`。

实际第 22 次更新验证通过：首组 32 条真实轨迹、121 个决策（最长 24 轮）在 8 个 rank 上通过 prompt 与 loss token 一致性校验，成功完成 `global_step=22/160`、`epoch=0.6875`，保存完整 checkpoint-22（adapter、optimizer、scheduler、8 个 rank RNG、trainer_state）。TensorBoard 已确认本次 run 的 step 22 指标；训练进入下一组，进程仍在运行。本组 reward=0，单次更新耗时约 10 分 12 秒，记录显存 38.2 GiB。
