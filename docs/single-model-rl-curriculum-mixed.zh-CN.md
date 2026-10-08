# mixed 模型的单模型 RL 课程学习

## 新模型与单题预算

模型为 `/inspire/qb-ilm/project/exploration-topic/ky26060/model/Qwen3.5-4B-embodied-mixed`，SFT 合并记录指向 mixed checkpoint-1320（74 个训练 episode、3521 个样本）。第一阶段从该完整权重新建 LoRA，不恢复旧 RL 训练；后续阶段完整恢复前一阶段的模型、优化器、调度器、RNG 与 global step。

用户指定的 48 指每道题的一条轨迹中，模型与环境最多交互 48 轮，不是训练 update 数。此次明确采用 48 轮（此前运行实际上是 60 轮）。每轮生成上限由 768 增到 1536 token；整条轨迹生成预算为 48 × 1536 = 73728 token，高于此前 46080。上下文上限由 16384 增到 32768，保留最多 5 张原生图像、4 组历史。最多 400 个控制步、一次动作最多重复 10 个控制步保持原设置。

仍使用 8 H200、group 32、每卡 batch 4、学习率 5e-6、LoRA rank 8/alpha 16、beta 0.01。输入与 XML 协议、一致性审计、原生任务成功二值奖励保持原实现；不添加中间奖励，也不替模型修复输出。

## 课程顺序

| 阶段 | 任务 | 每题初始化 seed 数 | 阶段 epoch | 新增 update |
|---|---|---:|---:|---:|
| 01_easy | reach-v3、button-press-v3、button-press-topdown-v3、drawer-close-v3 | 4 | 2 | 32 |
| 02_medium | reach-v3、push-v3、window-open-v3、handle-press-v3、lever-pull-v3、PullCube-v1、libero_90_task35、PickSingleYCB-v1 | 2 | 2 | 32 |
| 03_original32 | 原来的 20 道题 + 12 道 ManiSkill，共 32 道 | 1 | 1 | 32 |

第一阶段优先不需要抓取或多对象规划的单目标任务。第二阶段增加接触控制、推动和抓取，保留 reach 作为简单任务复习。第三阶段才回到绘图、灵巧手、双臂、复杂装配和厨房多步任务。这是根据任务结构确定的初始难度排序，mixed 模型的实际成功率需通过采样验证。

三个阶段分开启动，不会把阶段一起随机打乱；阶段内也按固定顺序训练，第一组明确为 reach-v3。同一 GRPO group 的 32 条轨迹使用相同原生初始化，跨 group 用不同 seed 增加覆盖。全部任务通过 baseline-40 和 holdout 排除校验。

总计划为 5 个阶段 epoch（2+2+1）、96 次 update，累计终点为 32、64、96。通过显式累计 max_steps 及阶段切换时的 ignore_data_skip 保留完整训练状态并更换数据集。这里的 update 数与每条轨迹 48 轮是两个独立维度。

## 升级与全零奖励处理

阶段 1、2 的完整训练结束后，只有同时满足以下条件才自动进入下一阶段：

- 阶段平均原生成功率至少 10%。
- 至少 2 种任务出现成功。
- 至少一组出现成功与失败混合（有 GRPO 奖励差异）；若整体成功率达到 80%，也允许视为已掌握后升难度。

连续 4 个 group 原生 reward 全零时自动保存 checkpoint 并停止该阶段。监督进程将状态记为 `needs_reward_signal`，不自动进入难阶段。任务已有成功则重置连续全零计数。这避免在没有任务优势信号时持续花费 GPU。

这些条件是本次执行采用的初始课程规则，后续可根据真实成功率调整；不声称已证明每个简单任务都容易被当前模型解决。

## 实现与验证

实现提交 `b9d73db`：

- `configs/single_model_rl_curriculum_mixed.json`
- `scripts/single_model_rl_curriculum_env.sh`
- `scripts/run_single_model_curriculum.py`
- `agent/training/curriculum.py`

34 项相关测试通过，覆盖阶段顺序、数据预算、baseline 排除、完整 seed 唯一性、旧阶段日志排除、checkpoint 跨阶段传递、升阶条件、全零奖励停止，以及已有输入/梯度/优化器修复。

简单阶段四道任务已通过真实 reset、指纹一致性、原生动作与相机输入检查。新课程监督进程已提交，第一阶段已从新模型启动；实际参数确认 model 为 mixed 路径，resume_from_checkpoint=null，max_turns=48，max_completion_length=1536，vllm_max_model_len=32768，dataset_shuffle=false，group=32、每卡 batch=4。首组是 reach-v3，32 条轨迹实际生成 prompt 校验已通过，采样正在进行。首次 optimizer 更新尚未完成，成功率尚未确认。阶段输出版本为 `v0-20261008-055005`。

## 运行位置

- Notebook：`embodied-selfplay-copy`，分布式训练空间，8 张 H200。
- 运行根目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_curriculum_mixed_20261008`
- 计划：根目录下 `curriculum_plan.json`
- 状态：根目录下 `curriculum_status.json`
- 监督进程记录：根目录下 `curriculum.submission.json`
- 阶段日志：根目录下 `stages/01_easy/train.log`（以后阶段分别为 `02_medium`、`03_original32`）
- 模型产物和 TensorBoard event：根目录下 `formal/trainer/<阶段名>/v*`
- TensorBoard：`openeta-rl-curriculum-mixed-20261008`，已确认 RUNNING，读取整个阶段输出根目录，自动停止上限 72 小时。

当前 notebook 已重启为新实例，旧训练和环境服务已不在；旧 embodied 训练产物保留，旧 TensorBoard 已停止。本次只使用新模型和新的运行目录。
