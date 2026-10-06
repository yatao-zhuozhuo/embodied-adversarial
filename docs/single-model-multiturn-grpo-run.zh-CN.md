# 单模型多轮 GRPO 实现与运行

分支 `feat/rl` 位于 `.worktrees/rl`，基础为包含输入修复的 `a97439f`。训练使用 `embodied-selfplay-copy` 的八张 H200，模型为本地 Qwen3.5-4B，从 base 新建一个 LoRA；没有 Alice、冻结对手或教师 API 请求。

## 训练协议

输入直接调用 `two_tool_input` 和 `InputBuilder`：原生相机 profile、上一决策前主图与当前视角、至多四组历史、实际 tokenizer 预算、图片去重、隐藏状态隔离。输出直接使用 `two_tool_xml.parse_decision`，模型工具仅为 `observe_env`、`step_env`，`task_complete` 只是结束响应。

每个任务的同 seed 独立 rollout 组成 GRPO group。新增宿主专用状态指纹接口核对物理状态、场景配置及可取得的目标信息；接口需要 host token，不在 MCP 模型工具目录，也不进入模型输入。MetaWorld 构造与 LIBERO reset 显式传递 seed，避免只设置 reset 参数却未固定实际任务。

成功检查只读取原生 `step_env.info.success`。成功为 1，正常失败为 0；环境 dense reward 和格式正确率仅留档，不计入奖励。基础设施异常中止整个训练批次，不转换成失败标签。

训练仍使用 `InputGRPOTrainer`：完整 episode 结束后按真实决策上下文拆样本，保留原始 token/logprob/mask，逐 token 核验采样条件与训练条件。组内优势按 episode 去重，环境反馈和历史不重复监督。零方差组没有任务学习信号，不以 checkpoint 存在冒充能力提升。

## 入口

| 入口 | 用途 |
| --- | --- |
| `scripts/create_single_model_rl_dataset.py` | 从原 SFT 二十题清单生成任务、seed 和预算，不读取教师答案 |
| `scripts/single_model_rl_env.sh` | GPU Notebook 中的模拟器和共享路径配置 |
| `scripts/launch_single_model_rl_mcp.sh` | 独立 MCP 服务和受保护的宿主 token |
| `scripts/preflight_single_model_rl.py` | 同 seed 重置、宿主物理指纹、图像 profile、原生动作和成功字段检查 |
| `scripts/run_single_model_grpo.sh` | 单策略 Swift colocate GRPO |
| `scripts/submit_single_model_rl.py` | 后台启动并记录 PID、主机、进程身份、日志和退出码 |
| `scripts/stop_single_model_rl.py` | 按提交回执和进程身份停止本次任务及其子进程 |
| `scripts/summarize_single_model_rl.py` | 汇总真实轨迹、成功、错误、训练输入校验和 checkpoint |

超参数起点记录在 `configs/single_model_rl_20tasks.json`，实际环境覆盖、版本和源码哈希记录在每次训练的 `run_manifest.json`。完整二十题短跑使用 G=8、60 轮、400 控制步、单轮 768 tokens、episode 46080 tokens、上下文 16384、一个 epoch。新运行从 base 初始化；canary 权重不自动混入正式训练。

GPU 空间离线，启动脚本使用本地权重并设置 `--check_model false`，避免 Swift 初始化时等待 ModelScope 在线版本检查。运行时需要本项目 HDD 与 qb-ilm 挂载；RoboCasa Python 和 ManiSkill YCB 资产路径见 [INSPIRE.md](../INSPIRE.md)。

## 本轮验证证据

运行根目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_20261006`。

- `preflight_acceptance.json`：四环境全部 20 题通过，包含两次独立同 seed reset 的宿主状态对比和真实单步动作检查。
- `multibench_encoding.json`：四 benchmark 加 ManiSkill 手腕子 profile，共五种 profile、35 个逐轮编码检查通过；使用预检真实图片和明确标记的测试决策，不作为模型任务成功证据。
- `tests.log`：64 项 reward、XML、输入、分组和 scheduler 回归测试通过。
- `hook_validation.log`：实际 Swift hooks、跨 rank 变长优势分发、episode 权重反传和实时首帧路由通过。

- `final_environment_tests.log`：最终环境接口与单模型回归为 67 项通过、1 项条件跳过。
- `canary_v3/acceptance.json`：八卡单任务训练退出码 0，8 条轨迹、26 次决策、8 rank 输入与 loss 校验通过，保存完整 checkpoint。
- `canary_four_bench/acceptance.json`：重载前一 checkpoint 后，四环境各两条轨迹，共 30 次决策；8 rank 校验通过，完成一次更新并保存 checkpoint，退出码 0。
- checkpoint 包含 adapter、optimizer、scheduler、8 rank RNG 和 trainer state，496 个 adapter 张量均为有限值。这里实际验证了 adapter 重载；完整 optimizer 断点续训仍由 Swift 标准入口处理，未另做中断恢复实验。

两个 canary 均为 0/8 成功，episode 奖励标准差为 0；单任务 canary 的非零梯度来自 KL 项，不能报告为任务学习。严格 XML 的有效率偏低，正式训练会继续记录格式错误和零方差组。正式训练从 base 新建 LoRA，不加载上述 canary adapter。

## 完整二十题训练

预定正式运行目录为本轮根目录下 `formal/`，使用独立的 `formal_service/` MCP 服务。后台提交使用 `scripts/submit_single_model_rl.py`，运行后的 PID、主机、日志和退出码分别保存在 `train.submission.json`、`train.log`、`train.exit`。

完整训练的提交状态与首批采样检查在交付时记录；完整 epoch 结果不由 canary 推断。
