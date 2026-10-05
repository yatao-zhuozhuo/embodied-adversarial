# 四模型 × 四环境 baseline 评测计划

协议：`rgb_native_actions_v1`。四模型分别是 DeepSeek Flash (`deepseek-flash`)、Qwen3.8-27B、Qwen3.5-9B、Qwen3.5-4B。每模型运行同一份 40 题，每题一次，共 160 条 episode。

## 题目

| 环境 | 数量 | 来源 |
|---|---:|---|
| LIBERO | 10 | spatial/object/goal/libero_10 按 3/3/2/2 分层 |
| Meta-World | 10 | 仓库去重任务池 |
| ManiSkill3 | 10 | 单臂 Panda 桌面兼容子集 |
| RoboCasa | 10 | 官方 target50，5 个原子任务＋5 个组合任务 |

一题为一个任务及固定初始场景。抽样 seed=20261002，分层内按 SHA256(seed:task_id) 排序无放回选择；每环境第 i 题 reset seed=20261002+i。完整候选顺序和最终题目在 `manifests/`，与模型表现无关的初始化兼容选择记录在 `configs/task_exclusions.json`。不因模型失败而替换题目。

仓库实际使用 ManiSkill3，不能将当前结果标为 ManiSkill2。LIBERO 使用原生 seed reset，不声称采用官方 init-state 索引协议。题单校验和见 `manifests/SHA256SUMS`。

## 输入与工具

模型输入包括任务文本、主相机 512×512 RGB、当前腕部 RGB（可用时）、相机参数、机器人自身状态和最近最多 12 轮动作文本历史。ManiSkill 腕部保留原生分辨率。不提供物体真值、目标位置、深度图或奖励塑形信息。

仅允许 `observe_env` 和 `step_env`，后者一次执行 1–10 个物理步。模型可返回 `task_complete` 停止，但不能给自己计分。host 控制 create/reset/close，reset 不附加隐式 settling 动作。原生动作维度分别为 LIBERO 7、Meta-World 4、ManiSkill3 7、RoboCasa 12，使用模拟器返回的真实动作边界。

本轮不使用完整 Agent 的外部感知、抓取估计、代码执行、任务专用 playbook 或跨 episode 学习。统一提示词、通用技能和动作说明冻结在 `snapshots/native-actions-v1/`，四模型共享。

## 推理和预算

| 项目 | 配置 |
|---|---|
| 解码 | temperature=0、max_tokens=1024、thinking 关闭 |
| 上下文 / 图像 | 32768 tokens / 最多两张当前图像 |
| 单题上限 | 100 次决策、200 次模型工具调用、3600 秒、5,000,000 tokens |
| 环境步数 | 保留各任务原生 horizon；与决策轮数区分 |
| 并发 | 每模型任务内部一次一个 episode、一条模型请求 |
| provider | 单次请求，无跨模型 fallback |
| 重试 | 仅基础设施错误最多两次尝试；模型任务失败不挑选重跑结果 |

每次尝试从空模型历史开始。初始化配对检查要求机器人状态一致；Meta-World RGB 允许 MAE≤1/255，其余环境初始 RGB 必须一致。ManiSkill3 使用固定软件 Vulkan 渲染，耗时应按环境分别报告。

## 评分与验收

主指标为环境原生 success@1：LIBERO `check_success()`；其余环境原生 `info.success`。worker 在序列化前归一化单元素布尔值，并记录 episode 内是否曾成功，避免重复动作中间的成功被遗漏。正奖励、模型声称成功、到达 horizon 均不等同于任务成功。

原生判据缺失视为基础设施错误。每模型报告成功数/40，各环境报告成功数/10，并记录模型 tokens、耗时、工具调用、资源超限和基础设施错误。10 题中的单题对应 10 个百分点，不对小差异做强排名。

正式任务先验证文件快照、四套环境原生评分链路和独立两轮模型动作 canary，再运行固定题单。原始请求、响应、图像、初始化证据、提交回执和诊断日志只保存在本地输出目录，不纳入 Git。运行数据与有效正式成绩应分开管理。
