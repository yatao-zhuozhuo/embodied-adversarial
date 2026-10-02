# ManiSkill SFT 轨迹查看器设计方案

## 目标与边界

给 `SFT_data/` 做一个本机、只读的离线查看器，让人能在几十秒内回答四个问题：采到了哪些任务、哪条轨迹值得看、机器人每一步为什么这样做、成功或失败的证据是什么。优先服务 GLM/Qwen 两位教师的 ManiSkill 数据质检，不改变采集、验收或导出逻辑。

现有 `sim/mcp_server/dashboard_html.py` 是实时环境相机页，不负责历史轨迹。新查看器单独启动，不连接或控制仿真器，不占 GPU。最初的 MP4 只是每个 turn 的画面拼接，缺少动作、工具结果和成功依据；新页面应以 **turn 为主轴**，视频/图片只是其中一个视图。原始记录只有决策前后帧时，明确标注“逐 turn 快照”，不能伪装成连续物理运动录像。

## 用户从打开页面到定位问题

1. 进入总览，直接看到各教师的验收数/目标数、20 个任务的覆盖、失败原因分布、最新采集时间。总览数字以当前索引计算，不写死。
2. 用教师、任务、成功/失败、seed、工具名、turn 数、耗时和错误类型筛选；列表默认按最近时间排序。每行展示首帧/终帧缩略图、任务、模型、结果、turn 数、时长和失败摘要。支持只看“成功但很长”“重复 IK 失败”“token 超限”等预设筛选。
3. 点进轨迹后默认显示首帧、关键事件时间线和最终状态。按 `Space` 播放/暂停，`←/→` 前后一步，`Shift+←/→` 跳关键事件，数字键切相机。URL 带筛选条件和轨迹 ID，方便分享和继续查看。
4. 点击某个 turn，同步看到：动作前/后画面、模型最后一次有效决策、工具参数与回执、reward/官方成功字段、对象和机器人状态变化。需要审计时再展开完整 prompt、原始 JSON 和 SFT 训练样本；默认不把长上下文铺满屏幕。

## 单页布局

```text
┌ 顶栏：GLM / Qwen 进度 · 搜索 · 任务筛选 · 成功/失败 · 刷新索引 ────────┐
│ 左侧轨迹列表          │ 中间视觉区                    │ 右侧步骤解释      │
│ 首/末帧缩略图          │ base / hand / render 切换       │ 决策与理由        │
│ 任务、seed、结果       │ 动作前  ↔  动作后              │ 工具参数/回执     │
│ turn、耗时、错误标签   │ 放大、平移、图像差分            │ reward/物体状态   │
├───────────────────────┴───────────────────────────────┴──────────────────┤
│ 0 ── 1 ── 2 ── ... ── N   关键事件标记 / reward 曲线 / 错误点 / 播放控制 │
└───────────────────────────────────────────────────────────────────────────┘
```

关键事件自动标记 `ik_preview_check`、`move_to`、夹爪开合、抓取/放置、官方成功、工具异常和资源上限；这是导航用的启发式标签，不替代环境回执。播放器提供 0.5×/1×/2×、逐 turn 步进、关键帧跳转。首版只展示保存的帧，不重放仿真；若以后有逐物理步录像，可作为另一条轨道接入。

## 数据源与关联规则

| 来源 | 用途 | 注意 |
| --- | --- | --- |
| `SFT_data/accepted/{glm,qwen}/episodes.v1.jsonl` | 已验收成功轨迹的可信入口与摘要 | 一行是一条轨迹；不能用导出样本行数当轨迹数 |
| `SFT_data/reports/{glm,qwen}.export.json` | 未验收 bundle 及拒绝原因 | 失败与未完成必须和成功分开标识 |
| `SFT_data/raw/<teacher>/sessions/<id>/rollout/manifest.json`、`episodes.jsonl` | 任务、seed、配置、终局与预算 | 原始文件可能很大，按需加载 |
| `transitions.jsonl` | turn、动作前后观察、reward、终止标记、时间 | 以 `turn_index` 为视觉时间轴；相机帧在 observation 的 `metadata.image_artifacts` 中 |
| `model_calls.jsonl`、`tool_calls.jsonl` | 模型决策、校验结果、工具参数与回执 | 有重试/拒绝/辅助调用，不能直接假设 `seq == turn_index` |
| `SFT_data/exported/<teacher>/train.v1.jsonl` | 查看某 turn 最终用于 SFT 的消息和目标答案 | 通过 `source_bundle` + `source_model_call_seq` 反查，不用它判定轨迹成功 |
| `SFT_data/reports/waves/*.batch.json` | 没形成有效 bundle 的 wave/资源错误 | 作为尝试级诊断来源，不混入已验收轨迹 |

每条记录以 `session_id`/bundle 作为内部主键；`episode_id`、seed 用于显示和搜索。构建 turn 时先筛选被校验接受的主规划器调用，再按执行时间和动作语义与 transition 对齐；优先使用显式执行 ID，缺失时按时间窗口与顺序匹配，并记录“关联不确定”警告。工具事件同样优先按执行 ID、其次按时间窗口归属。不能只凭两个 JSONL 文件的行号配对。

成功标签只来自验收索引/官方环境回执；模型自己说“完成”、reward 较高或视频看着像成功都不算。原始帧是 512×512 的主视角 RGB，部分轨迹另有 128×128 手部相机；GLM 虽保存图像，但其训练输入是结构化状态，Qwen 的视觉样本可在 SFT 面板中单独查看。

## 实现方式

- 增加独立的 `scripts/serve_sft_trajectory_viewer.py`，基于仓库已有的 Starlette/uvicorn，默认仅监听 `127.0.0.1`。前端先用原生 HTML/CSS/JS，不引入 Node 构建链；与实时 MCP dashboard 分开部署。
- 建一个可增量更新的 SQLite 索引 `SFT_data/viewer/index.sqlite`：存轨迹摘要、任务/教师/状态、关键事件、帧引用和源文件 mtime。启动/手动刷新时只重扫变化的 bundle，不在每次筛选时遍历所有 JSONL。原始文件仍是事实来源，索引可删除重建。
- 最小 API：`GET /api/summary`、`GET /api/episodes`（分页筛选）、`GET /api/episodes/{id}`（摘要）、`GET /api/episodes/{id}/turns`（精简时间轴）、`GET /api/episodes/{id}/turns/{n}`（详情）、`GET /api/media/{opaque_id}`（图像）。列表与时间轴只返回小字段；完整 prompt/回执在用户展开时读取。
- 缩略图离线缓存为 WebP；当前 turn 的原尺寸 PNG 按需传输，前后帧预取各 2 步。几十 MB 的 `model_calls.jsonl` 不整份发给浏览器。媒体 URL 使用服务端生成的不透明 ID，不接受任意文件路径。
- JSON 解析与 join 写单元测试，前端用 Playwright 或等效浏览器测试覆盖筛选、键盘步进、相机切换、缺帧和失败轨迹。可复用 `agent.runtime.rollout.validate_rollout_bundle` 做离线完整性检查，但不要在每次页面加载时重新哈希全部媒体。

## 安全与异常处理

查看器只读、不提供机器人控制、重放执行或删除接口。服务端限制读取根目录为传入的 `SFT_data/`，解析真实路径后拒绝目录逃逸；HTML 中的任务文本、模型输出和错误信息按纯文本渲染。默认不展示原始 provider request/response 中可能含有的敏感字段；必要时只在“审计”页给经脱敏的片段。媒体和 SQLite 缓存留在本机，默认不开放到公网。

部分 wave 可能只有 batch 错误、没有完整 rollout；部分 rollout 可能缺帧、序号不齐或 join 不确定。页面要给出清晰状态和缺失原因，仍允许查看已有信息，不能将其误标为成功。对重复 seed、同名 episode 也必须保留独立 session。

## 分阶段交付与验收

1. **MVP：能快速看成功轨迹。** 建索引、总览/筛选、三栏 turn 视图、主视角前后帧、键盘控制、官方成功证据；拿 GLM 的 PickCube/StackCube/PullCube 与 Qwen PickCube 各一条做端到端验证。
2. **质检版：能解释失败。** 接入 export report 和 batch report；显示失败分类、资源上限、工具错误、缺帧与 join 警告；增加 SFT 样本反查、状态差分、关键事件导航。
3. **体验版：批量审阅。** 任务级首末帧 contact sheet、轨迹对比（同任务不同教师/seed）、审阅备注与可导出质检报告。备注单独存储，不修改不可变 rollout；真正连续录像仅在采集端以后提供逐物理步帧时支持。

MVP 完成标准：在普通本机浏览器中无需 GPU/重新跑仿真，能在 3 次点击内从任务总览进入指定成功轨迹，1 秒左右切换已索引轨迹的 turn；每个画面、决策与工具回执的来源可追溯到原始 bundle，且不会把未验收轨迹当成功数据。
