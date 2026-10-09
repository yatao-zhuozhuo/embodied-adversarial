# SFT 轨迹查看器 (fronted/)

## 新版两工具轨迹（Meta-World / RoboCasa / LIBERO / ManiSkill3）

新版默认读取 `deepseek_xml_v1`，只收录老师模型自主输出 XML 并实际执行成功的数据，不进入下面的旧 ManiSkill 索引。
可用 `OPENETA_TWO_TOOL_DATASET=qwen_xml_v1` 查看历史 Qwen 数据。在仓库根目录运行：

```bash
uv run python SFT_data/fronted/two_tool_server.py --host 0.0.0.0 --port 8770
```

在 Inspire Notebook 打开 `/proxy/8770/`。页面只列官方 `success=true` 的完整轨迹；
可按 benchmark 筛选、逐步播放多相机快照、查看 `step_env` 原生动作向量、
reward、环境成功标记及 Qwen 当轮原始输出。没有事后补写的解释；未返回文本时明确显示缺失。
采集期间点“刷新”即可看到新增轨迹。
该服务只读，不连接仿真器，也不占 GPU；快照并非连续运动视频。
页面沿用旧查看器的深色配色。播放采用固定画布、提前解码和后续帧预加载，
新帧就绪前保留当前画面；慢网速时等待加载，不跳过尚未显示的步骤。
列表刷新或筛选不改变当前打开的轨迹，拖动时间轴时只显示最后一次请求的步骤。

## 旧版 ManiSkill 轨迹

本机、只读的 ManiSkill SFT 历史轨迹查看器，实现见
`docs/sft-trajectory-viewer-plan.zh-CN.md`。不连接/控制仿真器，不占 GPU。

## 启动

```bash
cd SFT_data/fronted
python3 server.py            # 默认 http://127.0.0.1:8765 ，索引不存在时自动全量构建
python3 server.py --reindex  # 启动前强制增量重扫
python3 indexer.py           # 只建索引不起服务
```

通过 Inspire Notebook 的 `/proxy/8765/` 地址访问时也可使用；页面资源和 API
会相对于该代理前缀加载。修改静态文件后若仍显示旧页面，请在浏览器强制刷新。

索引与缩略图缓存写在 `SFT_data/viewer/`（`index.sqlite` + `thumbs/`），
可整体删除重建；原始 JSONL 始终是事实来源。

## 功能

- 总览：各教师验收数/目标数、20 个任务覆盖条、失败原因分布、最近采集时间（均由索引实时计算）。
- 列表：教师/任务/状态/seed/工具/turn 数/错误原因筛选 + 预设（成功但很长、重复 IK 失败、token 超限、无 turn 数据）；首/末帧缩略图；默认按最近时间排序。
- 轨迹视图：三栏布局（列表 / 视觉区 / 步骤解释）+ 底部 turn 时间轴（关键事件标记、reward 曲线、播放控制）。
  - `Space` 播放/暂停，`←/→` 逐 turn，`Shift+←/→` 跳关键事件，`1/2/3` 切相机，`B` 切动作前/后。
  - 视觉区支持滚轮缩放、拖拽平移、双击复位、动作前后对比与差分闪烁。
  - 注意：帧为逐 turn 决策前后快照，不是连续物理运动录像。
- 详情：模型决策与理由、工具参数与回执（含 stdout）、reward/官方成功字段、对象与末端位姿变化、SFT 训练样本反查；完整 prompt 与原始 transition JSON 在“审计”卡片按需展开。
- 成功标签只来自验收索引/官方环境回执；未验收、被拒绝、官方成功但未验收的轨迹均明确分色标识，不混入成功数据。

## 结构

| 文件 | 作用 |
| --- | --- |
| `indexer.py` | 扫描 accepted/reports/raw/exported，增量写入 SQLite（按 mtime） |
| `server.py` | Starlette 只读 API + 静态页 + 不透明 id 媒体服务（拒绝目录逃逸） |
| `static/` | 原生 HTML/CSS/JS，无 Node 构建链 |

API：`/api/summary`、`/api/episodes`、`/api/episodes/{id}`、
`/api/episodes/{id}/turns`、`/api/episodes/{id}/turns/{n}`（首次访问时按需解析
model_calls/tool_calls 并缓存对齐结果）、`.../context`、`.../raw`、
`/api/sft/{sample_id}`、`/api/media/{id}[?thumb=1&w=]`、`POST /api/reindex`。
