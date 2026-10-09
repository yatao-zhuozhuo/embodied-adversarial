# OpenETA 轨迹可视化

统一入口复用 `SFT_data/fronted/static/two_tool.html` 的深色三栏布局、固定画布和逐步播放逻辑。服务读取已保存的结果，不启动仿真器、不使用 GPU。

在仓库根目录启动：

```bash
.venv/bin/python visual/server.py --host 0.0.0.0 --port 8780
```

浏览器打开 `http://127.0.0.1:8780/`；Notebook 可通过 `/proxy/8780/` 访问。页面显示当前仓库分支及 baseline worktree 分支，无需切换分支。

- **Baseline 跑分**：自动寻找 `feat/baseline` 的 worktree，扫描 `baseline/results`。同一 attempt 优先读取 `result.json`，尚未完成时读取 `progress.json`，忽略重复的 `final.json`。可筛选运行批次、benchmark、模型和成功状态。成功率按当前筛选范围内有布尔判定的 attempt 计算，不将不同运行批次自动合并成模型榜单；`success` 与 `native_success_observed` 保留原记录的不同含义。
- **SFT 构造**：读取 `SFT_data/qwen_live_v1` 和 `SFT_data/qwen_xml_v1` 下的 `*/*/attempt-*/episode.json`，以及 `qwen_xml_v1/audit/*/*/attempt-*.json` 审计记录（已删除的图片无法恢复，仍可查看原始输出和动作），包括成功与失败的构造过程。多相机快照、模型 reasoning/content、动作、回执均可逐步查看。左侧“旧版 ManiSkill SFT 构造轨迹”打开挂载于 `/sft-history/` 的原完整查看器，保留历史 accepted/rejected、教师、任务、训练样本反查等功能。
- **训练 Rollout**：默认读取 `runs/formal_experiment2/**/rollouts/*/*.json`，显示 Alice/Bob、round、终止原因和最终 rollout reward。曲线使用每步 `environment_reward`；最终训练 reward 独立显示，不混用两个指标。保存的 `image_hashes[0]` 对应初始图像，后续 hash 对应动作后的图像。

支持搜索、动作前/后切换、时间轴、点击奖励曲线跳转、播放速度、空格播放与左右方向键。手动刷新或启用 30 秒自动刷新可以读取新增结果；首次读取 baseline 的大 JSON 文件可能需要一分钟以上；后台索引会显示文件进度，短请求轮询避免代理超时，后续按文件修改时间缓存。请求失败会显示明确错误和重试提示。缺失图像明确标注，动作与文本仍可查看；正在写入或损坏的 JSON 会提示数量，刷新后重试。

其他数据目录可以在启动时指定：

```bash
.venv/bin/python visual/server.py --port 8780 \
  --baseline-root /path/to/baseline/results \
  --sft-root /path/to/SFT_data \
  --training-root /path/to/runs/another_experiment
```

新查看器不修改原始数据。旧版 SFT 页面继续使用原来的 SQLite/缩略图索引；缺少索引时首次启动会构建。API 返回的精简详情省略重复模型请求历史与 base64 图像，原始文件路径保留在元数据中。图像服务仅允许访问配置的数据根目录内的图像文件。

验证：

```bash
.venv/bin/python -m unittest visual.test_server -v
node --check visual/static/app.js
```

如果使用后台启动，可把日志和 PID 放在已忽略的 `visual/.runtime/` 内。

## Artificial 专家合成轨迹

“Artificial 专家轨迹”入口扫描 `SFT_data/expert_replay_v1/episodes/*/*/seed-*/episode.json`。展示官方成功判定、专家构造 XML、逐步相机图和 `samples.jsonl` 中的实际训练输入。独立 `reasoning*/episodes/.../annotations/turn-*.json` 的补写会核对源样本 hash 后显示，并注明模型和事后补写来源；刷新会更新逐轮补写进度。原始 XML 保持可查看。专家回放目前未保存某些最终动作后的图像，页面按缺帧显示。
