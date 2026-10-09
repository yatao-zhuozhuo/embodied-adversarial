# SFT 数据采集

## Flash 扩展采集 / 100 条成功目标

后续调度更新：当前用 `--concurrency 8` 从共享任务池最多并发 8 题（脚本默认 4），不再每个 bench 固定占一条队列。
`--retry-rounds 2` 表示首轮完成后仅筛失败题追加两轮（每题最多三遍），一旦成功不再重跑。
各轮有全局屏障，不让快任务提前进入下一轮；累计 100 个成功任务仍停止调度。
`--adopt-existing` 可接管旧调度器留下的在途任务，接管前必须停止旧调度器调度但保留其任务子进程。
新增同 bench 并发的导出文件锁；每条任务保留独立端口、episode、图片与日志。

`scripts/run_flash_expanded_sft.py --output-root SFT_data/gemini_flash_expanded_<时间戳>`
使用 Flash 默认 thinking，共享指定数量的执行位置。冻结完整任务池及两份协议文档 hash，
排除现有测试 holdout、文档点名任务、旧对照清单及已有采集 ledger 的题，合并重复任务别名。
首轮每题一次，失败最多额外两次，每次使用独立 attempt/seed；同一轮 API 可以重试，不重复执行动作。
只计本批次环境确认且通过逐轮 SFT 导出审计的成功轨迹；总目标 100，调度预留在途名额避免超采。
旧重复轨迹不计入。三轮耗尽不足 100 时明确报不足；API 重试耗尽或鉴权失败暂停新题，
避免接口问题耗尽整个任务池。成功/失败图片均保留，`campaign.json` 记录逐轮调度、进度及终态。
默认 60 轮/400 控制步、8192 输出预算。缺资产或相机/控制契约不匹配的任务记录失败，不改环境验收。
当前安装版本的 `PickClutterYCB-v1.evaluate()` 恒返回 false，排除并记录原因；
新增 AssemblingKits 和 TurnFaucet 的固定 Panda wristcam 子 profile，运行时仍严格校验实际图片。

Gemini Flash/Pro 三遍对照使用 `--attempts-per-task 3`：每模型 20 题各三遍，合计 120 次，
默认 thinking、8192 输出预算和统一输入不变。Flash 已有首遍计入总数，成功也继续补足三遍。
续跑保持原 `--stamp`（共享输出和 ledger），用 `--report-tag repeat3` 隔离日志，
`--wait-for-report <原报告目录>` 等待对应 bench 原进程结束后再补采，避免两个进程写同一轨迹目录。
Pro 使用独立进程与 MCP 端口；全部轨迹及图片保留，同题同遍使用一致 seed。

## CodingRouter GPT 单模型试跑

设置环境变量 `CODINGROUTER_API_KEY`，运行
`uv run python scripts/run_gemini_sft_comparison.py --provider codingrouter --models gpt-5.6-luna`。
使用网关 `/v1/chat/completions`，模型名是网关标识，不保证底层映射；完整保存 API 响应和用量。
不指定 thinking/reasoning 参数，沿用提供方默认值。多模态 XML 预检通过才启动原 20 个任务各一次，
四 bench 并行，60 轮/400 控制步、8192 输出预算、统一输入和图片、成功/失败全保留。
目录为 `codingrouter_input_v1_gpt-5.6-luna_<UTC时间戳>/`；仅官方成功回执轨迹导出 SFT。

## Gemini 默认 thinking 对照

`scripts/run_gemini_two_tool_sft.sh` 使用 `CODINGROUTER_API_KEY` 请求 CodingRouter 的原生
`/v1beta/models/{model}:generateContent`；密钥仅通过请求头发送，不进入 URL 或轨迹文件。
`gemini-3.6-flash` 与 `gemini-3.1-pro-preview` 各跑原 20 个任务一次，分别保存在
`gemini_input_v1_<model>_<UTC时间戳>/`，成功与失败轨迹及图片均保留。
使用统一输入、严格 XML、60 轮/400 控制步和 8192 输出 token 预算。
不发送 `thinkingConfig`，使用模型默认 thinking；`enable_thinking` 记为 null，
`thinking_mode` 为 `provider_default`，不能把无思考文本或缺失用量误认成关闭 thinking。
适配器只把统一消息映射为 `systemInstruction`、user/model `contents` 和有序 PNG `inlineData`，
不添加原生函数工具、不修复答案。保存完整原生响应（含签名、模型版本和 `usageMetadata`），
另保存兼容的决策响应和原生请求 hash；思考文本不用于动作执行或 SFT 标签。
8192 为请求的 `maxOutputTokens`，不是指定 thinking budget。截断/阻断/原生工具调用不执行。

启动入口：设置 `CODINGROUTER_API_KEY` 后执行 `uv run python scripts/run_gemini_sft_comparison.py`。
入口先用已保存的真实多模态首轮输入验证两个模型（只请求，不执行，单列为预检），
均通过才启动四个 bench 队列，每队列先后运行两个模型，最多四个模拟器并发。
预检失败不启动任务；报告位于 `SFT_data/reports/gemini_comparison_<UTC时间戳>/`。
同一 `--stamp` 可续跑已经完成的任务清单；已有轨迹不会覆盖。API 重试仅重发同一轮模型请求。

## 新输入协议 / DeepSeek 关闭 thinking 对照

按 `docs/trajectory-input-unification.zh-CN.md` 接入 `agent/runtime/two_tool_input.py`，
采集使用 `--unified-input --no-enable-thinking`，输出独立保存到 `deepseek_input_v1_nothink_<UTC时间戳>/`。
同一张 20 任务清单，各跑 3 次，共 60 次；每 bench 使用
`--task-limit 5 --attempt-limit 15 --max-attempts-per-task 3 --target-per-task 3 --retain-failures --continue-on-infrastructure-error`。
输出预算仍为 8192，XML 格式与成功验收不变；禁用的是 DeepSeek 隐藏 thinking，XML `<reasoning>` 仍属于回答。

新请求固定四组文本历史，不重放历史隐藏思考，不携带旧消息里的图片；按 profile 附上上一决策前主图及当前视角。
MetaWorld 去重；LIBERO 包含 wrist；ManiSkill 使用已预检的有/无 hand_camera 子 profile（原生 128 手腕不放大）；
RoboCasa 包含左右主图和 wrist，请求最多五图。图片逐张标明相机、时刻和动作关联，去重保留全部时刻引用。
控制契约从运行环境提取，记录原生坐标系、旋转表示、尺度和夹爪方向；未知的底盘/torso 尺度显式标 unknown。
每轮白名单输入包含剩余预算、真实请求/执行步数、本体状态及实测差分，不发送 reward、对象真值或会话 handle。
FMBAssembly1Easy 和 PickSingleYCB 的描述根据安装环境任务类的任务定义做等价表述，不添加动作流程。

每轮保存 `input_package`、输入消息/hash、有序图片/hash、provider wire hash、API 返回的真实 token usage。
DeepSeek 本地没有对应完整视觉 tokenizer，预请求 token 数标 null，不用字符数冒充 token 数；
四组文本/图片上限固定，provider 输入超限会显式记录错误，不静默截断。该项不声称完成跨模型精确预算验收。
导出逐轮复用实际发送的条件上下文并验证图片 hash；只监督当前回答。旧 60 条不重写、不混入新批次。
统一查看器增加“本轮模型实际输入”，与宿主动作后图片及评分分开。

20 任务的真实 reset/observe/单步控制预检在 `reports/input_v1_preflight/`，属于宿主测试，不是教师轨迹。
本次修改接入 SFT 采集/导出；共享构造器供其他链路接入，不据此宣称 baseline/GRPO 训练已完成全部文档验收。

## DeepSeek 20 个任务诊断采集：保留失败轨迹和图片

2026-10-06 追加要求：同一批 20 个任务，每个任务累计尝试 3 次，共 60 次。
第一遍保留原样；先修复 RoboCasa 相机花屏并验证，再在同一输出根目录补跑两遍。
续跑使用 `--task-limit 5 --attempt-limit 10 --max-attempts-per-task 3 --target-per-task 3`，
同时保留 `--retain-failures --continue-on-infrastructure-error`。已有 ledger 防止重复跑第一遍，
每个任务的 `attempt-0000/0001/0002` 分别保存三次尝试，后两次使用递增 seed。
即使中途成功，也继续补齐三次；每个 bench 最多新增 10 次，不启用无限重启。

RoboCasa 花屏根因：reset 遗留的旧渲染器延迟析构，在新 EGL 上下文中释放了旧编号的 GPU 资源，
导致新渲染器读出花屏或其他相机画面。`sim/envs/robocasa/rendering.py` 将清理绑定到所属上下文，
并恢复此前上下文；RoboCasa worker 的仿真调用也固定在同一线程，避免 EGL 跨线程占用。
不修改历史图片、不替换相机、不添加用于掩盖问题的额外环境动作。
复现及修复后检查帧在 `reports/robocasa_camera_debug/`，检查工具为 `scripts/check_robocasa_camera_frames.py`。
新轨迹的 `source_sha256` 记录渲染修复代码版本。真实 RGB-D 垃圾回收回归测试可用
`OPENETA_TEST_ROBOCASA_RENDER=1 .venv/bin/python -m pytest -q tests/test_robocasa_render_cleanup.py` 运行。

### 第一遍配置（历史记录）

按用户要求，每个 bench 选择 5 个不同任务、每个任务尝试 1 次，共 20 次。
任务表为 `configs/deepseek_xml_debug5.v1.json`；输出使用独立目录 `deepseek_xml_debug5_<UTC时间戳>/`。
启动参数为 `--task-limit 5 --attempt-limit 5 --max-attempts-per-task 1 --target-per-task 1 --retain-failures --continue-on-infrastructure-error`。
thinking 开启、输出上限 8192、每任务最多 60 轮/400 控制步、模型接口最多请求 5 次均不变。
成功、失败、截断、格式错误以及 API 错误的 `episode.json` 和已获取的所有相机帧均保留，不搬走或删除失败目录。
基础设施错误记录后进入下一任务；手动中断仍停止该路。环境创建失败时如无图像，保留错误说明，不生成替代图片。
RoboCasa 的花屏原图也保留作为排查证据，本轮不自动过滤或修复。
成功样本单独导出到本轮目录的 `exports/`；失败记录不会进入 SFT 训练集。
统一查看器 `/proxy/8780/` 的 SFT 页面可查看本轮所有成功、失败和运行中的记录，按 `deepseek_xml_debug5_` 分组区分。
本轮不启用每小时自动重启，避免跑超 20 次。

## 当前试采：DeepSeek Flash（deepseek_xml_v1）

2026-10-06 按用户要求停止 Qwen 四路采集，并移除其每小时自动重启任务；旧 Qwen 证据保留，不混入 DeepSeek 数据。
使用 DeepSeek 视觉文档指定的 `deepseek-flash`，端点 `https://api.deepseek.com/chat/completions`。
图像仍以真实 RGB 的 base64 `image_url` 传入；请求使用 `thinking:{type:enabled}`、`reasoning_effort:high`、`max_tokens:8192`。
不同于 Qwen，不传 `chat_template_kwargs`；密钥只读取 `DEEPSEEK_API_KEY`，不会回退使用 Qwen 密钥。
其余 XML 校验、两工具执行、原始 thinking/answer 留档、成功判定、重试和 SFT loss 策略保持不变。

当前仅用 `configs/deepseek_xml_canary.v1.json` 的四个任务，各尝试一次，最多 60 轮/400 控制步；没有自动扩量或自动重启。
轨迹目录 `deepseek_xml_v1/`，schema 为 `openeta.deepseek_live_two_tool_xml.v1`，动作来源为 `deepseek_api`。
成功样本自动导出到 `exported/deepseek_xml_<bench>/train.v1.jsonl`；失败只保留 `deepseek_xml_v1/audit/` 审计，不计入训练。

```bash
OPENETA_TWO_TOOL_MANIFEST=SFT_data/configs/deepseek_xml_canary.v1.json \
OPENETA_TWO_TOOL_MCP_PORT=8766 \
bash scripts/run_deepseek_two_tool_sft.sh --bench metaworld \
  --attempt-limit 1 --target-per-task 1 --max-turns 60 --max-control-steps 400

python scripts/status_two_tool_sft.py --root SFT_data/deepseek_xml_v1
```

调用前在环境中设置 `DEEPSEEK_API_KEY`，不要写入仓库或命令日志。
统一查看器 `/proxy/8780/` 同时区分 Qwen/DeepSeek 数据目录；两工具查看器 `/proxy/8770/` 默认读取 DeepSeek，
查看历史 Qwen 可设置 `OPENETA_TWO_TOOL_DATASET=qwen_xml_v1` 后启动。
参考：[图像理解](https://api-docs.deepseek.com/zh-cn/guides/vision/)、[思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode/)。

## Qwen 真实 XML 两工具轨迹（qwen_xml_v1）

仅使用 Qwen3.8-27B 在 OpenETA 的 MCP 环境中实时调用 `observe_env` 和 `step_env`。
宿主提供初始 RGB 和本体状态，模型逐轮直接输出 XML `<decision>`；不使用 API 原生 tool_calls。
协议为 `openeta.two_tool_xml.v1`，宿主复用共享解析器，按创建环境返回的维度、上下界和剩余控制步预算校验。
数组必须使用 `<action type="array"><item type="number">…</item></action>`。
宿主拥有创建、重置、初始观察、关闭和验收能力；`task_complete` 不是工具，也不意味着成功。

不调用专家策略或重放示范，不进行事后 think 补写。保留 provider 的原始
`content`、`reasoning_content`、完整 provider 响应、解析结果及每轮实际请求上下文。
provider 请求中的 base64 图片落盘为本地文件引用，原始工具回执中的图片保留文件引用和 SHA256。
默认请求服务端开启 thinking，单次输出上限为 8192 tokens；服务是否返回推理以原始响应为准。
续跑使用 `--enable-thinking --max-tokens 8192`；若模型未返回解释文本，保持缺失，不生成替代内容。

输出：
- `qwen_xml_v1/<bench>/<task>/attempt-*/episode.json`：成功及运行中轨迹。
- `qwen_xml_v1/<bench>.attempts.jsonl`：尝试结果摘要。
- `qwen_xml_v1/audit/`：失败尝试的原始文本、请求和回执审计；失败图片删除且标记缺失，不导出训练。
- `exported/qwen_xml_<bench>/train.v1.jsonl`：成功轨迹的逐决策 Swift `messages` + `images` 数据，assistant.content 为原始最终 XML。
- `reports/qwen_xml_v1/`：运行日志和 Swift 编码验收记录。
- 查看器 `/proxy/8770/`：仅显示成功轨迹，同时展示原始 thinking 和 XML。

正式严格校验不修复 XML。共享无效策略：无效输出消耗一轮、不执行；连续三次无效结束。
连续 observe 超过三次视为无效；截断回答不执行。实际控制步数只取 MCP `executed_control_steps`。
奖励、成功标签、objects/goal 坐标不进入模型反馈；原始评分证据单独保存在宿主记录。
固定 256×256、最多两张模型输入图像、五轮历史窗口；记录 prompt/观测版本、revision 和源码 hash。

### Swift 监督约束

使用 `configs/two_tool_xml_sft.v1.json` 配置：`--loss_scale last_round+ignore_empty_think`，历史 assistant 带 `loss:false`，仅当前 XML 为目标。
学生模板自动插入的空 think 标签也屏蔽 loss，不将模板占位符当成老师推理。
thinking 原文保留在 episode，不补写；此版 SFT 采用 answer-only 监督，`<reasoning>` 仍为模型原文。
图像按消息顺序变为 `<image>` 占位符与 `images` 列表；不提供 tools 字段，避免 agent template 改写动作。
本机 Swift 4.4.2 / Qwen3.5-4B 学生模板使用以下命令验证实际 token labels 和图像顺序；更换学生必须重新验收。

```bash
/opt/openeta-swift/bin/python scripts/validate_two_tool_xml_sft.py
```

无 dataset 时使用独立编码测试夹具，不计入采集；报告保存渲染 prompt、input_ids 和 labels。

```bash
OPENETA_TWO_TOOL_MANIFEST=SFT_data/manifests/qwen_live_canary.v1.json \
OPENETA_TWO_TOOL_MCP_PORT=8766 \
bash scripts/run_two_tool_sft_collection.sh --bench metaworld \
  --attempt-limit 1 --target-per-task 1 --max-turns 40 --max-control-steps 200
```

采集需要环境变量 `INF_API_KEY` 或 `OPENETA_LLM_API_KEY`，不得将密钥写入仓库。
其他 benchmark 使用独立 MCP 端口。正式任务使用各 benchmark 的 manifest；测试任务及保留 seed 不变。
每条成功轨迹自动触发本 benchmark 导出，无独立补标或定时导出进程。
模型接口的 408/429/500/502/503/504、连接中断及超时会重试同一轮、同一观察的请求：默认总共最多 5 次，单次 socket 超时 300 秒，等待 10/20/40/60 秒。
可用 `--model-max-attempts`、`--model-timeout`、`--model-retry-backoff` 调整；MCP 的 `--timeout` 独立保留为 180 秒。
重试期间不调用环境动作、不消耗模型决策轮数；每次请求的耗时、错误和等待时间实时写入 `model_calls[].attempts`。
401/403 等永久错误、无效 XML、MCP 动作请求不使用网络重试；预算耗尽仍停止该路，避免无限请求或重复动作。
客户端加长超时不能消除服务端网关的 504；已退出并关闭环境的旧尝试不会伪装成断点恢复。

### 每小时自动巡检

**已于 2026-10-06 停用 Qwen 定时任务。以下为保留的操作说明，不表示当前仍在运行。**

`scripts/watch_two_tool_sft.py` 检查四路采集，异常停止且仍有任务预算时，以现有 XML/thinking/8192/接口重试配置重新启动。
本机 crontab 条目标记为 `openeta-qwen-xml-hourly-watchdog`；每小时执行一次，不依赖聊天窗口保持打开。
巡检使用文件锁；已运行、正在启动、配额完成或尝试预算耗尽的任务不会重复启动。
端口被其他进程占用、数据损坏等情况只记录异常，不误杀其他服务；本巡检不主动终止仍存活但卡住的进程。
认证环境保存在仓库外的 `~/.config/openeta/qwen_xml_watchdog.env.json`，权限为 0600，不输出到日志。
`reports/qwen_xml_v1/watchdog/latest.json` 为最近状态，`checks.jsonl` 保存检查历史，`*.collection.log` 保存自动重启后的采集日志。
暂停自动重启：使用 `crontab -e` 删除上述标记对应的一行；这不会停止已运行的采集。
机器/容器关闭时本机 cron 无法检查，也不会自动向聊天发送通知；容器重启后需确认 cron 服务已启动。
按用户要求清理此前 `qwen_live_v1` 原生 tool-call 轨迹及对应导出，不作历史 XML 转换。
下方旧版 GLM/Qwen ManiSkill 数据是独立历史资产，不清理。本次只验收 SFT，不代表 baseline/GRPO 已全部迁移。

## 旧版 ManiSkill SFT 数据采集

目标：GLM-5.3-w8a8c8 与 Qwen3.8-27B 各采集 500 条可信成功轨迹；1000 条全部作为 SFT 训练数据，不划分验证集或测试集。

详细设计见 `ManiSkill_500_Trajectory_SFT_Plan.zh-CN.md`。

## 已生成资产

- `configs/task_catalog.v1.json`：20 个任务及每模型配额。
- `manifests/glm.train.500.v1.json`：GLM 500 条目标 manifest。
- `manifests/qwen.train.500.v1.json`：Qwen 500 条目标 manifest。
- `configs/provider.*.example.env`：不含密钥的 provider 示例。

## 启动前

在启动采集的同一个 shell 中注入密钥，不要把密钥写入仓库：

```bash
read -rsp 'INF_API_KEY: ' INF_API_KEY
export INF_API_KEY
```

确认 GPU 0 的既有任务已释放：

```bash
nvidia-smi
```

## Canary

先顺序运行两个 5-episode PickCube canary：

```bash
./scripts/run_maniskill_sft_canary.sh glm
./scripts/run_maniskill_sft_canary.sh qwen
```

调试时可以先只跑 1 条，并保持 thinking 开启、提高输出预算：

```bash
OPENETA_CANARY_COUNT=1 \
OPENETA_LLM_ENABLE_THINKING=true \
OPENETA_LLM_MAX_TOKENS=16384 \
./scripts/run_maniskill_sft_canary.sh glm
```

正式采集默认关闭 thinking：实测 GLM 开启 thinking 后单轮会超过 180 秒，
且 4096 tokens 曾在输出 `<decision>` 前被截断。关闭 thinking 后仍要求模型
输出完整的 `<reasoning>` 与工具调用 XML，并显著提高 1000 条采集的可行性。

脚本默认拒绝在 GPU 利用率高于 25% 时启动。不要用 `OPENETA_ALLOW_BUSY_GPU=true` 绕过，除非已经确认资源不会互相影响。

## 正式采集

Canary 验收后顺序运行两个可续跑采集器：

```bash
./scripts/run_maniskill_sft_collection.sh glm
./scripts/run_maniskill_sft_collection.sh qwen
```

默认参数：

- 单卡 ManiSkill worker pool：1
- environment concurrency：1
- provider concurrency：1
- wave size：5
- 每模型最大尝试数：2500

稳定后可通过环境变量把 environment/provider concurrency 提升到 2，再观察 GPU/Vulkan、429 和 cleanup 指标。

采集器会持续扫描 raw bundle，只把以下 episode 计入配额并导出：

- rollout bundle 完整且 artifact hash 正确；
- ManiSkill 官方 `success` 为 true；
- 无人工或 guidance 介入；
- teacher model 与 shard 一致；
- 主 planner decision 已通过 host validator。

感知通道按服务能力分开配置：Qwen 端点接收 `image_url`，使用原始
`base_camera` 图像，同时读取 ManiSkill MCP 的结构化对象状态；GLM 端点
明确是非多模态服务，因此不发送图像，只使用同一结构化状态。本数据版本
训练 planner/tool use，不宣称是纯视觉策略数据。两个 teacher 都独立规划
和调用动作工具，不互相评分或提供感知结果。

输出位置：

```text
raw/glm/                         GLM 原始 rollout
raw/qwen/                        Qwen 原始 rollout
accepted/glm/episodes.v1.jsonl   GLM 接收清单
accepted/qwen/episodes.v1.jsonl  Qwen 接收清单
exported/glm/train.v1.jsonl      GLM SFT 数据
exported/qwen/train.v1.jsonl     Qwen SFT 数据
reports/                         每波结果、导出报告和 MCP 日志
```

## 最终审计

```bash
uv run python scripts/audit_maniskill_sft.py \
  --dataset SFT_data/exported/glm/train.v1.jsonl \
  --accepted-index SFT_data/accepted/glm/episodes.v1.jsonl \
  --catalog SFT_data/configs/task_catalog.v1.json \
  --teacher-model GLM-5.3-w8a8c8 \
  --require-complete-quota

uv run python scripts/audit_maniskill_sft.py \
  --dataset SFT_data/exported/qwen/train.v1.jsonl \
  --accepted-index SFT_data/accepted/qwen/episodes.v1.jsonl \
  --catalog SFT_data/configs/task_catalog.v1.json \
  --teacher-model Qwen3.8-27B \
  --require-complete-quota
```
