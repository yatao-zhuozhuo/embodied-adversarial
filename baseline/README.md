## 2026-10-06：统一输入 v1（baseline 接入）

本分支使用 `openeta.two_tool_input.v1` 输入和 `openeta.two_tool_xml.v1` 输出。目标规范是主工作区的 `docs/trajectory-input-unification.zh-CN.md`。共享构造器位于 `agent/runtime/two_tool_input.py`；本次只接入 baseline，尚未声称 SFT/GRPO 或 Swift loss 已完成接入。

修复：XML kind/name 首尾空白在校验和执行中统一；PushT 从实际 panda_stick 控制器导出 6 维无夹爪契约；ManiSkill 读取 TCP link pose 而非 robot root pose，并将 Sapien wxyz 转为 xyzw。解析拒绝与执行拒绝分开统计；传输失败保留执行结果未知，不能按零步重发。

每个任务使用 `baseline/configs/input_profiles.json` 中固定的相机与控制器能力。主图原生 256；ManiSkill 原生手腕 128。顺序为上一决策前主图、当前主图、当前手腕；RoboCasa 上限 5 图。解码像素相同时合并附件并保留所有时刻引用。ManiSkill 当前适配器的占位标定不作为可信标定发送，相关字段为 null。

保留最近四组回答和真实执行反馈；历史隐藏 thinking 不重放。每轮提供实际步数、测量的 TCP 位移和夹爪变化，缺测为 null。只允许白名单观测字段，reward、隐藏评分、对象真值和宿主 handle 不进入模型输入。当前仍关闭 thinking、temperature=0、生成预算 1024，以免把输入修复和生成策略变更混为同一实验。

上下文预算调用同一 vLLM 服务的 `/tokenize`，包含实际多模态消息及模板；先删最老完整文本组，再删纯历史图。必要当前图、任务和反馈不可删，最小输入超限则报错。服务需用新版 `serve_vllm.sh` 启动（允许 5 图）；不支持准确多模态计数的 provider 不会静默使用估算。

新快照：`baseline/snapshots/two-tool-input-v1/`。旧 `two-tool-xml-v1` 快照与 `inspire-20261005-xml-final` 结果保留，禁止在旧输出目录续跑新输入。新正式运行使用独立 campaign 名称，并先通过输入 profile 和模型 canary 检查。

每轮 `attempt-N/inputs/NNN.json` 保存规范输入；轨迹中的 `exchange.attempts[].request_body` 保存实际 provider 请求。查看单轮模型可见输入：

```bash
python baseline/scripts/render_input_audit.py <inputs/NNN.json> <output.html>
```

验证记录：`baseline/results/input-v1-preflight/` 是 40 个环境的实际能力、图片和动作回执；`baseline/results/input-v1-canary/` 是四环境真实模型多轮请求验收。两者都是验收证据，不是新一轮正式成功率。

---

# 当前运行协议

新评测使用 `openeta.two_tool_xml.v1`，严格 XML `<decision>` 与 `<action type="array"><item type="number">…</item></action>`。`num_steps` 必填，观察/结束参数为空；拒绝 JSON 数组字符串、多 decision、重复字段和 XML 修复。共享接口见 `agent/runtime/two_tool_xml.py`。规范输出带 type 属性；解析保留原 planner 对无 type 属性数值叶子的类型推断（如 `<num_steps>3</num_steps>`），但仍严格检查整数、有限数值和 XML item 数组，不接受 JSON 数组字符串。本次只迁移 baseline；SFT/GRPO 尚未接入该接口。

新快照在 `baseline/snapshots/two-tool-xml-v1`，旧快照仅作版本历史，不能用于新评测。旧协议 campaign 已按用户要求删除。环境预检和模型资源保留。

三个 Qwen 的并行入口为 `baseline/scripts/run_xml_parallel_campaign.py --campaign NAME`，使用独立 MCP 端口 8200/8201/8202；三个推理服务分别使用 GPU 0/1/2，仿真使用 GPU 3。先完成每模型四环境 XML canary，全部通过才开始正式评测。每条仍为 100 轮、3600 秒；无效输出消耗轮次但不执行，控制步预算默认为 `max_turns*10`，同时受环境原生 horizon 限制。轨迹独立保存 XML、校验、MCP 请求/回执、前后帧及实际执行步数。

# 具身 baseline

本目录实现 DeepSeek Flash、Qwen3.8-27B、Qwen3.5-9B、Qwen3.5-4B 的 **RGB 图像＋原生动作** 评测。四模型共用 40 道题：LIBERO、Meta-World、ManiSkill3、RoboCasa 各 10 题，每题一次，共 160 条 episode。题目见 [all_40.json](manifests/all_40.json)，协议见[评测计划](evaluation_plan.zh-CN.md)。本仓库没有附带正式评测成绩。

模型只能调用 `observe_env` 和 `step_env`；动作维度分别为 7、4、7、12，每次重复 1–10 个物理步。创建、固定 seed 重置和关闭由评测程序管理。成功只取环境原生判据，正奖励或模型自行声明完成不会得分。工具集不包含外部感知模型、抓取估计、代码执行或技能编辑。

## 文件布局

- `runtime/`：显式模型参数、原生动作评测与 DeepSeek 共享存储转发。
- `manifests/`：固定任务、seed、候选顺序及题单校验和。
- `snapshots/two-tool-xml-v1/`：统一提示词、通用技能、动作定义和冻结文件哈希。
- `configs/`：模型服务地址、环境版本、依赖锁与资产哈希。
- `scripts/`：推理服务、环境检查、冻结及任务提交入口。
- `sdk/`：使用本地 `openapi-sdk-go` 的备选提交客户端。

`results/`、`logs/`、`reports/`、`jobs/`、`.runtime/` 和初始化原始证据均为本地产物，不提交 Git。模型权重、仿真资产和虚拟环境也不在此目录的版本管理范围内。

## Inspire 部署

独立分支工作目录：`/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA/.worktrees/baseline`。
所有命令从这个目录执行。模型、独立 Swift 环境、协调器、API overlay 和本地渲染库分别放在 `baseline/.runtime/models`、`swift`、`coordinator`、`api-compat`、`mesa` 下；仿真环境放在此 checkout 的 `sim/venvs/` 下。评测输出使用 `baseline/results/<campaign>`。

`serve_vllm.sh` 默认读取此 checkout 内的模型和 Swift 环境，可用 `MODEL_ROOT`、`SWIFT_VENV` 覆盖；`environment.sh` 默认使用 GPU 0，可用 `OPENETA_WORKER_GPUS` 覆盖。原平台的任务提交脚本不适用于 Inspire；不要执行 `submit_brainctl.py` 或 `submit_sdk_jobs.py`。

迁移完成后，先通过四环境预检和模型 canary，再运行 `freeze_protocol.py` 更新迁移后的快照。正式 campaign 开始后保持代码、题单和快照不变。DeepSeek 凭据由环境变量或未跟踪的 `baseline/.runtime/secrets/deepseek.env` 提供。

三个本地模型服务就绪并启动两轮四环境预检后，可启动受检查约束的评测入口：

```bash
baseline/.runtime/coordinator/bin/python baseline/scripts/run_inspire_campaign.py --campaign inspire-YYYYMMDD
```

它等待 80 份环境检查通过，验证重复性、冻结题单与协议，检查四环境 MCP 链路，并要求三个 Qwen 模型都通过图像测试和两轮原生动作 canary，之后才依次运行 120 条正式 episode。任何失败会停止，状态写入 campaign 的 `status.json`。DeepSeek 接口需单独证明支持图像后再纳入；当前平台提供的 `dsv4-flash-0731` 拒绝图像输入。

## 运行环境

当前共享存储部署根目录为 `/mnt/shared-storage-user/self-evolve/embodied-adversarial`，模型目录为 `/mnt/shared-storage-user/self-evolve/model`。脚本中的绝对路径用于该部署；迁移机器时需要调整挂载与解释器路径。

Qwen 使用 [formal5](../scripts/run_embodied_swift_staged_formal5.sh) 对应的共享环境，通过 vLLM 提供服务；BF16、TP=1、32768 上下文，默认端口为 8101/8102/8103。API 兼容 overlay 依赖见 `configs/requirements-api-overlay.txt`。下载模型使用 `scripts/download_models.py`，来源为 ModelScope。

LIBERO、Meta-World、RoboCasa 使用 `sim/venvs/` 下的隔离环境；ManiSkill3 复用 formal5 环境。控制服务使用 `.runtime/coordinator` 中的 MCP 1.x。版本锁保留当前安装状态，不是跨平台一键安装器。仿真资源下载入口为 `scripts/download_robocasa_assets.py`，源码和资产版本见 `configs/environment_versions.json`、`configs/assets-lock.json`。

ManiSkill 明确为 **ManiSkill3 3.0.1**，不是 ManiSkill2。当前部署使用 Mesa lavapipe 软件 Vulkan；其他环境使用 MuJoCo EGL。环境变量见 `scripts/environment.sh`。新机器必须通过实际渲染与原生判据验收后才能开始正式评测。

## 预检与运行

在配置好的 GPU 机器、仓库根目录执行：

```bash
baseline/.runtime/coordinator/bin/python baseline/scripts/start_simulator.py
bash baseline/scripts/probe_all.sh libero metaworld maniskill3 robocasa
python3 baseline/scripts/freeze_questions.py
python3 baseline/scripts/freeze_protocol.py

# 本地 vLLM 已就绪时，运行一个模型的固定题单。
PYTHONPATH="$PWD" baseline/.runtime/coordinator/bin/python \
  baseline/scripts/launch_model.py --model qwen35_4b \
  --root baseline/results/my-campaign/formal
```

`prepare_questions.py` 会重新选题；不要用它覆盖已经开始评测的题单。`freeze_questions.py` 需要先完成两轮环境预检；会把原始初始化证据留在被忽略的本地文件中。`freeze_protocol.py` 生成新的协议哈希；一批正式任务运行期间不得修改冻结文件。

模型参数固定为 temperature=0、max_tokens=1024、关闭 thinking、无跨模型 fallback。DeepSeek 密钥从 `DEEPSEEK_API_KEY` 读取；不得写入配置或提交 Git。无公网出口的任务可使用 `runtime/relay.py`，由有公网访问能力的主机读取已有凭据文件，处理共享存储中的固定模型请求。

## 提交任务

`prepare_sdk_jobs.py` 生成请求文件；`submit_brainctl.py` 使用已有平台登录提交同等资源的 RJob。每次使用新的 campaign 名称，生成的文件和平台回执保存在被忽略的 `jobs/` 中。

```bash
python3 baseline/scripts/prepare_sdk_jobs.py --campaign my-baseline-run
python3 baseline/scripts/submit_brainctl.py --campaign my-baseline-run --prepare-only
baseline/.runtime/coordinator/bin/python baseline/scripts/submit_brainctl.py \
  --campaign my-baseline-run
python3 baseline/scripts/submit_brainctl.py --campaign my-baseline-run --inspect
```

当前模板使用项目 `fdj-infra`、配额组 `fdj_gpu_task`，每模型一个副本，1 GPU、16 CPU、96 GiB 内存。运行上限 48 小时、排队上限 24 小时；容器不自动重启，平台 `backoffLimit` 使用允许的最小值 1。资源型号和是否能及时调度由平台决定。迁移账号时也需调整提交脚本中的 creator 与镜像拉取凭据引用。

任务入口 `run_job.sh` 会先验证快照、四环境原生判据以及两轮真实模型动作，再启动正式题单。验收不通过时停止，canary 结果不计入正式成绩。

Go SDK 备选方式：在 `baseline/sdk` 中用 Go 1.21+ 编译到 `../.runtime/baseline-submit`，随后使用 `submit_sdk_jobs.py --campaign NAME --credentials /existing/yicloud.env`。SDK 通过 `go.mod` 的 replace 引用工作区旁的 `openapi-sdk-go`，需要 `YICLOUD_PUBLIC_KEY` / `YICLOUD_SECRET_KEY`。同一批任务只选择一种提交方式，避免重复运行。

## 验证

```bash
baseline/.runtime/coordinator/bin/python -m pytest -q \
  tests/test_baseline_native_protocol.py tests/test_baseline_libero_seed.py \
  tests/test_baseline_inference_smoke.py tests/test_bench_worker_terminal_result.py \
  tests/test_parallel_episode_harness.py tests/test_planner_xml_decision.py
```

单元测试覆盖模型参数、XML、动作边界、原生成功判定、图像投影和 seed 传递；不能替代真实环境与模型的端到端验收。
