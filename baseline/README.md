# 四模型具身 baseline

本目录实现 DeepSeek Flash、Qwen3.8-27B、Qwen3.5-9B、Qwen3.5-4B 的 **RGB 图像＋原生动作** 评测。四模型共用 40 道题：LIBERO、Meta-World、ManiSkill3、RoboCasa 各 10 题，每题一次，共 160 条 episode。题目见 [all_40.json](manifests/all_40.json)，协议见[评测计划](evaluation_plan.zh-CN.md)。本仓库没有附带正式评测成绩。

模型只能调用 `observe_env` 和 `step_env`；动作维度分别为 7、4、7、12，每次重复 1–10 个物理步。创建、固定 seed 重置和关闭由评测程序管理。成功只取环境原生判据，正奖励或模型自行声明完成不会得分。工具集不包含外部感知模型、抓取估计、代码执行或技能编辑。

## 文件布局

- `runtime/`：显式模型参数、原生动作评测与 DeepSeek 共享存储转发。
- `manifests/`：固定任务、seed、候选顺序及题单校验和。
- `snapshots/native-actions-v1/`：统一提示词、通用技能、动作定义和冻结文件哈希。
- `configs/`：模型服务地址、环境版本、依赖锁与资产哈希。
- `scripts/`：推理服务、环境检查、冻结及任务提交入口。
- `sdk/`：使用本地 `openapi-sdk-go` 的备选提交客户端。

`results/`、`logs/`、`reports/`、`jobs/`、`.runtime/` 和初始化原始证据均为本地产物，不提交 Git。模型权重、仿真资产和虚拟环境也不在此目录的版本管理范围内。

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
