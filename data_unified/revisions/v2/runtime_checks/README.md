# MetaWorld / ManiSkill 初态实测

2026-10-10 在本机 GPU 上实际新建环境、reset / 恢复缓存状态并采集 RGB。冻结题单、seed 与缓存引用未改写。`index.json` 是实测结果叠加，不表示全部 3460 个候选初态已验收。

| Benchmark | 尝试题数 | 抽样通过题数 | 成功采集初态 | 代表 case 跨环境复现 |
|---|---:|---:|---:|---:|
| MetaWorld | 20 | 20 | 115 | 20 / 20 |
| ManiSkill | 20 | 16 | 97 | 16 / 16 |

每个用途抽取前两个 case，验证集取其唯一 case。212 个已采集初态均在执行一次零动作后重复 reset / 恢复，物理状态及任务物体/目标状态在 `1e-5` 容差内复现；每个可运行任务的首个 case 还在另一个新建环境中检查。212 个 case 均满足所声明动作维度，初始未成功；抽样内任务状态指纹各不相同，无跨训练/验证/测试用途重复。指纹是数值筛查，不是全面物理等价审计，也没有证明未抽取的 case 独立。

同环境重复重置生成的 242 张相机 PNG 均与首次采集逐字节相同。两种任务的不同初态示例见 [seed-comparison.png](seed-comparison.png)。

## 修复的 MetaWorld 问题

原包装没有传入环境创建 seed；本机 MetaWorld 的 `reset(seed=...)` 又明确忽略该参数，task 选择发生在 reset 之前。因此同一个 reset seed 在两次新建环境中得到不同目标位置。

已在 `sim/env_registry.py` 修复两个路径：构造 `Meta-World/MT1` 时传入 seed，reset 前显式调用底层 `seed()`，再进行任务选择。新协议名为 `metaworld-create-and-preselect-seed.v1`。三个直接 MetaWorld 专家采集/重放脚本也显式传入创建 seed。

**创建 seed 是场景配置的一部分。**本次每个 case 使用 `creation_seed=reset_seed=case.seed` 新建环境。若 RL 复用一个环境而切换 reset seed，使用的是原来创建 seed 生成的有限 task pool，不能默认等于另一个创建 seed 的新环境。执行冻结 case 时必须保持创建配置一致，或显式恢复经过核验的完整状态。

修复改变了新运行的初态协议。旧 MetaWorld 示范/动作不能仅凭相同数字 seed 自动归入新初态；重采集或恢复其真实保存状态，并重新核验动作与成功。旧结果没有改写。

例子：`assembly-v3` 的两个 SFT seed：

- `1337552129`：目标位置 `(0.042620, 0.835503, 0.100000)`。
- `2107042686`：目标位置 `(0.099440, 0.804556, 0.100000)`。

该任务物体起点基本相同，变化发生在目标位置；不能要求每个任务都改变所有物体。

## ManiSkill

本次严格采用清单指定的机器人、控制器、观测模式和 PhysX CUDA。相同 seed 的保证仅适用于相同配置/版本/后端；CPU 和 CUDA 的初始化随机序列不能默认相同。

原生 seed case 确实改变对象/目标或关节参数。例子：`PickCube-v1` 的两个 SFT seed：

- `1981270821`：方块位置 `(0.072003, -0.071693, 0.020000)`。
- `908121213`：方块位置 `(0.019611, 0.006276, 0.020000)`。

LiftPegUpright、PokeCube、PullCube 的缓存 case 根据示范编号和保存状态恢复，不是根据页面上的“恢复前 seed”生成。抽查的 21 个缓存 case 的物理恢复误差最大约 `1.1921e-7`。此检查不包括专家动作重放或 controller 历史恢复：本机 `set_state_dict` 恢复物理状态，不恢复 controller 历史。

以下四题缺资产，未通过运行核查，不等于 seed 失效：

- `AssemblingKits-v1`：objects/kits 资产缺失。
- `TurnFaucet-v1`：PartNet Mobility faucet 资产组不全。
- `OpenCabinetDoor-v1`：PartNet Mobility cabinet 数据缺失。
- `OpenCabinetDrawer-v1`：所需 cabinet/drawer 资产不全。

曾尝试官方 AssemblingKits 下载命令及 HTTPS 连接；官方存储端点 TLS 连接以 `UNEXPECTED_EOF_WHILE_READING` 失败。没有降低 TLS 校验或将这些题记为通过。

## 文件与重跑

`<bench>/<task>/results.json` 保存每个 case 的 seed、选择器、物理/任务状态、重复误差、跨环境结果和截图引用；`probe.log` 保存独立进程日志。各用途目录保存真实截图，`repeat/` 和 `fresh/` 是复现检查证据。`index.json` 汇总覆盖与错误，查看器优先显示对应 case 的真实截图。未抽取的 case 仍显示未检查。

```bash
# 默认每个用途最多两个初态；按 benchmark 独立进程运行。
python data_unified/probe_initial_states.py --benches metaworld --limit-per-phase 2
python data_unified/probe_initial_states.py --benches maniskill --limit-per-phase 2

# 重新汇总已保存结果，不启动环境。
python data_unified/probe_initial_states.py --summarize

# 每个候选都检查（更昂贵；仍需先补齐资产）。
python data_unified/probe_initial_states.py --benches metaworld maniskill --limit-per-phase 0
```

未来训练与评测必须接入同一创建/reset/缓存恢复协议。当前通过的是这些实测初态的可复现性与差异检查，不是完整训练、示范重放或所有候选初态的最终验收。
