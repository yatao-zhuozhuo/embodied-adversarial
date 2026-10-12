# 统一 SFT / RL / 评测题单 v3

按 2026-10-10 的要求建立：四个 benchmark 各 **15 个共同题**，各额外预留 **5 个泛化题**，共同题各保留 **1 个独立验证初态**。共 60 个共同题、20 个泛化题。`validation` 用于调参和选 checkpoint；所有模型共享 `eval_seen` 与 `eval_generalization`，最终测试集不用来选 checkpoint、改提示词或挑 seed。历史快照保存在 `revisions/v1/` 和 `revisions/v2/`；当前格式 schema 仍为 v1，划分版本 `split_revision=3`。v3 根据用户要求替换四个缺资产的 ManiSkill 题，记录见 `replacement-record.v3.json`。

当前完成的是题目选择、缓存引用索引、初态候选划分和静态一致性检查。`runtime_ready=false`：没有启动新的采集、训练或评测，也没有把静态状态去重当作环境可复现验收。

| Benchmark | 共同题 | 泛化题 | 有官方初态缓存的共同题 |
|---|---:|---:|---:|
| LIBERO | 15 | 5 | 15 |
| MetaWorld | 15 | 5 | 0；15 题有官方脚本专家 |
| ManiSkill | 15 | 5 | 3 |
| RoboCasa | 15 | 5 | 15 |
| 合计 | 60 | 20 | 33 |

每个共同题暂定 SFT 15 个初态、RL 15 个其他初态、验证 1 个初态、同任务评测 20 个初态；泛化题各评测 20 个初态。合计 900 个 SFT 候选、900 个 RL 初态、60 个验证初态、1200 个同任务测试初态、400 个泛化测试初态，共 3460 个 case。数量在 `selection.v1.json` 配置，不是训练轨迹数或决策样本数；同一初态反复 rollout 不增加独立状态数。RL 后续扩充训练初态时仍须避开测试状态。

缓存参考初态有 1668 个 case；其余 1792 个是尚未捕获实际状态的 native seed 候选。没有复制 HDF5、轨迹、图片或环境资产。

## 文件与复现

- `selection.v1.json`：人工选题、技能类别、选择理由、初态数量；共同题在 `shared`，泛化题在 `generalization`。
- `catalogs/`：本机环境注册表快照、版本、LIBERO BDDL/初态/示范路径、RoboCasa 示范注册信息。注册 ID 数量不能直接当作独立有效任务数。
- `task_catalog.json`：统一任务身份、说明、机器人、预期动作维度、horizon、缓存覆盖与已知历史使用情况。
- `inventory/`：每题缓存初态索引；只保存位置、编号、指纹和校验信息。
- `initial_states.jsonl`：唯一的 case 总表，含用途、seed 与显式初态选择器。
- `source_refs.jsonl`：case 对应的缓存文件、内部状态位置、示范编号和状态/XML 校验信息；不包含专家动作。
- `manifests/sft_train.json`、`rl_train.json`、`validation.json`、`eval_seen.json`、`eval_generalization.json`：从同一总表生成的用途清单。
- `audit.json`：覆盖、历史源计划保护、旧 holdout 重合、排除题目和待完成事项。
- `checksums.json`：输入快照与生成文件校验值。

在仓库根目录执行：

```bash
# 只有需要更新本机环境元数据时才运行；更新后须重建并重新审阅划分。
python data_unified/discover.py

# 复用已冻结 inventory，重建用途清单。
python data_unified/build.py

# 检查文件校验、任务覆盖、用途隔离、引用一致性和历史示范保护。
python data_unified/check.py
python -m unittest data_unified.test_split -v
```

`build.py` 需要 NumPy、h5py；`check.py` 只依赖标准库。新建 inventory 会读取缓存中的初态，可能受 HDD 速度影响。重建默认复用 `catalogs/historical_evidence.json` 的历史快照，避免正在进行的旧采集改变冻结结果；要更新历史覆盖可显式使用 `--refresh-history`。重建不会自动重新扫描已有 inventory；更换缓存、版本或环境协议时要显式更新相应 inventory 并发布新的划分版本。旧运行始终保留原始划分。

## 初态协议

`native_seed` 的 seed 用于环境创建和 reset，必须补上真实初态记录、任务相关变化检查和可复现性检查，才能确认划分有效。不同整数 seed 并不保证不同状态。

`reference_demo` 的权威选择器是缓存示范编号及保存的状态/XML。case 的 seed 是恢复前的 seed，标记为 `seed_role=pre_restore_seed`，不能代替缓存初态编号。ManiSkill 另外记录缓存的 `episode_seed`；底层恢复可能使用它。SFT、RL、验证、测试使用不同初态，且同一缓存状态等价组不会跨用途。新增验证不改变原来的 SFT、RL 与测试缓存分配；有额外缓存时取未使用的独立初态，否则生成新的 native seed 候选。LIBERO 每题现有 50 个缓存初态已全部分配，15 个验证初态因此采用待捕获和核验的 native seed，不从测试集中挪用。

缓存按保存状态在 1e-8 舍入下筛查重复；MuJoCo 向量忽略仿真时钟，另外记录 XML 校验值。这是静态重复筛查，不是经过验证的物理等价判定。不同指纹仍可能只改变无关量；必须核查任务物体、目标、初始成功状态和有效场景。

LIBERO 共同题采用已安装示范的保存初态，因此不假定 `demo_index` 等于官方 `.pruned_init` 编号。LIBERO 泛化题保留 native seed 候选；后续可以显式接入官方初态编号。MetaWorld 从脚本专家在指定训练 seed 上新采集；不直接复制旧 seed 的动作到新初态。ManiSkill 缓存三题要求匹配 `pd_ee_delta_pose` 与 PhysX CUDA。RoboCasa 全部使用 `pretrain`，缓存 case 从各自 episode 的 XML、状态和语言恢复；多语言任务的 case 保留对应指令。

评测只能向模型提供约定的观测与指令；缓存专家动作、未来帧、特权状态只供宿主侧恢复/审计，不进入模型输入。所有模型在同一 case 上使用相同接口、horizon、观察规则和控制预算；官方成功判据决定成绩。缓存测试初态来自官方示范分布，应据此描述结论，不能称作整个原生随机任务分布的无偏抽样。

## 选题与历史隔离

选题兼顾技能覆盖与已安装缓存，没有按模型成功率筛题。LIBERO 覆盖相对位置、物体选择、柜门/微波炉、多物体搬运与放置；MetaWorld 覆盖抓取、推、旋转、插入、工具和扫动；ManiSkill 覆盖操作、装配、叠放与工具；RoboCasa 覆盖厨房原子与组合操作。

泛化预留题，每个 benchmark 5 题：

- LIBERO：`libero_spatial_task1`、`libero_object_task6`、`libero_spatial_task5`、`libero_object_task8`、`libero_goal_task1`。
- MetaWorld：`drawer-open-v3`、`stick-push-v3`、`coffee-button-v3`、`pick-out-of-hole-v3`、`reach-wall-v3`。
- ManiSkill：`PlugCharger-v1`、`PushT-v1`、`RotateSingleObjectInHandLevel3-v1`、`InsertFlower-v1`、`SO100GraspCube-v1`。
- RoboCasa：`OpenDrawer`、`PickPlaceCounterToCabinet`、`OpenCabinet`、`TurnOffStove`、`LoadDishwasher`。

它们在已检查的实际 SFT、RL 任务池和成功采集记录中没有训练覆盖；这项检查不涵盖全部历史 checkpoint、外部数据或未知预训练。ManiSkill 的手中旋转使用 Allegro 灵巧手、插花使用浮动 Ability Hand，SO100 抓取使用 SO100。这些题按任务配置 / 机器人迁移单列；手中旋转 Level3 的随机旋转轴、对象与难度不同，不能把注册 ID 变化解释成完全未见的任务家族。

任务身份归并 MetaWorld suite 别名与 RoboCasa `pretrain/target` 别名，防止同一任务换注册 ID 后进入泛化集。泛化指未训练任务，不意味着未见基本技能或完全未见任务家族。预留题可能有历史评测使用，相关证据保存在任务表中。

现有专家源计划引用过的示范编号及其重复状态组被保守放到训练侧，验证与测试不使用这些编号。这保护已知示范来源，不证明历史 RL 或 live seed 训练状态与新测试完全分离；旧 checkpoint 做严格实验前仍需审计真实训练状态。建议用本划分重新训练并比较 Base、SFT、SFT+RL。

为达到 ManiSkill 15 个共同题，新协议将部分旧 holdout 题纳入训练；具体列表在 `audit.json`。旧 `SFT_data/configs/two_tool_test_holdout.v1.json` 和历史成绩保持原样，旧采集器不能直接沿用它来过滤新题单。

## 接入前待完成

2026-10-10 已进行 MetaWorld / ManiSkill 实际抽样检查，结果和真实截图见 [runtime_checks/README.md](runtime_checks/README.md)。MetaWorld 创建/reset seed 问题已修复；原冻结清单保留不变，实测通过不等于全部候选都已验收。

清单使用独立 schema，不是可直接传给旧采集器的启动配置。执行适配器必须读取 `initial_state`，不能忽略它只传 seed。

1. LIBERO 当前主分支直接 reset 忽略 seed；共同题必须显式恢复缓存状态，native seed 路径需修复。
2. MetaWorld 已统一创建与 reset 的 seed，并记录抽查初态真实参数；后续执行必须使用同一创建配置。仍需确认 15 个脚本策略在对应训练初态上的官方成功。
3. ManiSkill 的两工具 camera subprofile 当前只支持部分任务，需要扩展；核验资产、原生机器人、动作维度、对象模型范围与重配置行为。
4. `PickClutterYCB-v1` 被排除：本机 `evaluate()` 始终返回失败。以 `PegInsertionSide-v1` 替换。泛化 `PushT-v1` 使用 Panda stick、原生 6 维动作，Panda 夹爪题为 7 维动作，绘图任务使用 Panda stick 的 6 维笛卡尔动作，Allegro 手中旋转为 16 维关节目标，浮动手插花为 12 维关节目标，SO100 抓取题为 6 维原生关节动作，不能套用旧的每 benchmark 固定 7 维假设。
5. 逐题验证缓存恢复与新 seed 的可复现性、跨用途状态隔离和初始未成功；通过后生成可执行 manifest。源示范成功不等于恢复/重放成功。

6. 四个替代题已接入原生默认控制器、相机子配置与动作契约；灵巧手保留完整关节观测，旋转轴作为任务指令提供。绘图题按原生 300/500 步上限执行。SO100 使用 `pd_joint_target_delta_pos` 与 `rgb+segmentation`，不能套用 Panda 夹爪的动作维度与模式。

验证集每个共同题仅 1 个初态，适合轻量调参和选 checkpoint；最终成绩仍使用冻结的独立测试集。
