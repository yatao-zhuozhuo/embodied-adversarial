# RoboCasa 跨进程重置一致性修复

问题：单模型 GRPO 从 checkpoint-5 恢复后，首组 `ScrubCuttingBoard`（seed 20321400）在环境初始化阶段失败。模型恢复已完成，失败发生在采样前的同组初始状态校验。

原因：RoboCasa 1.0.1 的 `Counter.get_reset_regions` 先用 `set` 去重 XML 元素，再按集合遍历顺序给台面区域分配 `geom_N`。XML 元素的对象地址影响顺序，因此同一个 RNG seed 在不同进程里可能选择不同几何区域。原始复现实验中，机器人初始位置差约 2.2 米；固定 PYTHONHASHSEED 仍不能消除差异。

修复位于 RL 工作树 `sim/envs/robocasa/direct_env.py`：在官方 Counter 方法返回后，按区域 offset 和 size 排序，再稳定分配采样索引。保留全部原始区域及其几何数据，沿用原生 RNG 和成功判定。补丁在环境构造前安装，且重复安装无副作用；未知区域标识会显式报错。一致性校验继续执行。

验证输出目录：`/inspire/hdd/project/exploration-topic/ky26060/outputs/OpenETA/single_model_rl_32g32_20261006/reset_diagnosis`。

- 修复前四个独立进程显示不同 qpos/qvel/ctrl/求解器状态。
- 修复后四个独立进程（不同 hash seed）导出的完整 MuJoCo 状态逐字段完全一致。
- 相关测试 30 项通过，1 项需要特殊运行环境的条件测试跳过。
- 五个 RoboCasa 训练任务通过真实 MCP 验证；其余四题各 8 个独立环境，ScrubCuttingBoard 使用 32 个环境；完整结果写入 `group_acceptance.json`。

续训继续使用原来的 32 道题、group 32、每卡 batch 4、5 个 epoch，总共 160 次更新，从 checkpoint-5 恢复。模型输入与 XML 协议沿用原实现；TensorBoard 沿用 `openeta-rl-32g32-20261006`。
