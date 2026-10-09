"""运行一轮真实的 Alice -> Bob 具身课程实验。

模型只加载一次。Alice 先执行多步轨迹，尝试构造一个 PickCube 任务；然后 host
保存/恢复 ManiSkill 的初始状态，让 Bob 从同一个状态开始执行。环境返回的
``is_obj_placed`` 真值用于判断 Alice 的提案和 Bob 是否成功。最后将结果写成
adversarial-mle-agents 课程模块可以读取的 JSON 文件。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from adapter.protocol import EnvAction
from agent.runtime.embodied_goal import InfoGoalChecker
from qwen_embodied_rollout import _decide, _load_model


# 当前这个 runner 使用 ManiSkill 的官方成功字段作为目标判定。
# 更完整的 cube_at_position 目标编译逻辑位于 agent/runtime/embodied_task_compiler.py。
GOAL_PREDICATE = {"type": "is_obj_placed", "source": "maniskill_info"}
# 目标检查器只读取环境返回的可信状态，不相信模型自己声称“成功”。
GOAL_CHECKER = InfoGoalChecker()


def _flag(info: dict[str, Any], key: str) -> bool:
    """从环境 info 中读取布尔字段，并兼容 ManiSkill 的单元素列表格式。"""
    value = info.get(key, False)
    if isinstance(value, list):
        return bool(value[0]) if value else False
    return bool(value)


def _step_record(step: int, action: str, reason: str, raw: str, result: Any) -> dict[str, Any]:
    """把一次环境交互整理成可写入 JSON 的轨迹记录。"""
    # result.info 来自模拟器；GOAL_CHECKER 在这里计算本步的目标分数。
    evaluation = GOAL_CHECKER.evaluate(state=result.info, predicate=GOAL_PREDICATE)
    return {
        "step": step,
        "action": action,
        "reason": reason,
        "raw": raw,
        "reward": result.reward,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "success": _flag(result.info, "success"),
        "is_obj_placed": evaluation.success,
        "goal_score": evaluation.score,
        "is_grasped": _flag(result.info, "is_grasped"),
        "info": result.info,
    }


def run_round(
    *,
    model_path: str,
    output_dir: Path,
    env_id: str,
    instruction: str,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    """运行一轮 Alice -> Bob，并生成任务、分数和课程反馈 artifact。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    # Alice 和 Bob 共用同一个已加载模型；role 通过 prompt/rollout 上下文区分。
    processor, model = _load_model(model_path)
    env = ManiSkillSimulatorAdapter(
        env_id=env_id,
        camera_resolution=128,
        snapshot_dir=output_dir / "snapshots",
    )
    try:
        # 先创建并重置 ManiSkill 环境，然后保存 Alice 开始前的完整状态。
        # snapshot 不只是随机种子，还包含模拟器状态、步数和夹爪 latch。
        initial_observation = env.reset(seed=seed)
        snapshot = env.capture_snapshot()
        snapshot_path = output_dir / "initial_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")

        # ------------------------- Alice 阶段 -------------------------
        # Alice 看到初始 observation 和 instruction，通过真实动作改变环境。
        alice_steps: list[dict[str, Any]] = []
        observation = initial_observation
        alice_started = time.time()
        for step in range(max_steps):
            action, reason, raw = _decide(processor, model, observation, instruction)
            # 将模型输出的离散动作交给适配器，再由适配器转换为 ManiSkill 原生动作。
            result = env.step(EnvAction(action_type=action, code=action))
            alice_steps.append(_step_record(step, action, reason, raw, result))
            # 环境结束，或方块已经被放到目标处，就停止 Alice rollout。
            if result.terminated or result.truncated or alice_steps[-1]["is_obj_placed"]:
                break
            observation = result.observation
        alice_success = bool(alice_steps and alice_steps[-1]["is_obj_placed"])
        alice_trajectory = {
            "agent": "alice",
            "env_id": env_id,
            "instruction": instruction,
            "seed": seed,
            "goal_predicate": GOAL_PREDICATE,
            "success": alice_success,
            "steps": alice_steps,
            "duration_s": time.time() - alice_started,
        }
        (output_dir / "alice_trajectory.json").write_text(
            json.dumps(alice_trajectory, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ------------------------- Bob 阶段 -------------------------
        # 恢复 Alice 之前保存的状态，确保 Bob 和 Alice 的物理初态一致。
        # 这里的 replay_equal 是一个简单检查：恢复后的关节位置是否与初始位置相同。
        env.restore_snapshot(snapshot)
        replay_observation = env.observe()
        replay_equal = (
            replay_observation.robot.joint_positions
            == initial_observation.robot.joint_positions
        )
        bob_steps: list[dict[str, Any]] = []
        observation = replay_observation
        bob_started = time.time()
        for step in range(max_steps):
            action, reason, raw = _decide(processor, model, observation, instruction)
            result = env.step(EnvAction(action_type=action, code=action))
            bob_steps.append(_step_record(step, action, reason, raw, result))
            # Bob 的成功同样由环境真值判断，而不是由模型生成的文本决定。
            if result.terminated or result.truncated or bob_steps[-1]["is_obj_placed"]:
                break
            observation = result.observation
        bob_success = bool(bob_steps and bob_steps[-1]["is_obj_placed"])
        bob_trajectory = {
            "agent": "bob",
            "env_id": env_id,
            "instruction": instruction,
            "seed": seed,
            "initial_snapshot_id": snapshot.snapshot_id,
            "success": bob_success,
            "steps": bob_steps,
            "duration_s": time.time() - bob_started,
        }
        (output_dir / "bob_trajectory.json").write_text(
            json.dumps(bob_trajectory, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ------------------------- 任务 artifact -------------------------
        # task.json 描述本轮任务以及 Alice 轨迹的引用，供其它模块消费。
        task = {
            "task_id": f"maniskill-{env_id.lower()}-seed{seed}",
            "env_id": f"openeta/maniskill_{env_id}-v0",
            "seed": seed,
            "initial_state_ref": snapshot.state_uri,
            "initial_state_sha256": snapshot.state_sha256,
            "instruction": instruction,
            "goal_predicate": GOAL_PREDICATE,
            "alice_trajectory_ref": "alice_trajectory.json",
            "alice_valid": alice_success,
            "max_steps": max_steps,
        }
        (output_dir / "task.json").write_text(
            json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 当前脚本只运行 1 次 Bob，因此 success_rate 实际上是 0.0 或 1.0。
        # 完整实验通常会对同一 snapshot 采样 K 次，再计算聚合成功率。
        score = {
            "task_id": task["task_id"],
            "success_rate": float(bob_success),
            "episodes": 1,
            "success": bob_success,
            "alice_valid": alice_success,
            "snapshot_replay_equal": replay_equal,
        }
        (output_dir / "bob_scores.json").write_text(
            json.dumps(score, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 如果同一工作区中存在 adversarial-mle-agents，就调用它计算 Alice 的课程奖励。
        # 如果该仓库未检出，则写入一个简单 fallback，保证 OpenETA runner 仍可运行。
        feedback: dict[str, Any]
        adversarial_src = Path(__file__).resolve().parents[2] / "adversarial-mle-agents" / "src"
        if adversarial_src.is_dir():
            sys.path.insert(0, str(adversarial_src))
            from mle_agent_rl.embodied_curriculum import BobTaskScore, alice_task_reward
            from mle_agent_rl.embodied_task import EmbodiedTaskSpec

            spec = EmbodiedTaskSpec(
                task_id=task["task_id"],
                env_id=task["env_id"],
                seed=seed,
                initial_state_ref=snapshot.state_uri,
                instruction=instruction,
                goal_predicate=task["goal_predicate"],
                alice_trajectory_ref=str(output_dir / "alice_trajectory.json"),
                feasibility_score=float(alice_success),
                novelty_score=1.0,
                difficulty_target=0.45,
                max_steps=max_steps,
            )
            bob_score = BobTaskScore(spec.task_id, float(bob_success), 1)
            reward = alice_task_reward(spec, bob_score)
            feedback = {
                "task": spec.to_dict(),
                "bob_score": bob_score.__dict__,
                "alice_reward": reward,
                "validity_gate": alice_success,
                "source": "mle_agent_rl.embodied_curriculum",
            }
        else:
            feedback = {
                "task_id": task["task_id"],
                "bob_score": score,
                "alice_reward": 0.0 if alice_success else -1.0,
                "validity_gate": alice_success,
                "source": "fallback",
            }
        (output_dir / "curriculum_feedback.json").write_text(
            json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 将公开的课程交接文件复制到 adversarial-mle-agents，
        # 这样它的下一轮可以通过 JSON 读取结果，而不需要导入 OpenETA。
        if adversarial_src.is_dir():
            curriculum_dir = (
                adversarial_src.parent
                / "runs"
                / "embodied"
                / task["task_id"]
            )
            curriculum_dir.mkdir(parents=True, exist_ok=True)
            for name in ("task.json", "bob_scores.json", "curriculum_feedback.json"):
                (curriculum_dir / name).write_text(
                    (output_dir / name).read_text(encoding="utf-8"), encoding="utf-8"
                )
            feedback["adversarial_curriculum_dir"] = str(curriculum_dir)
            (output_dir / "curriculum_feedback.json").write_text(
                json.dumps(feedback, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return {
            "task": task,
            "alice": alice_trajectory,
            "bob": bob_trajectory,
            "score": score,
            "feedback": feedback,
            "output_dir": str(output_dir),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--instruction", default="Pick up the cube and place it in the target area.")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = run_round(
        model_path=args.model,
        output_dir=args.output_dir,
        env_id=args.env_id,
        instruction=args.instruction,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print(json.dumps({
        "task_id": summary["task"]["task_id"],
        "alice_valid": summary["task"]["alice_valid"],
        "bob_success": summary["score"]["success"],
        "snapshot_replay_equal": summary["score"]["snapshot_replay_equal"],
        "alice_reward": summary["feedback"]["alice_reward"],
        "output_dir": summary["output_dir"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
