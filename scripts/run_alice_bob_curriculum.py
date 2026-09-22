"""Run one real Alice -> Bob embodied curriculum round.

The model is loaded once. Alice executes a multi-step trajectory and creates a
candidate PickCube task. The host captures the initial ManiSkill state, restores
it, and runs Bob on the same task. Trusted ``is_obj_placed`` receipts determine
both proposal validity and Bob's score. A JSON feedback record is written in
the adversarial-mle-agents curriculum format.
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


GOAL_PREDICATE = {"type": "is_obj_placed", "source": "maniskill_info"}
GOAL_CHECKER = InfoGoalChecker()


def _flag(info: dict[str, Any], key: str) -> bool:
    value = info.get(key, False)
    if isinstance(value, list):
        return bool(value[0]) if value else False
    return bool(value)


def _step_record(step: int, action: str, reason: str, raw: str, result: Any) -> dict[str, Any]:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    processor, model = _load_model(model_path)
    env = ManiSkillSimulatorAdapter(env_id=env_id, camera_resolution=128)
    try:
        initial_observation = env.reset(seed=seed)
        snapshot = env.capture_snapshot()
        snapshot_path = output_dir / "initial_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")

        alice_steps: list[dict[str, Any]] = []
        observation = initial_observation
        alice_started = time.time()
        for step in range(max_steps):
            action, reason, raw = _decide(processor, model, observation, instruction)
            result = env.step(EnvAction(action_type=action, code=action))
            alice_steps.append(_step_record(step, action, reason, raw, result))
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

        # Use the adversarial curriculum implementation when this workspace is
        # available; keep a JSON fallback so the OpenETA runner remains usable
        # when checked out independently.
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
