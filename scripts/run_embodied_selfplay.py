#!/usr/bin/env python3
"""Run resumable Alice/Bob embodied asymmetric self-play rounds."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
ADVERSARIAL_SRC = WORKSPACE_ROOT / "adversarial-mle-agents" / "src"
for source_dir in (REPO_ROOT, ADVERSARIAL_SRC):
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from adapter.protocol import EnvAction
from agent.runtime.embodied_task_compiler import CompiledTask, compile_cube_position_task
from agent.training.embodied_grpo import PolicyEpisode, PolicyStep, parse_action, receipt_flag
from mle_agent_rl.embodied_curriculum import BobTaskScore, alice_task_reward
from mle_agent_rl.embodied_task import EmbodiedTaskSpec
from scripts.train_embodied_grpo import (
    _generate_action,
    _prompt,
    _rollout_episode,
    _train_group,
    load_optimizer_state,
    load_lora_policy,
    save_optimizer_state,
    supervised_action_update,
)


ALICE_INSTRUCTION = (
    "Create a new feasible task by moving the red cube to a meaningfully different reachable "
    "position. Use real actions and finish with DONE when the target state is established."
)
BOB_INSTRUCTION = "Move the red cube to the green target marker."


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _plain_step(step: PolicyStep) -> dict[str, Any]:
    return step.artifact() | {"prompt": step.prompt}


def _model_alice_episode(
    model: Any,
    processor: Any,
    env: ManiSkillSimulatorAdapter,
    snapshot: Any,
    *,
    device: torch.device,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    oracle_mix: float,
) -> PolicyEpisode:
    env.restore_snapshot(snapshot)
    observation = env.observe()
    steps: list[PolicyStep] = []
    grasp_attempts = 0
    for step_index in range(max_steps):
        prompt = _prompt("alice", ALICE_INSTRUCTION, observation, step_index)
        image = np.asarray(observation.cameras[0].rgb, dtype=np.uint8)
        teacher_action = (
            _oracle_action(env, observation, grasp_attempts) if oracle_mix > 0.0 else None
        )
        teacher_forced = teacher_action is not None and random.random() < oracle_mix
        if teacher_forced:
            action = teacher_action
            completion = action
            completion_ids = processor.tokenizer.encode(completion, add_special_tokens=False)
            parsed = parse_action(completion)
        else:
            completion_ids, completion = _generate_action(
                model,
                processor,
                observation,
                prompt,
                device=device,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            parsed = parse_action(completion)
        grasp_attempts += int(parsed.action == "GRASP")
        result = env.step(EnvAction(action_type=parsed.action, code=parsed.action))
        steps.append(PolicyStep(
            prompt=prompt,
            image=image,
            completion_ids=completion_ids,
            completion=completion,
            action=parsed.action,
            valid_action=parsed.valid,
            environment_reward=float(result.reward),
            success=receipt_flag(result.info, "success"),
            is_grasped=receipt_flag(result.info, "is_grasped"),
            is_obj_placed=receipt_flag(result.info, "is_obj_placed"),
            terminated=bool(result.terminated),
            truncated=bool(result.truncated),
            teacher_forced=teacher_forced,
            teacher_action=teacher_action,
        ))
        observation = result.observation
        if result.terminated or result.truncated or parsed.action == "DONE":
            break
    return PolicyEpisode(role="alice", reward=0.0, success=False, steps=steps)


def _oracle_action(env: ManiSkillSimulatorAdapter, observation: Any, grasp_attempts: int) -> str:
    state = env.task_state()
    cube = state["cube_position"]
    goal = state["goal_position"]
    tcp = observation.robot.end_effector_pose["xyz"]
    grasped = bool(state["is_grasped"])
    threshold = 0.004

    def axis_action(delta: float, positive: str, negative: str) -> str | None:
        if abs(delta) <= threshold:
            return None
        base = positive if delta > 0 else negative
        if abs(delta) <= 0.025:
            return f"{base}_FINE"
        return base

    def horizontal_target(target: list[float], *, tolerance: float = threshold) -> str | None:
        for delta, positive, negative in (
            (target[0] - tcp[0], "MOVE_X_POS", "MOVE_X_NEG"),
            (target[1] - tcp[1], "MOVE_Y_POS", "MOVE_Y_NEG"),
        ):
            if abs(delta) <= tolerance:
                continue
            base = positive if delta > 0 else negative
            action = f"{base}_FINE" if abs(delta) <= 0.025 else base
            if action is not None:
                return action
        return None

    if not grasped:
        # Align from above before descending; this avoids pushing the cube along
        # the table.  The official PickCube grasp pose is at the cube centre.
        action = horizontal_target(cube)
        if action is not None:
            return action
        action = axis_action(cube[2] - tcp[2], "MOVE_Z_POS", "MOVE_Z_NEG")
        if action is not None:
            return action
        return "GRASP" if grasp_attempts < 8 else "MOVE_Z_POS_FINE"

    safe_z = max(float(goal[2]), 0.16)
    horizontal_error = max(abs(goal[0] - cube[0]), abs(goal[1] - cube[1]))
    # Lift before a large lateral transport.  Once aligned, tolerate small
    # controller cross-axis drift while descending instead of repeatedly
    # returning to the safe height.
    if horizontal_error > 0.03:
        action = axis_action(safe_z - tcp[2], "MOVE_Z_POS", "MOVE_Z_NEG")
        if action is not None:
            return action
        horizontal_action = horizontal_target(goal, tolerance=0.012)
        if horizontal_action is not None:
            return horizontal_action
    elif cube[2] >= safe_z - 0.02 and horizontal_error > 0.012:
        horizontal_action = horizontal_target(goal, tolerance=0.012)
        if horizontal_action is not None:
            return horizontal_action
    action = axis_action(goal[2] - cube[2], "MOVE_Z_POS", "MOVE_Z_NEG")
    if action is not None:
        return action
    return "DONE"


def _oracle_alice_episode(
    processor: Any,
    env: ManiSkillSimulatorAdapter,
    snapshot: Any,
    *,
    max_steps: int,
) -> PolicyEpisode:
    env.restore_snapshot(snapshot)
    observation = env.observe()
    steps: list[PolicyStep] = []
    grasp_attempts = 0
    for step_index in range(max_steps):
        action = _oracle_action(env, observation, grasp_attempts)
        if action == "GRASP":
            grasp_attempts += 1
        prompt = _prompt("alice", ALICE_INSTRUCTION, observation, step_index)
        completion = action
        completion_ids = processor.tokenizer.encode(completion, add_special_tokens=False)
        image = np.asarray(observation.cameras[0].rgb, dtype=np.uint8)
        result = env.step(EnvAction(action_type=action, code=action))
        steps.append(PolicyStep(
            prompt=prompt,
            image=image,
            completion_ids=completion_ids,
            completion=completion,
            action=action,
            valid_action=True,
            environment_reward=float(result.reward),
            success=receipt_flag(result.info, "success"),
            is_grasped=receipt_flag(result.info, "is_grasped"),
            is_obj_placed=receipt_flag(result.info, "is_obj_placed"),
            terminated=bool(result.terminated),
            truncated=bool(result.truncated),
        ))
        observation = result.observation
        if result.terminated or result.truncated or action == "DONE":
            break
    return PolicyEpisode(role="alice", reward=0.0, success=False, steps=steps)


def _replay_and_compile(
    env: ManiSkillSimulatorAdapter,
    snapshot: Any,
    initial_state: dict[str, Any],
    episode: PolicyEpisode,
    *,
    min_displacement: float,
    tolerance: float,
) -> tuple[CompiledTask, dict[str, Any], bool]:
    generated_final = env.task_state()
    env.restore_snapshot(snapshot)
    for step in episode.steps:
        env.step(EnvAction(action_type=step.action, code=step.action))
    replay_final = env.task_state()
    replay_equal = _distance(
        generated_final["cube_position"], replay_final["cube_position"]
    ) <= 0.005
    compiled = compile_cube_position_task(
        snapshot=snapshot,
        initial_state=initial_state,
        final_state=replay_final,
        min_displacement=min_displacement,
        tolerance=tolerance,
    )
    if not replay_equal:
        compiled = replace(compiled, valid=False, reason="Alice trajectory replay mismatch")
    return compiled, replay_final, replay_equal


def _load_optimizer_if_present(path: Path, optimizer: torch.optim.Optimizer) -> None:
    if not path.is_file():
        return
    load_optimizer_state(path, optimizer)


def _holdout_evaluation(
    model: Any,
    processor: Any,
    env: ManiSkillSimulatorAdapter,
    seeds: list[int],
    *,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        env.reset(seed=seed)
        snapshot = env.capture_snapshot()
        episode = _rollout_episode(
            model,
            processor,
            env,
            snapshot,
            role="bob",
            instruction="Pick up the red cube and place it at the green target marker.",
            max_steps=int(config["selfplay"]["holdout_max_steps"]),
            max_new_tokens=int(config["model_input"]["max_new_tokens"]),
            temperature=float(config["grpo"].get("bob_temperature", config["grpo"]["temperature"])),
            top_p=float(config["grpo"].get("bob_top_p", config["grpo"]["top_p"])),
            device=device,
        )
        episodes.append({"seed": seed, **episode.artifact()})
    return {
        "episodes": episodes,
        "success_rate": sum(bool(item["success"]) for item in episodes) / max(1, len(episodes)),
        "reward_mean": float(np.mean([item["reward"] for item in episodes])),
    }


def run(config: dict[str, Any], *, output_dir: Path, oracle_alice: bool, round_limit: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config)
    state_path = output_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.is_file() else {
        "version": config["version"],
        "next_round": 0,
        "seen_goal_cells": [],
        "alice_adapter": config.get("initial_alice_adapter"),
        "bob_adapter": config.get("initial_bob_adapter"),
        "rounds": [],
    }
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    grpo = config["grpo"]
    model_input = config["model_input"]
    environment = config["environment"]
    selfplay = config["selfplay"]
    alice_device = torch.device("cuda", int(config["devices"]["alice"]))
    bob_device = torch.device("cuda", int(config["devices"]["bob"]))
    processor = AutoProcessor.from_pretrained(config["model"])
    processor.image_processor.min_pixels = int(model_input["min_image_pixels"])
    processor.image_processor.max_pixels = int(model_input["max_image_pixels"])

    alice_model = None
    alice_optimizer = None
    if not oracle_alice:
        alice_model = load_lora_policy(
            config["model"],
            device=alice_device,
            lora_rank=int(grpo["lora_rank"]),
            lora_alpha=int(grpo["lora_alpha"]),
            adapter_path=Path(state["alice_adapter"]) if state["alice_adapter"] else None,
        )
        alice_optimizer = torch.optim.AdamW(
            (parameter for parameter in alice_model.parameters() if parameter.requires_grad),
            lr=float(grpo["learning_rate"]),
        )
        _load_optimizer_if_present(output_dir / "alice_optimizer.pt", alice_optimizer)

    bob_model = load_lora_policy(
        config["model"],
        device=bob_device,
        lora_rank=int(grpo["lora_rank"]),
        lora_alpha=int(grpo["lora_alpha"]),
        adapter_path=Path(state["bob_adapter"]) if state["bob_adapter"] else None,
    )
    bob_optimizer = torch.optim.AdamW(
        (parameter for parameter in bob_model.parameters() if parameter.requires_grad),
        lr=float(grpo["learning_rate"]),
    )
    _load_optimizer_if_present(output_dir / "bob_optimizer.pt", bob_optimizer)

    env = ManiSkillSimulatorAdapter(
        env_id=environment["env_id"],
        camera_resolution=int(environment["camera_resolution"]),
        translation_step_m=float(environment["translation_step_m"]),
        fine_translation_step_m=float(environment["fine_translation_step_m"]),
        max_episode_steps=int(environment["episode_horizon"]),
        snapshot_dir=output_dir / "snapshots",
    )
    total_rounds = int(selfplay["rounds"])
    if round_limit is not None:
        total_rounds = min(total_rounds, int(state["next_round"]) + round_limit)
    try:
        for round_index in range(int(state["next_round"]), total_rounds):
            started = time.time()
            round_dir = output_dir / f"round-{round_index:04d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            candidates: list[dict[str, Any]] = []
            alice_train_metrics: list[dict[str, float]] = []
            mix_schedule = list(selfplay.get("alice_oracle_mix_schedule") or [0.0])
            oracle_mix = float(mix_schedule[min(round_index, len(mix_schedule) - 1)])
            seen_cells = set(state["seen_goal_cells"])

            for group_index in range(int(selfplay["proposal_groups_per_round"])):
                seed = int(environment["train_seed_start"]) + round_index * 1000 + group_index
                env.reset(seed=seed)
                snapshot = env.capture_snapshot()
                initial_state = env.task_state()
                alice_episodes: list[PolicyEpisode] = []
                for generation_index in range(int(grpo["group_size"])):
                    if oracle_alice:
                        episode = _oracle_alice_episode(
                            processor, env, snapshot, max_steps=int(selfplay["alice_max_steps"])
                        )
                    else:
                        assert alice_model is not None
                        alice_model.eval()
                        episode = _model_alice_episode(
                            alice_model,
                            processor,
                            env,
                            snapshot,
                            device=alice_device,
                            max_steps=int(selfplay["alice_max_steps"]),
                            max_new_tokens=int(model_input["max_new_tokens"]),
                            temperature=float(grpo.get("alice_temperature", grpo["temperature"])),
                            top_p=float(grpo.get("alice_top_p", grpo["top_p"])),
                            oracle_mix=oracle_mix,
                        )
                    compiled, final_state, replay_equal = _replay_and_compile(
                        env,
                        snapshot,
                        initial_state,
                        episode,
                        min_displacement=float(selfplay["min_goal_displacement"]),
                        tolerance=float(selfplay["goal_tolerance"]),
                    )
                    candidate_dir = round_dir / "proposals" / f"group-{group_index:02d}-gen-{generation_index:02d}"
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    alice_path = candidate_dir / "alice_trajectory.json"
                    _write_json(alice_path, {
                        "seed": seed,
                        "initial_snapshot": snapshot.to_dict(),
                        "initial_state": initial_state,
                        "final_state": final_state,
                        "replay_equal": replay_equal,
                        "compiled": {
                            "task_id": compiled.task_id,
                            "valid": compiled.valid,
                            "reason": compiled.reason,
                            "displacement": compiled.displacement,
                            "goal_cell": compiled.goal_cell,
                            "goal_predicate": compiled.goal_predicate,
                        },
                        "steps": [_plain_step(step) for step in episode.steps],
                    })

                    repeated = compiled.goal_cell in seen_cells if compiled.goal_cell else False
                    novelty = 0.0 if repeated else 1.0
                    bob_episodes: list[PolicyEpisode] = []
                    spec = None
                    score = None
                    alice_reward = -1.0
                    if compiled.valid:
                        spec = EmbodiedTaskSpec(
                            task_id=compiled.task_id,
                            env_id=snapshot.env_id,
                            seed=seed,
                            initial_state_ref=snapshot.state_uri,
                            instruction=compiled.instruction,
                            goal_predicate=compiled.goal_predicate,
                            alice_trajectory_ref=str(alice_path),
                            feasibility_score=1.0,
                            novelty_score=novelty,
                            difficulty_target=float(selfplay["target_bob_success_rate"]),
                            max_steps=int(selfplay["bob_max_steps"]),
                            metadata={"goal_cell": compiled.goal_cell, "snapshot_sha256": snapshot.state_sha256},
                        )
                        goal_position = compiled.goal_predicate["position"]
                        bob_model.eval()
                        bob_episodes = [
                            _rollout_episode(
                                bob_model,
                                processor,
                                env,
                                snapshot,
                                role="bob",
                                instruction=BOB_INSTRUCTION,
                                max_steps=int(selfplay["bob_max_steps"]),
                                max_new_tokens=int(model_input["max_new_tokens"]),
                                temperature=float(grpo.get("bob_temperature", grpo["temperature"])),
                                top_p=float(grpo.get("bob_top_p", grpo["top_p"])),
                                device=bob_device,
                                goal_position=goal_position,
                            )
                            for _ in range(int(selfplay["bob_evaluations_per_task"]))
                        ]
                        success_rate = sum(item.success for item in bob_episodes) / len(bob_episodes)
                        score = BobTaskScore(compiled.task_id, success_rate, len(bob_episodes))
                        alice_reward = alice_task_reward(
                            spec,
                            score,
                            target_success_rate=float(selfplay["target_bob_success_rate"]),
                            repeated=repeated,
                        )
                        seen_cells.add(compiled.goal_cell)
                        _write_json(candidate_dir / "task.json", spec.to_dict())
                        _write_json(candidate_dir / "bob_scores.json", score.__dict__)
                        _write_json(candidate_dir / "bob_trajectories.json", {
                            "goal_position": goal_position,
                            "episodes": [item.artifact() for item in bob_episodes],
                        })
                    episode.reward = alice_reward
                    episode.success = compiled.valid
                    alice_episodes.append(episode)
                    _write_json(candidate_dir / "curriculum_feedback.json", {
                        "alice_reward": alice_reward,
                        "validity_gate": compiled.valid,
                        "novelty_score": novelty,
                        "repeated": repeated,
                        "bob_score": score.__dict__ if score is not None else None,
                    })
                    candidates.append({
                        "compiled": compiled,
                        "spec": spec,
                        "score": score,
                        "alice_reward": alice_reward,
                        "bob_episodes": bob_episodes,
                        "path": str(candidate_dir),
                    })

                if not oracle_alice:
                    assert alice_model is not None and alice_optimizer is not None
                    grpo_metrics = _train_group(
                        alice_model,
                        processor,
                        alice_episodes,
                        alice_optimizer,
                        device=alice_device,
                        update_epochs=int(grpo["update_epochs"]),
                        clip_epsilon=float(grpo["clip_epsilon"]),
                        beta=float(grpo["beta"]),
                        max_grad_norm=float(grpo["max_grad_norm"]),
                    )
                    teacher_steps = [
                        replace(
                            step,
                            completion=str(step.teacher_action),
                            completion_ids=processor.tokenizer.encode(
                                str(step.teacher_action), add_special_tokens=False
                            ),
                            action=str(step.teacher_action),
                            valid_action=True,
                            teacher_forced=True,
                        )
                        for episode in alice_episodes
                        for step in episode.steps
                        if step.teacher_action is not None
                    ]
                    dagger_metrics = supervised_action_update(
                        alice_model,
                        processor,
                        teacher_steps,
                        alice_optimizer,
                        device=alice_device,
                        max_grad_norm=float(grpo["max_grad_norm"]),
                    )
                    alice_train_metrics.append({
                        "oracle_mix": oracle_mix,
                        "grpo": grpo_metrics,
                        "dagger": dagger_metrics,
                    })

            valid_candidates = [item for item in candidates if item["spec"] is not None]
            selected = sorted(
                valid_candidates,
                key=lambda item: (-float(item["alice_reward"]), item["compiled"].task_id),
            )[: int(selfplay["selected_tasks_per_round"])]
            bob_train_metrics: list[dict[str, float]] = []
            for item in selected:
                bob_train_metrics.append(_train_group(
                    bob_model,
                    processor,
                    item["bob_episodes"],
                    bob_optimizer,
                    device=bob_device,
                    update_epochs=int(grpo["update_epochs"]),
                    clip_epsilon=float(grpo["clip_epsilon"]),
                    beta=float(grpo["beta"]),
                    max_grad_norm=float(grpo["max_grad_norm"]),
                ))

            holdout = _holdout_evaluation(
                bob_model,
                processor,
                env,
                list(map(int, environment["holdout_seeds"])),
                device=bob_device,
                config=config,
            )
            alice_adapter = round_dir / "alice_adapter"
            bob_adapter = round_dir / "bob_adapter"
            if alice_model is not None:
                alice_model.save_pretrained(alice_adapter)
                assert alice_optimizer is not None
                save_optimizer_state(
                    output_dir / "alice_optimizer.pt", alice_optimizer, next_iteration=round_index + 1
                )
            bob_model.save_pretrained(bob_adapter)
            processor.save_pretrained(bob_adapter)
            save_optimizer_state(
                output_dir / "bob_optimizer.pt", bob_optimizer, next_iteration=round_index + 1
            )
            metrics = {
                "round": round_index,
                "seconds": time.time() - started,
                "proposal_count": len(candidates),
                "valid_proposal_count": len(valid_candidates),
                "selected_task_ids": [item["compiled"].task_id for item in selected],
                "alice_reward_mean": float(np.mean([item["alice_reward"] for item in candidates])),
                "alice_train": alice_train_metrics,
                "bob_train": bob_train_metrics,
                "holdout": holdout,
                "oracle_alice": oracle_alice,
                "alice_oracle_mix": oracle_mix,
            }
            _write_json(round_dir / "metrics.json", metrics)
            state["next_round"] = round_index + 1
            state["seen_goal_cells"] = sorted(seen_cells)
            state["alice_adapter"] = str(alice_adapter) if alice_model is not None else None
            state["bob_adapter"] = str(bob_adapter)
            state["rounds"].append(metrics)
            _write_json(state_path, state)
            print(json.dumps({
                "round": round_index,
                "valid": len(valid_candidates),
                "selected": len(selected),
                "holdout_success": holdout["success_rate"],
                "output": str(round_dir),
            }, ensure_ascii=False), flush=True)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "embodied_selfplay_pilot.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--round-limit", type=int)
    parser.add_argument("--alice-adapter", type=Path)
    parser.add_argument("--bob-adapter", type=Path)
    parser.add_argument(
        "--oracle-alice",
        action="store_true",
        help="Use a scripted feasibility proposer to validate the pipeline without updating Alice",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.alice_adapter is not None:
        config["initial_alice_adapter"] = str(args.alice_adapter.resolve())
    if args.bob_adapter is not None:
        config["initial_bob_adapter"] = str(args.bob_adapter.resolve())
    output_dir = args.output_dir or REPO_ROOT / config["output_dir"]
    run(
        config,
        output_dir=output_dir.resolve(),
        oracle_alice=args.oracle_alice,
        round_limit=args.round_limit,
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
