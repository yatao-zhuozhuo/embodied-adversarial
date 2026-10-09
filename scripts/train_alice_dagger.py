#!/usr/bin/env python3
"""Train Alice with online DAgger and fixed model-only evaluation seeds."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from agent.training.embodied_grpo import PolicyEpisode, PolicyStep
from scripts.run_embodied_selfplay import (
    ALICE_INSTRUCTION,
    _model_alice_episode,
    _replay_and_compile,
    _write_json,
)
from scripts.train_embodied_grpo import (
    load_lora_policy,
    load_optimizer_state,
    save_optimizer_state,
    supervised_action_update,
)


def _longest_repeat(actions: list[str]) -> int:
    longest = current = 0
    previous: str | None = None
    for action in actions:
        current = current + 1 if action == previous else 1
        longest = max(longest, current)
        previous = action
    return longest


def _episode_summary(
    episode: PolicyEpisode,
    compiled: Any,
    replay_equal: bool,
    *,
    seed: int,
) -> dict[str, Any]:
    actions = [step.action for step in episode.steps]
    return {
        "seed": seed,
        "valid": bool(compiled.valid),
        "reason": compiled.reason,
        "task_id": compiled.task_id,
        "displacement": float(compiled.displacement),
        "replay_equal": bool(replay_equal),
        "steps": len(episode.steps),
        "teacher_steps": sum(step.teacher_forced for step in episode.steps),
        "teacher_labels": sum(step.teacher_action is not None for step in episode.steps),
        "grasped": any(step.is_grasped for step in episode.steps),
        "environment_success": any(step.success or step.is_obj_placed for step in episode.steps),
        "longest_repeated_action": _longest_repeat(actions),
        "actions": actions,
        "trajectory": episode.artifact(),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float]:
    count = max(1, len(records))
    return {
        "episodes": float(len(records)),
        "valid_rate": sum(item["valid"] for item in records) / count,
        "grasp_rate": sum(item["grasped"] for item in records) / count,
        "environment_success_rate": sum(item["environment_success"] for item in records) / count,
        "replay_rate": sum(item["replay_equal"] for item in records) / count,
        "mean_displacement": float(np.mean([item["displacement"] for item in records])),
        "mean_steps": float(np.mean([item["steps"] for item in records])),
        "mean_longest_repeated_action": float(np.mean([
            item["longest_repeated_action"] for item in records
        ])),
    }


def _teacher_labels(episodes: list[PolicyEpisode], processor: Any) -> list[PolicyStep]:
    labels: list[PolicyStep] = []
    for episode in episodes:
        for step in episode.steps:
            if step.teacher_action is None:
                continue
            action = str(step.teacher_action)
            labels.append(replace(
                step,
                completion=action,
                completion_ids=processor.tokenizer.encode(action, add_special_tokens=False),
                action=action,
                valid_action=True,
                teacher_forced=True,
            ))
    return labels


_CONTROL_PHASE_PATTERN = re.compile(r"control_phase: ([a-z_]+)")


def _control_phase(step: PolicyStep) -> str:
    match = _CONTROL_PHASE_PATTERN.search(step.prompt)
    return match.group(1) if match is not None else "unknown"


def _balance_phase_labels(
    labels: list[PolicyStep],
    *,
    max_multiplier: int,
) -> tuple[list[PolicyStep], dict[str, dict[str, int]]]:
    """Oversample sparse controller phases before shuffling DAgger examples.

    Action-level loss weighting alone leaves GRASP examples isolated in a few
    minibatches.  Phase balancing repeats complete state/action examples so
    each update round repeatedly sees transitions such as close_gripper_now.
    """

    if max_multiplier <= 0:
        raise ValueError("phase_balance_max_multiplier must be positive")
    grouped: dict[str, list[PolicyStep]] = collections.defaultdict(list)
    for step in labels:
        grouped[_control_phase(step)].append(step)
    if not grouped:
        return [], {"before": {}, "after": {}}
    target = max(len(group) for group in grouped.values())
    balanced: list[PolicyStep] = []
    for group in grouped.values():
        repeats = min(max_multiplier, max(1, (target + len(group) - 1) // len(group)))
        expanded = (group * repeats)[:target]
        balanced.extend(expanded)
    random.shuffle(balanced)
    return balanced, {
        "before": {phase: len(group) for phase, group in sorted(grouped.items())},
        "after": dict(sorted(collections.Counter(map(_control_phase, balanced)).items())),
    }


def _rollout(
    model: Any,
    processor: Any,
    env: ManiSkillSimulatorAdapter,
    *,
    seed: int,
    oracle_mix: float,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[PolicyEpisode, dict[str, Any]]:
    environment = config["environment"]
    optimization = config["optimization"]
    env.reset(seed=seed)
    snapshot = env.capture_snapshot()
    initial_state = env.task_state()
    model.eval()
    episode = _model_alice_episode(
        model,
        processor,
        env,
        snapshot,
        device=device,
        max_steps=int(environment["max_steps"]),
        max_new_tokens=int(config["model_input"]["max_new_tokens"]),
        temperature=float(optimization["temperature"]),
        top_p=float(optimization["top_p"]),
        oracle_mix=oracle_mix,
    )
    compiled, final_state, replay_equal = _replay_and_compile(
        env,
        snapshot,
        initial_state,
        episode,
        min_displacement=float(optimization["min_goal_displacement"]),
        tolerance=float(optimization["goal_tolerance"]),
    )
    summary = _episode_summary(episode, compiled, replay_equal, seed=seed)
    summary.update({
        "snapshot": snapshot.to_dict(),
        "initial_state": initial_state,
        "final_state": final_state,
    })
    return episode, summary


def train(
    config: dict[str, Any],
    *,
    output_dir: Path,
    initial_adapter: Path,
    initial_optimizer: Path | None,
    round_limit: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config.json", config)
    state_path = output_dir / "state.json"
    optimization = config["optimization"]
    environment = config["environment"]
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {
        "version": config["version"],
        "next_round": 0,
        "oracle_mix": float(optimization["oracle_mix_start"]),
        "adapter": str(initial_adapter.resolve()),
        "rounds": [],
    }
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cuda", int(config["device"]))
    torch.cuda.set_device(device)
    processor = AutoProcessor.from_pretrained(config["model"])
    processor.image_processor.min_pixels = int(config["model_input"]["min_image_pixels"])
    processor.image_processor.max_pixels = int(config["model_input"]["max_image_pixels"])
    model = load_lora_policy(
        config["model"],
        device=device,
        lora_rank=int(optimization["lora_rank"]),
        lora_alpha=int(optimization["lora_alpha"]),
        adapter_path=Path(state["adapter"]),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(optimization["learning_rate"]),
    )
    optimizer_path = output_dir / "alice_optimizer.pt"
    if optimizer_path.is_file():
        load_optimizer_state(optimizer_path, optimizer)
    elif initial_optimizer is not None:
        load_optimizer_state(initial_optimizer, optimizer)
    env = ManiSkillSimulatorAdapter(
        env_id=environment["env_id"],
        camera_resolution=int(environment["camera_resolution"]),
        translation_step_m=float(environment["translation_step_m"]),
        fine_translation_step_m=float(environment["fine_translation_step_m"]),
        max_episode_steps=int(environment["episode_horizon"]),
        snapshot_dir=output_dir / "snapshots",
    )
    stop_round = int(optimization["rounds"])
    if round_limit is not None:
        stop_round = min(stop_round, int(state["next_round"]) + round_limit)
    try:
        for round_index in range(int(state["next_round"]), stop_round):
            started = time.time()
            oracle_mix = float(state["oracle_mix"])
            train_episodes: list[PolicyEpisode] = []
            train_records: list[dict[str, Any]] = []
            seed_start = int(environment["train_seed_start"]) + (
                round_index * int(environment["train_seeds_per_round"])
            )
            for seed in range(seed_start, seed_start + int(environment["train_seeds_per_round"])):
                episode, record = _rollout(
                    model, processor, env, seed=seed, oracle_mix=oracle_mix,
                    device=device, config=config,
                )
                train_episodes.append(episode)
                train_records.append(record)
            labels = _teacher_labels(train_episodes, processor)
            labels, phase_balance = _balance_phase_labels(
                labels,
                max_multiplier=int(optimization["phase_balance_max_multiplier"]),
            )
            update = supervised_action_update(
                model,
                processor,
                labels,
                optimizer,
                device=device,
                max_grad_norm=float(optimization["max_grad_norm"]),
                gradient_accumulation=int(optimization["gradient_accumulation"]),
                rare_action_weight_cap=float(optimization["rare_action_weight_cap"]),
            )
            eval_records: list[dict[str, Any]] = []
            for seed in map(int, environment["eval_seeds"]):
                _, record = _rollout(
                    model, processor, env, seed=seed, oracle_mix=0.0,
                    device=device, config=config,
                )
                eval_records.append(record)
            train_metrics = _aggregate(train_records)
            eval_metrics = _aggregate(eval_records)
            next_mix = oracle_mix
            if (
                eval_metrics["valid_rate"] >= float(optimization["promotion_valid_rate"])
                and eval_metrics["grasp_rate"] >= float(optimization["promotion_grasp_rate"])
            ):
                next_mix = max(
                    float(optimization["oracle_mix_min"]),
                    oracle_mix - float(optimization["mix_decrement"]),
                )
            round_dir = output_dir / f"round-{round_index:04d}"
            adapter_dir = round_dir / "alice_adapter"
            model.save_pretrained(adapter_dir)
            processor.save_pretrained(adapter_dir)
            save_optimizer_state(
                optimizer_path, optimizer, next_iteration=round_index + 1
            )
            metrics = {
                "round": round_index,
                "seconds": time.time() - started,
                "oracle_mix": oracle_mix,
                "next_oracle_mix": next_mix,
                "dagger_update": update | {"phase_balance": phase_balance},
                "train": train_metrics,
                "model_only_eval": eval_metrics,
            }
            _write_json(round_dir / "train_trajectories.json", train_records)
            _write_json(round_dir / "model_only_eval.json", eval_records)
            _write_json(round_dir / "metrics.json", metrics)
            state["next_round"] = round_index + 1
            state["oracle_mix"] = next_mix
            state["adapter"] = str(adapter_dir)
            state["rounds"].append(metrics)
            _write_json(state_path, state)
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "alice_dagger_curriculum.json",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--initial-optimizer", type=Path)
    parser.add_argument("--round-limit", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or REPO_ROOT / config["output_dir"]
    train(
        config,
        output_dir=output_dir.resolve(),
        initial_adapter=args.initial_adapter,
        initial_optimizer=args.initial_optimizer,
        round_limit=args.round_limit,
    )


if __name__ == "__main__":
    main()
