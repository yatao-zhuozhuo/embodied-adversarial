#!/usr/bin/env python3
"""Warm-start an Alice or Bob LoRA from trusted scripted PickCube actions."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from adapter.protocol import EnvAction
from agent.training.embodied_grpo import PolicyStep, receipt_flag
from scripts.run_embodied_selfplay import ALICE_INSTRUCTION, BOB_INSTRUCTION, _oracle_action
from scripts.train_embodied_grpo import (
    DEFAULT_MODEL,
    _completion_logps,
    _prompt,
    load_lora_policy,
)


def _collect_demonstration(
    env: ManiSkillSimulatorAdapter,
    processor: Any,
    *,
    role: str,
    seed: int,
    max_steps: int,
) -> tuple[list[PolicyStep], bool]:
    observation = env.reset(seed=seed)
    instruction = ALICE_INSTRUCTION if role == "alice" else BOB_INSTRUCTION
    steps: list[PolicyStep] = []
    grasp_attempts = 0
    for step_index in range(max_steps):
        action = _oracle_action(env, observation, grasp_attempts)
        grasp_attempts += int(action == "GRASP")
        prompt = _prompt(role, instruction, observation, step_index)
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
        if steps[-1].success or steps[-1].is_obj_placed or result.truncated or action == "DONE":
            break
    return steps, any(step.success or step.is_obj_placed for step in steps)


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)
    processor.image_processor.min_pixels = args.min_image_pixels
    processor.image_processor.max_pixels = args.max_image_pixels
    model = load_lora_policy(
        args.model,
        device=device,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        adapter_path=args.adapter_path,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    env = ManiSkillSimulatorAdapter(
        camera_resolution=args.camera_resolution,
        translation_step_m=args.translation_step_m,
        fine_translation_step_m=args.fine_translation_step_m,
        max_episode_steps=args.max_steps,
        snapshot_dir=output_dir / "snapshots",
    )
    started = time.time()
    demonstrations: list[tuple[int, list[PolicyStep], bool]] = []
    try:
        for offset in range(args.demonstrations):
            seed = args.seed + offset
            steps, success = _collect_demonstration(
                env, processor, role=args.role, seed=seed, max_steps=args.max_steps
            )
            if not success:
                raise RuntimeError(f"scripted demonstration failed for seed {seed}")
            demonstrations.append((seed, steps, success))
    finally:
        env.close()

    losses: list[float] = []
    grad_norms: list[float] = []
    model.train()
    action_counts = Counter(
        step.action for _, steps, _ in demonstrations for step in steps
    )
    maximum_action_count = max(action_counts.values())
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    for _ in range(args.epochs):
        random.shuffle(demonstrations)
        for _, steps, _ in demonstrations:
            for step in steps:
                token_logps = _completion_logps(model, processor, step, device=device)
                action_weight = min(
                    args.rare_action_weight_cap,
                    maximum_action_count / action_counts[step.action],
                )
                loss = -token_logps.mean() * action_weight
                (loss / args.gradient_accumulation).backward()
                losses.append(float(loss.detach().cpu()))
                pending += 1
                if pending == args.gradient_accumulation:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    grad_norms.append(float(norm.detach().cpu()))
                    pending = 0
    if pending:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        grad_norms.append(float(norm.detach().cpu()))

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    records = [{
        "seed": seed,
        "success": success,
        "steps": [step.artifact() for step in steps],
    } for seed, steps, success in demonstrations]
    (output_dir / "demonstrations.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = {
        "role": args.role,
        "demonstrations": len(demonstrations),
        "samples": sum(len(steps) for _, steps, _ in demonstrations) * args.epochs,
        "loss_mean": float(np.mean(losses)),
        "loss_final": losses[-1],
        "grad_norm_mean": float(np.mean(grad_norms)),
        "action_counts": dict(sorted(action_counts.items())),
        "rare_action_weight_cap": args.rare_action_weight_cap,
        "seconds": time.time() - started,
        "adapter": str(adapter_dir),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("alice", "bob"), required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demonstrations", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rare-action-weight-cap", type=float, default=4.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--camera-resolution", type=int, default=64)
    parser.add_argument("--translation-step-m", type=float, default=0.05)
    parser.add_argument("--fine-translation-step-m", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--min-image-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-image-pixels", type=int, default=112 * 112)
    return parser


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train(build_parser().parse_args())
