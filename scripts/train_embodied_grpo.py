#!/usr/bin/env python3
"""Train a small LoRA policy with image-conditioned ManiSkill GRPO rollouts.

This is a single-process correctness entrypoint, not the final throughput
trainer.  Each iteration captures one initial simulator state, samples a group
of trajectories from that exact snapshot, standardizes their trusted rewards,
and applies a clipped token-level GRPO update to generated action-code tokens only.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from adapter.protocol import EnvAction
from agent.training.embodied_grpo import (
    ALLOWED_ACTIONS,
    PolicyEpisode,
    PolicyStep,
    clipped_grpo_objective,
    episode_reward,
    group_advantages,
    parse_action,
    receipt_flag,
)


DEFAULT_MODEL = "/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B"
DEFAULT_INSTRUCTION = "Pick up the cube and place it in the target area."
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
)


def _prompt(role: str, instruction: str, observation: Any, step_index: int) -> str:
    xyz = observation.robot.end_effector_pose.get("xyz", [])
    visible_objects = [
        {"name": item.get("name"), "position": item.get("position")}
        for item in observation.objects
    ]
    role_text = (
        "You are Bob, the task-solving embodied agent. Complete the given task."
        if role == "bob"
        else "You are Alice, the task-demonstration agent. Produce a feasible successful trajectory."
    )
    positions = {
        item.get("name"): item.get("position")
        for item in observation.objects
        if isinstance(item.get("position"), list) and len(item["position"]) >= 3
    }
    cube = positions.get("cube")
    goal = positions.get("goal")
    cube_from_tcp = (
        [round(float(cube[i]) - float(xyz[i]), 4) for i in range(3)]
        if cube is not None and len(xyz) >= 3 else []
    )
    goal_from_cube = (
        [round(float(goal[i]) - float(cube[i]), 4) for i in range(3)]
        if cube is not None and goal is not None else []
    )
    is_grasped = bool(observation.metadata.get("is_grasped", False))
    control_phase = "unknown"
    phase_error = 0.0
    if len(cube_from_tcp) == 3 and len(goal_from_cube) == 3:
        if not is_grasped:
            for name, error in (
                ("align_cube_x", cube_from_tcp[0]),
                ("align_cube_y", cube_from_tcp[1]),
                ("descend_to_cube", cube_from_tcp[2]),
            ):
                if abs(error) > 0.004:
                    control_phase, phase_error = name, error
                    break
            else:
                control_phase = "close_gripper_now"
        else:
            safe_z = max(float(goal[2]), 0.16)
            horizontal_error = max(abs(goal_from_cube[0]), abs(goal_from_cube[1]))
            if horizontal_error > 0.03 and abs(safe_z - float(xyz[2])) > 0.004:
                control_phase = "lift_for_transport"
                phase_error = safe_z - float(xyz[2])
            elif abs(goal_from_cube[0]) > 0.012:
                control_phase, phase_error = "align_goal_x", goal_from_cube[0]
            elif abs(goal_from_cube[1]) > 0.012:
                control_phase, phase_error = "align_goal_y", goal_from_cube[1]
            elif abs(goal_from_cube[2]) > 0.004:
                control_phase, phase_error = "align_goal_z", goal_from_cube[2]
            else:
                control_phase = "task_complete"
    return (
        f"{role_text}\n"
        f"Task: {instruction}\n"
        f"Step: {step_index}; end-effector xyz: {xyz}; visible objects: {visible_objects}; "
        f"cube_minus_tcp_xyz: {cube_from_tcp}; goal_minus_cube_xyz: {goal_from_cube}; "
        f"is_grasped: {is_grasped}; control_phase: {control_phase}; "
        f"phase_signed_error: {round(float(phase_error), 4)}.\n"
        "Control rule: before grasping, align TCP x then y with the cube, lower z to the cube, "
        "then GRASP. After is_grasped becomes true, lift in z, align the cube x/y with the goal, "
        "then align z. Pick the sign from the corresponding relative vector; use *_FINE when "
        "the remaining absolute error is below 0.025. If control_phase is close_gripper_now, "
        "choose GRASP.\n"
        "Choose one bounded action from MOVE, MOVE_X_POS, MOVE_X_NEG, MOVE_Y_POS, "
        "MOVE_Y_NEG, MOVE_Z_POS, MOVE_Z_NEG, MOVE_X_POS_FINE, MOVE_X_NEG_FINE, "
        "MOVE_Y_POS_FINE, MOVE_Y_NEG_FINE, MOVE_Z_POS_FINE, MOVE_Z_NEG_FINE, "
        "GRASP, RELEASE, DONE. Use *_FINE close to the cube or goal. "
        "Return only the action code, with no JSON, markdown, or explanation."
    )


def _model_inputs(processor: Any, image: np.ndarray, prompt: str) -> tuple[dict[str, Any], str]:
    pil_image = Image.fromarray(image.astype(np.uint8))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil_image},
        {"type": "text", "text": prompt},
    ]}]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return processor(text=[rendered], images=[pil_image], return_tensors="pt"), rendered


def _to_device(inputs: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def _generate_action(
    model: Any,
    processor: Any,
    observation: Any,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    constrain_actions: bool = True,
) -> tuple[list[int], str]:
    image = np.asarray(observation.cameras[0].rgb, dtype=np.uint8)
    inputs, _ = _model_inputs(processor, image, prompt)
    inputs = _to_device(inputs, device)
    prompt_length = inputs["input_ids"].shape[1]
    prefix_allowed_tokens_fn = None
    if constrain_actions:
        action_tokens = [
            processor.tokenizer.encode(action, add_special_tokens=False)
            for action in ALLOWED_ACTIONS
        ]
        eos_token_id = int(processor.tokenizer.eos_token_id)

        def allowed_tokens(_batch_id: int, input_ids: torch.Tensor) -> list[int]:
            generated = input_ids[prompt_length:].tolist()
            matches = [tokens for tokens in action_tokens if tokens[: len(generated)] == generated]
            allowed = {
                tokens[len(generated)]
                for tokens in matches
                if len(tokens) > len(generated)
            }
            if any(len(tokens) == len(generated) for tokens in matches):
                allowed.add(eos_token_id)
            return sorted(allowed) if allowed else [eos_token_id]

        prefix_allowed_tokens_fn = allowed_tokens
    with torch.inference_mode():
        sequence = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
    completion_ids = sequence[0, prompt_length:].detach().cpu().tolist()
    completion = processor.tokenizer.decode(completion_ids, skip_special_tokens=True)
    return completion_ids, completion


def _rollout_episode(
    model: Any,
    processor: Any,
    env: ManiSkillSimulatorAdapter,
    snapshot: Any,
    *,
    role: str,
    instruction: str,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    goal_position: list[float] | None = None,
) -> PolicyEpisode:
    env.restore_snapshot(snapshot)
    if goal_position is not None:
        env.set_goal_position(goal_position)
    observation = env.observe()
    steps: list[PolicyStep] = []
    for step_index in range(max_steps):
        prompt = _prompt(role, instruction, observation, step_index)
        image = np.asarray(observation.cameras[0].rgb, dtype=np.uint8)
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
        result = env.step(EnvAction(action_type=parsed.action, code=parsed.action))
        step = PolicyStep(
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
        )
        steps.append(step)
        observation = result.observation
        if step.success or step.is_obj_placed or step.terminated or step.truncated or parsed.action == "DONE":
            break
    reward = episode_reward(steps, role=role)
    return PolicyEpisode(
        role=role,
        reward=reward,
        success=any(step.success or step.is_obj_placed for step in steps),
        steps=steps,
    )


def _completion_logps(
    model: Any,
    processor: Any,
    step: PolicyStep,
    *,
    device: torch.device,
) -> torch.Tensor:
    inputs, _ = _model_inputs(processor, step.image, step.prompt)
    inputs = _to_device(inputs, device)
    prompt_length = int(inputs["input_ids"].shape[1])
    completion_ids = step.completion_ids or [processor.tokenizer.eos_token_id]
    completion = torch.tensor([completion_ids], dtype=torch.long, device=device)
    inputs["input_ids"] = torch.cat([inputs["input_ids"], completion], dim=1)
    suffix_mask = torch.ones_like(completion)
    inputs["attention_mask"] = torch.cat([inputs["attention_mask"], suffix_mask], dim=1)
    if "mm_token_type_ids" in inputs:
        type_suffix = torch.zeros_like(completion)
        inputs["mm_token_type_ids"] = torch.cat(
            [inputs["mm_token_type_ids"], type_suffix], dim=1
        )
    outputs = model(**inputs, use_cache=False)
    logits = outputs.logits[:, prompt_length - 1 : -1, :].float()
    return torch.log_softmax(logits, dim=-1).gather(-1, completion.unsqueeze(-1)).squeeze(0).squeeze(-1)


def _adapter_disabled(model: Any):
    disable = getattr(model, "disable_adapter", None)
    return disable() if callable(disable) else nullcontext()


def load_lora_policy(
    model_path: str,
    *,
    device: torch.device,
    lora_rank: int,
    lora_alpha: int,
    adapter_path: Path | None = None,
) -> Any:
    """Load a fresh or resumable trainable embodied policy."""

    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    if adapter_path is not None:
        model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=True)
    else:
        model = get_peft_model(
            base,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.0,
                target_modules=list(LORA_TARGETS),
                task_type="CAUSAL_LM",
            ),
        )
    return model


def save_optimizer_state(
    path: Path,
    optimizer: torch.optim.Optimizer,
    *,
    next_iteration: int,
) -> None:
    torch.save({
        "optimizer": optimizer.state_dict(),
        "next_iteration": next_iteration,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }, path)


def load_optimizer_state(path: Path, optimizer: torch.optim.Optimizer) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.random.set_rng_state(payload["torch_rng"])
    torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return int(payload["next_iteration"])


def _train_group(
    model: Any,
    processor: Any,
    episodes: list[PolicyEpisode],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    update_epochs: int,
    clip_epsilon: float,
    beta: float,
    max_grad_norm: float,
) -> dict[str, float]:
    rewards = [episode.reward for episode in episodes]
    advantages = group_advantages(rewards)
    eligible_steps = sum(
        not step.teacher_forced for episode in episodes for step in episode.steps
    )
    if eligible_steps == 0:
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "advantage_abs_mean": float(np.mean(np.abs(advantages))),
            "samples": 0.0,
            "skipped_no_on_policy_steps": 1.0,
        }
    if float(np.std(rewards)) < 1e-8:
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "advantage_abs_mean": 0.0,
            "samples": float(eligible_steps),
            "skipped_zero_variance": 1.0,
        }
    samples: list[tuple[PolicyStep, float, torch.Tensor, torch.Tensor]] = []
    model.eval()
    for episode, advantage in zip(episodes, advantages, strict=True):
        for step in episode.steps:
            if step.teacher_forced:
                continue
            with torch.no_grad():
                old_logps = _completion_logps(model, processor, step, device=device).detach()
                with _adapter_disabled(model):
                    reference_logps = _completion_logps(
                        model, processor, step, device=device
                    ).detach()
            samples.append((step, advantage, old_logps, reference_logps))

    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(update_epochs):
        optimizer.zero_grad(set_to_none=True)
        model.train()
        epoch_loss = 0.0
        for step, advantage, old_logps, reference_logps in samples:
            current_logps = _completion_logps(model, processor, step, device=device)
            sample_loss = clipped_grpo_objective(
                current_logps,
                old_logps,
                reference_logps,
                advantage,
                clip_epsilon=clip_epsilon,
                beta=beta,
            )
            # Backpropagate each independent trajectory step immediately.  This
            # is mathematically equivalent to differentiating the mean loss,
            # while avoiding retention of an entire n=16 multimodal group graph.
            (sample_loss / len(samples)).backward()
            epoch_loss += float(sample_loss.detach().cpu())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        losses.append(epoch_loss / len(samples))
        grad_norms.append(float(grad_norm.detach().cpu()))
    return {
        "loss": sum(losses) / len(losses),
        "grad_norm": sum(grad_norms) / len(grad_norms),
        "reward_mean": float(np.mean([episode.reward for episode in episodes])),
        "reward_std": float(np.std([episode.reward for episode in episodes])),
        "advantage_abs_mean": float(np.mean(np.abs(advantages))),
        "samples": float(len(samples)),
    }


def supervised_action_update(
    model: Any,
    processor: Any,
    steps: list[PolicyStep],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_grad_norm: float,
    gradient_accumulation: int = 8,
    rare_action_weight_cap: float = 4.0,
) -> dict[str, float]:
    """Apply one DAgger-style update to explicitly marked teacher actions."""

    if not steps:
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "samples": 0.0,
            "optimizer_steps": 0.0,
        }
    if gradient_accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive")
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.action] = counts.get(step.action, 0) + 1
    maximum = max(counts.values())
    optimizer.zero_grad(set_to_none=True)
    model.train()
    losses: list[float] = []
    grad_norms: list[float] = []
    pending = 0
    for step in steps:
        weight = min(rare_action_weight_cap, maximum / counts[step.action])
        loss = -_completion_logps(model, processor, step, device=device).mean() * weight
        (loss / gradient_accumulation).backward()
        losses.append(float(loss.detach().cpu()))
        pending += 1
        if pending == gradient_accumulation:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            grad_norms.append(float(norm.detach().cpu()))
            pending = 0
    if pending:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        grad_norms.append(float(norm.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "grad_norm": float(np.mean(grad_norms)),
        "samples": float(len(steps)),
        "optimizer_steps": float(len(grad_norms)),
    }


def train(args: argparse.Namespace) -> None:
    if args.group_size < 2:
        raise ValueError("--group-size must be at least 2")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(vars(args), default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
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
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    start_iteration = 0
    if args.trainer_state is not None:
        start_iteration = load_optimizer_state(args.trainer_state, optimizer)
    env = ManiSkillSimulatorAdapter(
        env_id=args.env_id,
        camera_resolution=args.camera_resolution,
        snapshot_dir=output_dir / "snapshots",
    )
    metrics_path = output_dir / "metrics.jsonl"
    rollouts_path = output_dir / "rollouts.jsonl"
    try:
        for iteration in range(start_iteration, start_iteration + args.iterations):
            env.reset(seed=args.seed + iteration)
            snapshot = env.capture_snapshot()
            model.eval()
            started = time.time()
            episodes = [
                _rollout_episode(
                    model,
                    processor,
                    env,
                    snapshot,
                    role=args.role,
                    instruction=args.instruction,
                    max_steps=args.max_steps,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    device=device,
                )
                for _ in range(args.group_size)
            ]
            with rollouts_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "iteration": iteration,
                    "snapshot": snapshot.to_dict(),
                    "episodes": [episode.artifact() for episode in episodes],
                }, ensure_ascii=False) + "\n")
            metrics = _train_group(
                model,
                processor,
                episodes,
                optimizer,
                device=device,
                update_epochs=args.update_epochs,
                clip_epsilon=args.clip_epsilon,
                beta=args.beta,
                max_grad_norm=args.max_grad_norm,
            )
            metrics.update({
                "iteration": iteration,
                "seconds": time.time() - started,
                "cuda_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
                "rewards": [episode.reward for episode in episodes],
                "actions": [[step.action for step in episode.steps] for episode in episodes],
            })
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            if (iteration + 1) % args.save_every == 0:
                checkpoint = output_dir / f"checkpoint-{iteration + 1:04d}"
                model.save_pretrained(checkpoint)
                processor.save_pretrained(checkpoint)
                save_optimizer_state(
                    checkpoint / "trainer_state.pt",
                    optimizer,
                    next_iteration=iteration + 1,
                )
        model.save_pretrained(output_dir / "final")
        processor.save_pretrained(output_dir / "final")
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--trainer-state", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/embodied_bob_grpo"))
    parser.add_argument("--role", choices=("alice", "bob"), default="bob")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--camera-resolution", type=int, default=64)
    parser.add_argument("--min-image-pixels", type=int, default=56 * 56)
    parser.add_argument("--max-image-pixels", type=int, default=112 * 112)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=1)
    return parser


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    train(build_parser().parse_args())
