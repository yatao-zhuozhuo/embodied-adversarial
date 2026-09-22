"""Run a small real VLM -> OpenETA simulator rollout with local Transformers.

This is a dependency-light smoke path for the Alice/Bob integration work.  It
uses the OpenETA dummy simulator until a ManiSkill environment is installed,
but the model observes the actual RGB frame emitted by the simulator and the
action is applied through the normal ``EnvAction``/``StepResult`` contract.

The default checkpoint is Qwen3.5-4B because this workspace does not contain a
Qwen3-4B checkpoint.  Pass ``--model`` to use an exact local checkpoint when
one is available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from adapter.protocol import EnvAction


DEFAULT_MODEL = "/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B"
ALLOWED_ACTIONS = ("MOVE", "GRASP", "RELEASE", "DONE")


def _parse_action(text: str) -> tuple[str, str]:
    """Recover a bounded action from model output without trusting free text."""

    raw = str(text or "").strip()
    try:
        payload = json.loads(raw)
        action = str(payload.get("action", "")).upper()
        reason = str(payload.get("reason", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        action = ""
        reason = "invalid JSON action"
    if action not in ALLOWED_ACTIONS:
        action = "DONE"
        reason = reason or "invalid action fallback"
    return action, reason


def _load_model(model_path: str):
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    return processor, model


def _decide(processor: Any, model: Any, observation: Any, instruction: str) -> tuple[str, str, str]:
    camera = observation.cameras[0]
    image = Image.fromarray(np.asarray(camera.rgb, dtype=np.uint8))
    prompt = (
        "You are the Bob embodied agent. Observe the image and follow the task.\n"
        f"Task: {instruction}\n"
        "Choose exactly one action. Return JSON only: "
        '{"action":"MOVE|GRASP|RELEASE|DONE","reason":"short reason"}'
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.cuda() if hasattr(value, "cuda") else value for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    generated = output[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
    action, reason = _parse_action(raw)
    return action, reason, raw


def run(
    model_path: str,
    output_path: Path,
    instruction: str,
    *,
    backend: str = "dummy",
    env_id: str = "PickCube-v1",
    max_steps: int = 1,
) -> dict[str, Any]:
    processor, model = _load_model(model_path)
    if backend == "maniskill":
        from adapter.maniskill_sim import ManiSkillSimulatorAdapter

        env = ManiSkillSimulatorAdapter(env_id=env_id, camera_resolution=128)
    elif backend == "dummy":
        from adapter.dummy_sim import DummySimulatorAdapter

        env = DummySimulatorAdapter()
    else:
        raise ValueError(f"unsupported backend: {backend}")

    observation = env.reset(task=instruction, seed=0)
    initial_observation = observation
    snapshot = env.capture_snapshot() if backend == "maniskill" else None
    alice_steps: list[dict[str, Any]] = []
    for _ in range(max(1, int(max_steps))):
        action, reason, raw = _decide(processor, model, observation, instruction)
        result = env.step(EnvAction(action_type=action, code=action))
        alice_steps.append({
            "action": action,
            "reason": reason,
            "raw": raw,
            "result": result.to_mcp_dict(),
        })
        if result.terminated or result.truncated or action == "DONE":
            break
        observation = result.observation

    bob_steps: list[dict[str, Any]] = []
    replay_equal = None
    if snapshot is not None:
        env.restore_snapshot(snapshot)
        replay_observation = env.observe()
        replay_equal = (
            replay_observation.robot.joint_positions == initial_observation.robot.joint_positions
        )
        bob_action, bob_reason, bob_raw = _decide(processor, model, replay_observation, instruction)
        bob_result = env.step(EnvAction(action_type=bob_action, code=bob_action))
        bob_steps.append({
            "action": bob_action,
            "reason": bob_reason,
            "raw": bob_raw,
            "result": bob_result.to_mcp_dict(),
        })
    payload = {
        "model": model_path,
        "backend": f"openeta/{backend}_{env_id}" if backend == "maniskill" else "openeta.dummy_sim-v0",
        "instruction": instruction,
        "observation": initial_observation.to_mcp_dict(),
        "snapshot": snapshot.to_dict() if snapshot is not None else None,
        "snapshot_replay_equal": replay_equal,
        "alice_steps": alice_steps,
        "bob_steps": bob_steps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("runs/qwen_embodied_rollout.json"))
    parser.add_argument("--instruction", default="Move the dummy robot toward the cube.")
    parser.add_argument("--backend", choices=("dummy", "maniskill"), default="dummy")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()
    payload = run(
        args.model,
        args.output,
        args.instruction,
        backend=args.backend,
        env_id=args.env_id,
        max_steps=args.max_steps,
    )
    alice_action = payload["alice_steps"][0]["action"] if payload["alice_steps"] else None
    bob_action = payload["bob_steps"][0]["action"] if payload["bob_steps"] else None
    print(json.dumps({
        "model": payload["model"],
        "backend": payload["backend"],
        "alice_action": alice_action,
        "bob_action": bob_action,
        "snapshot_replay_equal": payload["snapshot_replay_equal"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
