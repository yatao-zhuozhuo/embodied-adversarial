#!/usr/bin/env python3
"""Verify compressed MetaWorld actions in the same direct env used for originals."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter.protocol import EnvObservation
from agent.runtime import two_tool_input as inputs
from agent.runtime.two_tool_xml import PROTOCOL, parse_decision, serialize_decision
from scripts.collect_two_tool_sft import _save_frame, _write_json
from scripts.export_two_tool_sft import swift_messages
from scripts.replay_expert_two_tool_sft import SAMPLE_SCHEMA, SCHEMA
from scripts.generate_metaworld_expert_actions import POLICIES
from sim.env_registry import _make_metaworld_direct
from sim.two_tool_contract import controller_contract
from sim.unified_env import UnifiedEnv


def _mcp_observation(observation: dict) -> dict:
    return EnvObservation.from_dict(observation).to_mcp_dict()


def replay_one(source_path: Path, output_root: Path, task: str,
               max_control_steps: int = 400, adaptive_expert: bool = False) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("bench") != "metaworld" or source.get("source_success") is not True:
        raise ValueError("Expected a successful MetaWorld source candidate")
    task_id, seed = source["task_id"], source["seed"]
    segments = source["segments"]
    if not segments or sum(item["num_steps"] for item in segments) > max_control_steps:
        raise ValueError("Invalid compressed control-step budget")
    directory = output_root / "episodes" / "metaworld" / task_id / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    raw = _make_metaworld_direct(task_id, image_width=256, image_height=256, seed=seed)
    env = UnifiedEnv(raw, render_mode="rgb_array")
    if adaptive_expert:
        from metaworld import policies
        expert = getattr(policies, POLICIES[task_id])()
        tolerance = float(source["compression_tolerance_linf"])
    contract = {"action_dim": 4, "action_low": [-1.0] * 4, "action_high": [1.0] * 4}
    created = {**contract, "input_control_contract": controller_contract(env, "metaworld")}
    profile = inputs.make_profile("metaworld", task_id, created, contract)
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    record = {"schema_version": SCHEMA, "model": "artificial", "teacher_model": "artificial",
              "environment_seed_protocol": raw._openeta_seed_protocol, "creation_seed": seed,
              "provider": "artificial", "action_source": (
                  "official_scripted_policy_live_compressed" if adaptive_expert else "expert_replay"),
              "output_source": "constructed_expert_xml", "reasoning_source": "empty",
              "posthoc_annotation": False, "model_protocol": PROTOCOL,
              "input_protocol": inputs.INPUT_PROTOCOL, "input_profile": profile,
              "action_contract": contract, "bench": "metaworld", "task_id": task_id,
              "env_id": source["env_id"], "seed": seed,
              "source_path": str(source_path.resolve()), "source_sha256": source_hash,
              "source_kind": source["source_kind"],
              "source_dataset": source["source_dataset"],
              "source_steps": sum(item["num_steps"] for item in segments),
              "success": False, "stop_reason": None, "turns": [], "samples": 0}
    samples = []
    try:
        observation, _ = env.reset(seed=seed)
        for _ in range(5):
            observation, _, terminated, truncated, _ = env.step([0.0, 0.0, 0.0, -1.0])
            if terminated or truncated:
                raise RuntimeError("episode ended during host settle")
        current_obs = _mcp_observation(observation)
        current = inputs.snapshot(current_obs, _save_frame(current_obs, directory, 0),
                                  directory, profile, "obs-0", None)
        previous = None
        history = []
        used_steps = 0
        max_turns = max(60, max_control_steps if adaptive_expert else len(segments))
        pending_action = None
        for offset in range(max_control_steps):
            if used_steps >= max_control_steps or (not adaptive_expert and offset >= len(segments)):
                break
            turn = offset + 1
            if adaptive_expert:
                proposed = (pending_action if pending_action is not None else
                            expert.get_action(raw.unwrapped._get_obs()))
                pending_action = None
                action = np.clip(np.asarray(proposed, dtype=np.float32), -1.0, 1.0)
                requested_steps = min(10, max_control_steps - used_steps)
                if tolerance == -1:
                    requested_steps = 1
            else:
                segment = segments[offset]
                action = np.asarray(segment["action"], dtype=np.float32)
                requested_steps = segment["num_steps"]
            if action.shape != (4,) or not np.all(np.isfinite(action)):
                raise ValueError(f"invalid action at turn {turn}")
            if type(requested_steps) is not int or not 1 <= requested_steps <= 10:
                raise ValueError(f"invalid num_steps at turn {turn}")
            package = inputs.build_input(task=inputs.task_description(task_id, task),
                profile=profile, current=current, previous=previous, history=history,
                turn=turn, max_turns=max_turns,
                remaining_steps=max_control_steps - used_steps)
            success = False
            for executed in range(1, requested_steps + 1):
                observation, _, terminated, truncated, info = env.step(action)
                success = isinstance(info, dict) and info.get("success") == 1
                if success or terminated or truncated:
                    break
                if adaptive_expert and executed < requested_steps:
                    pending_action = np.clip(np.asarray(
                        expert.get_action(raw.unwrapped._get_obs()), dtype=np.float32), -1.0, 1.0)
                    if np.max(np.abs(pending_action - action)) > tolerance:
                        break
                    pending_action = None
            parameters = {"action": action.tolist(), "num_steps": executed}
            xml = serialize_decision("step_env", parameters, reasoning="")
            parse_decision(xml, contract=contract,
                           remaining_steps=max_control_steps - used_steps)
            used_steps += executed
            assistant = {"role": "assistant", "content": xml}
            messages, images = swift_messages(copy.deepcopy(package["messages"]) + [assistant])
            event = {"turn": turn, "valid": True, "validation": {"valid": True},
                     "tool": "step_env", "arguments": parameters, "model_message": assistant,
                     "tool_request": {"name": "step_env", "arguments": {**parameters, "strict": True}},
                     "executed_control_steps": executed, "terminated": bool(terminated),
                     "truncated": bool(truncated), "official_success": bool(success),
                     "input_sha256": package["messages_sha256"]}
            record["turns"].append(event)
            samples.append({"schema_version": SAMPLE_SCHEMA,
                            "sample_id": f"metaworld/{task_id}/seed-{seed}:turn-{turn:04d}",
                            "episode_id": str(directory.resolve()), "split": "train",
                            "messages": messages, "images": images,
                            "metadata": {"bench": "metaworld", "env_id": source["env_id"],
                                         "seed": seed, "teacher_model": "artificial",
                                         "model": "artificial", "provider": "artificial",
                                         "action_source": record["action_source"],
                                         "input_protocol": inputs.INPUT_PROTOCOL,
                                         "input_profile_hash": profile["hash"],
                                         "reasoning_source": "empty", "posthoc_annotation": False,
                                         "model_protocol": PROTOCOL,
                                         "output_source": "constructed_expert_xml",
                                         "supervision": "current_decision_only",
                                         "swift_loss_scale": "last_round+ignore_empty_think",
                                         "thinking_supervision": "answer_only",
                                         "source_episode": str(source_path.resolve()),
                                         "source_sha256": source_hash,
                                         "source_turn": turn,
                                         "template_profile": "two_tool_xml_sft.v1",
                                         "target_message_index": len(messages) - 1,
                                         "official_episode_success": True}})
            mcp_obs = _mcp_observation(observation)
            if success:
                terminal_frames = _save_frame(mcp_obs, directory, turn)
                record["terminal_frames"] = terminal_frames
                record["terminal_snapshot"] = inputs.snapshot(
                    mcp_obs, terminal_frames, directory, profile, f"obs-{turn}", turn)
                record["official_success_info"] = json.loads(json.dumps(
                    info, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
                record["success"] = True
                record["stop_reason"] = "official_success"
                break
            if terminated or truncated:
                record["stop_reason"] = "environment_done_without_success"
                break
            previous = current
            current = inputs.snapshot(mcp_obs, _save_frame(mcp_obs, directory, turn),
                                      directory, profile, f"obs-{turn}", turn)
            history.append(event)
        if record["stop_reason"] is None:
            record["stop_reason"] = ("control_budget_exhausted" if adaptive_expert
                                     else "source_exhausted_without_success")
        record["source_steps"] = used_steps
        if record["success"]:
            if adaptive_expert:
                actual_source_path = (output_root / "sources_live_compressed" / "metaworld" /
                                      task_id / f"seed-{seed}.json")
                _write_json(actual_source_path, {
                    "schema_version": "openeta.expert_action_source.v1",
                    "source_kind": "official_metaworld_scripted_policy_live_compressed",
                    "source_dataset": source["source_dataset"], "policy": POLICIES[task_id],
                    "bench": "metaworld", "task_id": task_id, "env_id": source["env_id"],
                    "seed": seed, "source_success": True,
                    "segments": [item["arguments"] for item in record["turns"]],
                    "original_episode": source["original_episode"],
                    "original_episode_sha256": source["original_episode_sha256"],
                    "compression_tolerance_linf": tolerance,
                    "start_protocol": "openeta_same_env_reset_plus_five_settle_steps",
                })
                record["source_kind"] = "official_metaworld_scripted_policy_live_compressed"
                record["source_path"] = str(actual_source_path.resolve())
                record["source_sha256"] = hashlib.sha256(actual_source_path.read_bytes()).hexdigest()
                for sample in samples:
                    sample["metadata"].update(source_episode=record["source_path"],
                                              source_sha256=record["source_sha256"])
            record["samples"] = len(samples)
            output = directory / "samples.jsonl"
            temporary = output.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n"
                                         for item in samples), encoding="utf-8")
            temporary.replace(output)
        return record
    finally:
        env.close()
        _write_json(directory / "episode.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-control-steps", type=int, default=400)
    parser.add_argument("--adaptive-expert", action="store_true")
    args = parser.parse_args()
    record = replay_one(args.source, args.output_root, args.task,
                        args.max_control_steps, args.adaptive_expert)
    print(json.dumps({"success": record["success"], "stop_reason": record["stop_reason"],
                      "turns": len(record["turns"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
