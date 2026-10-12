#!/usr/bin/env python3
"""Collect verified MetaWorld scripted-expert XML SFT in the same OpenETA env."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.protocol import EnvObservation
from agent.runtime import two_tool_input as inputs
from agent.runtime.two_tool_xml import PROTOCOL, parse_decision, serialize_decision
from scripts.collect_two_tool_sft import _save_frame, _validate_manifest, _write_json
from scripts.export_two_tool_sft import swift_messages
from scripts.generate_metaworld_expert_actions import POLICIES
from scripts.replay_expert_two_tool_sft import SAMPLE_SCHEMA, SCHEMA, _export_samples
from sim.env_registry import _make_metaworld_direct
from sim.two_tool_contract import controller_contract
from sim.unified_env import UnifiedEnv


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mcp_observation(observation: dict) -> dict:
    return EnvObservation.from_dict(observation).to_mcp_dict()


def collect_one(row: dict, seed: int, root: Path, max_steps: int) -> dict:
    import metaworld
    from metaworld import policies

    task_id = row["task_id"]
    directory = root / "episodes" / "metaworld" / task_id / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    raw = _make_metaworld_direct(task_id, image_width=256, image_height=256, seed=seed)
    env = UnifiedEnv(raw, render_mode="rgb_array")
    expert = getattr(policies, POLICIES[task_id])()
    contract = {"action_dim": 4, "action_low": [-1.0] * 4, "action_high": [1.0] * 4}
    created = {**contract, "input_control_contract": controller_contract(env, "metaworld")}
    profile = inputs.make_profile("metaworld", task_id, created, contract)
    record = {"schema_version": SCHEMA, "model": "artificial", "teacher_model": "artificial",
              "provider": "artificial", "action_source": "expert_replay",
              "output_source": "constructed_expert_xml", "reasoning_source": "empty",
              "posthoc_annotation": False, "model_protocol": PROTOCOL,
              "input_protocol": inputs.INPUT_PROTOCOL, "input_profile": profile,
              "action_contract": contract, "bench": "metaworld", "task_id": task_id,
              "env_id": row["env_id"], "seed": seed,
              "environment_seed_protocol": raw._openeta_seed_protocol,
              "creation_seed": seed,
              "source_kind": "official_metaworld_scripted_policy_live_openeta",
              "source_dataset": "Farama-Foundation/Metaworld",
              "source_version": getattr(metaworld, "__version__", "unreported"),
              "source_policy": POLICIES[task_id], "success": False,
              "stop_reason": None, "turns": [], "samples": 0}
    samples, actions = [], []
    try:
        observation, _ = env.reset(seed=seed)
        if row.get("unified_case"):
            state = raw.unwrapped
            verdict = state.evaluate_state(state._get_obs(), np.zeros(4))[-1]
            record["unified_case"] = row["unified_case"]
            record["initial_state_audit"] = {
                "qpos": state.data.qpos.tolist(), "qvel": state.data.qvel.tolist(),
                "task_random_parameters": np.asarray(state._last_rand_vec).tolist(),
                "goal_position": np.asarray(state._target_pos).tolist(),
                "initial_success": bool(verdict.get("success", False)),
            }
            if record["initial_state_audit"]["initial_success"]:
                raise ValueError("frozen initial state is already successful")
        # Match OpenETA reset_env's five open-gripper settling steps.
        for _ in range(5):
            observation, _, terminated, truncated, _ = env.step([0.0, 0.0, 0.0, -1.0])
            if terminated or truncated:
                raise RuntimeError("episode ended during host settle")
        current = inputs.snapshot(_mcp_observation(observation),
                                  _save_frame(_mcp_observation(observation), directory, 0),
                                  directory, profile, "obs-0", None)
        previous = None
        history = []
        clipped_count = 0
        for offset in range(max_steps):
            turn = offset + 1
            raw_state = raw.unwrapped._get_obs()
            proposed = np.asarray(expert.get_action(raw_state), dtype=np.float64)
            if proposed.shape != (4,) or not np.all(np.isfinite(proposed)):
                raise ValueError("scripted expert returned an invalid action")
            bounded = np.clip(proposed, -1.0, 1.0)
            clipped_count += int(np.any(bounded != proposed))
            action = bounded.tolist()
            parameters = {"action": action, "num_steps": 1}
            xml = serialize_decision("step_env", parameters, reasoning="")
            parse_decision(xml, contract=contract, remaining_steps=max_steps - offset)
            package = inputs.build_input(task=inputs.task_description(task_id, row["task"]),
                profile=profile, current=current, previous=previous, history=history,
                turn=turn, max_turns=max(60, max_steps), remaining_steps=max_steps - offset)
            assistant = {"role": "assistant", "content": xml}
            messages, images = swift_messages(copy.deepcopy(package["messages"]) + [assistant])
            observation, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32))
            success = isinstance(info, dict) and info.get("success") == 1
            actions.append(action)
            event = {"turn": turn, "valid": True, "validation": {"valid": True},
                     "tool": "step_env", "arguments": parameters, "model_message": assistant,
                     "tool_request": {"name": "step_env", "arguments": {**parameters, "strict": True}},
                     "executed_control_steps": 1, "terminated": bool(terminated),
                     "truncated": bool(truncated), "official_success": bool(success),
                     "input_sha256": package["messages_sha256"]}
            record["turns"].append(event)
            samples.append({"schema_version": SAMPLE_SCHEMA,
                            "sample_id": f"metaworld/{task_id}/seed-{seed}:turn-{turn:04d}",
                            "episode_id": str(directory.resolve()), "split": "train",
                            "messages": messages, "images": images,
                            "metadata": {"bench": "metaworld", "env_id": row["env_id"],
                                         "seed": seed, "teacher_model": "artificial",
                                         "model": "artificial", "provider": "artificial",
                                         "action_source": "expert_replay",
                                         "input_protocol": inputs.INPUT_PROTOCOL,
                                         "input_profile_hash": profile["hash"],
                                         "reasoning_source": "empty", "posthoc_annotation": False,
                                         "model_protocol": PROTOCOL,
                                         "output_source": "constructed_expert_xml",
                                         "supervision": "current_decision_only",
                                         "swift_loss_scale": "last_round+ignore_empty_think",
                                         "thinking_supervision": "answer_only",
                                         "source_turn": turn, "template_profile": "two_tool_xml_sft.v1",
                                         "target_message_index": len(messages) - 1,
                                         "official_episode_success": True}})
            if success:
                terminal_mcp = _mcp_observation(observation)
                terminal_frames = _save_frame(terminal_mcp, directory, turn)
                record["terminal_frames"] = terminal_frames
                record["terminal_snapshot"] = inputs.snapshot(
                    terminal_mcp, terminal_frames, directory, profile,
                    f"obs-{turn}", turn)
                record["official_success_info"] = json.loads(json.dumps(
                    info, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)))
                record["success"] = True
                record["stop_reason"] = "official_success"
                break
            if terminated or truncated:
                record["stop_reason"] = "environment_done_without_success"
                break
            previous = current
            mcp_obs = _mcp_observation(observation)
            current = inputs.snapshot(mcp_obs, _save_frame(mcp_obs, directory, turn),
                                      directory, profile, f"obs-{turn}", turn)
            history.append(event)
        if record["stop_reason"] is None:
            record["stop_reason"] = "control_budget_exhausted"
        record["source_steps"] = len(actions)
        record["clipped_policy_actions"] = clipped_count
        if record["success"]:
            source_path = root / "sources_live" / "metaworld" / task_id / f"seed-{seed}.json"
            _write_json(source_path, {"schema_version": "openeta.expert_action_source.v1",
                "source_kind": record["source_kind"], "source_dataset": record["source_dataset"],
                "source_version": record["source_version"], "policy": POLICIES[task_id],
                "bench": "metaworld", "task_id": task_id, "env_id": row["env_id"],
                "seed": seed, "source_success": True, "actions": actions,
                "start_protocol": "openeta_same_env_reset_plus_five_settle_steps"})
            record["source_path"] = str(source_path.resolve())
            record["source_sha256"] = _file_hash(source_path)
            for sample in samples:
                sample["metadata"].update(source_episode=str(source_path.resolve()),
                                          source_sha256=record["source_sha256"])
            sample_path = directory / "samples.jsonl"
            temporary = sample_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in samples),
                                 encoding="utf-8")
            temporary.replace(sample_path)
            record["samples"] = len(samples)
        return record
    finally:
        env.close()
        _write_json(directory / "episode.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "SFT_data/configs/deepseek_xml_debug5.v1.json")
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / "SFT_data/configs/two_tool_test_holdout.v1.json")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "SFT_data/expert_replay_v1")
    parser.add_argument("--target-per-task", type=int, default=5)
    parser.add_argument("--max-attempts-per-task", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed-start", type=int, default=31000000)
    parser.add_argument("--task-id", default="")
    args = parser.parse_args()
    rows = [row for row in json.loads(args.manifest.read_text(encoding="utf-8"))["tasks"]
            if row["bench"] == "metaworld"]
    if args.task_id:
        rows = [row for row in rows if row["task_id"] == args.task_id]
    if not rows:
        raise ValueError("no matching MetaWorld task")
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    _validate_manifest(rows, holdout, require_full=False)
    if any(row["task_id"] not in POLICIES for row in rows):
        raise ValueError("task without official scripted policy")
    for task_index, row in enumerate(rows):
        accepted = len(list((args.output_root / "episodes" / "metaworld" / row["task_id"]).glob("seed-*/samples.jsonl")))
        for attempt in range(args.max_attempts_per_task):
            if accepted >= args.target_per_task:
                break
            seed = args.seed_start + task_index * 10000 + attempt
            if seed in holdout["reserved_seeds"]:
                raise ValueError("reserved test seed")
            episode_path = args.output_root / "episodes" / "metaworld" / row["task_id"] / f"seed-{seed}" / "episode.json"
            if episode_path.is_file():
                continue
            record = collect_one(row, seed, args.output_root, args.max_steps)
            if record["success"]:
                accepted += 1
                _export_samples(args.output_root)
            print(json.dumps({"task": row["task_id"], "seed": seed,
                              "success": record["success"], "turns": len(record["turns"]),
                              "accepted_for_task": accepted}), flush=True)
        if accepted < args.target_per_task:
            print(json.dumps({"task": row["task_id"], "status": "shortfall",
                              "accepted": accepted, "target": args.target_per_task}), flush=True)


if __name__ == "__main__":
    main()
