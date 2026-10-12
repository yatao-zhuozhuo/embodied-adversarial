#!/usr/bin/env python3
"""Replay expert actions through OpenETA and export successful artificial XML SFT."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import numpy as np
from pathlib import Path
import sys
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.runtime import two_tool_input as inputs
from agent.runtime.two_tool_xml import PROTOCOL, action_contract, parse_decision, serialize_decision
from agent.tools.sim_mcp import SseSimulatorMcpTransport
from scripts.collect_two_tool_sft import _official_success, _save_frame, _validate_manifest, _write_json
from scripts.export_two_tool_sft import swift_messages
from scripts.maniskill_demo_actions import canonicalize_pd_ee_delta_pose

SCHEMA = "openeta.artificial_expert_replay.v1"
SAMPLE_SCHEMA = "openeta.sft.artificial_xml_decision.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_files(root: Path) -> list[Path]:
    return sorted(root.glob("*/*/seed-*.json"))


def _export_samples(root: Path) -> dict[str, int]:
    by_bench: dict[str, list[str]] = {}
    for path in sorted(root.glob("episodes/*/*/seed-*/samples.jsonl")):
        bench = path.parts[-4]
        by_bench.setdefault(bench, []).extend(path.read_text(encoding="utf-8").splitlines(True))
    counts = {}
    for bench, lines in by_bench.items():
        output = root / "exports" / bench / "train.v1.jsonl"
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(lines), encoding="utf-8")
        temporary.replace(output)
        counts[bench] = len(lines)
    return counts


def replay_one(transport: SseSimulatorMcpTransport, source_path: Path, row: dict,
               output_root: Path, max_control_steps: int) -> dict:
    source = json.loads(source_path.read_text(encoding="utf-8"))
    bench, task_id, seed = source["bench"], source["task_id"], source["seed"]
    reference_index = source.get("reference_demo_index")
    if reference_index is None and (source.get("source_success") is not True or
                                    not (source.get("actions") or source.get("segments"))):
        raise ValueError("source must contain a successful action trajectory or official reference index")
    if (bench, task_id, source["env_id"]) != (row["bench"], row["task_id"], row["env_id"]):
        raise ValueError("source does not match frozen task manifest")
    if reference_index is None and len(source.get("actions", [])) > max_control_steps:
        raise ValueError("source exceeds control-step budget")
    directory = output_root / "episodes" / bench / task_id / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    session = f"artificial-expert-{uuid4().hex}"
    created = transport.call_tool("create_env", {"env_id": row["env_id"], "session_id": session,
                          "seed": seed, "image_width": 256, "image_height": 256}, timeout_s=180)
    if created.get("error"):
        raise RuntimeError(f"create_env: {created['error']}")
    base = {"handle": created["handle"], "session_id": created.get("session_id", session)}
    record = {"schema_version": SCHEMA, "model": "artificial", "teacher_model": "artificial",
              "provider": "artificial", "action_source": "expert_replay",
              "output_source": "constructed_expert_xml", "reasoning_source": "empty",
              "posthoc_annotation": False, "model_protocol": PROTOCOL,
              "input_protocol": inputs.INPUT_PROTOCOL, "bench": bench, "task_id": task_id,
              "env_id": row["env_id"], "seed": seed, "source_path": str(source_path.resolve()),
              "source_sha256": _sha256(source_path), "source_kind": source["source_kind"],
              "source_dataset": source["source_dataset"], "source_version": source.get("source_version"),
              "source_policy": source.get("policy"), "reference_demo_index": reference_index,
              "source_steps": len(source.get("actions", [])),
              "success": False, "stop_reason": None, "turns": [], "samples": 0}
    samples = []
    try:
        contract = action_contract(created)
        profile = inputs.make_profile(bench, task_id, created, contract)
        record["action_contract"] = contract
        record["input_profile"] = profile
        reset_args = {**base, "seed": seed}
        if reference_index is not None:
            reset_args["reference_demo_index"] = reference_index
        reset = transport.call_tool("reset_env", reset_args, timeout_s=180)
        observed = transport.call_tool("observe_env", base, timeout_s=180)
        if reset.get("error") or observed.get("error"):
            raise RuntimeError(f"reset/observe: {reset.get('error') or observed.get('error')}")
        if source.get("unified_case") and source.get("unified_source_ref"):
            reference = source["unified_source_ref"]
            restored = reset.get("_reference_demo") or {}
            expected_path = (REPO_ROOT / reference["dataset_uri"]).resolve()
            if (Path(restored.get("dataset", "")).resolve() != expected_path or
                    restored.get("episode_index") != reference["reference_demo_index"]):
                raise ValueError("restored reference differs from frozen SFT case")
            if bench == "maniskill" and restored.get("episode_seed") != reference["episode_seed"]:
                raise ValueError("restored cached episode seed differs from frozen SFT case")
            if restored.get("initial_success") is not False:
                raise ValueError("reference initial success must be checked and false")
            if bench != "maniskill":
                restoration_error = restored.get("restored_state_max_abs_error", float("inf"))
                if not np.isfinite(restoration_error) or restoration_error > 1e-5:
                    raise ValueError("reference physics restoration differs from frozen state")
            record["unified_case"] = source["unified_case"]
            record["unified_source_ref"] = reference
            record["source_integrity_verified"] = source["source_integrity_verified"]
        elif source.get("unified_case"):
            case = source["unified_case"]
            if (case["phase"] != "sft_train" or case["initial_state"]["kind"] != "native_seed"
                    or case["seed"] != seed or contract["action_dim"] != row["expected_action_dim"]):
                raise ValueError("native expert source differs from frozen SFT case")
            record["unified_case"] = case
            record["expert_source_initial_state"] = source.get("initial_state")
            record["initial_state_audit"] = {"scope": "settled_openeta_observation_only",
                                             "robot": observed.get("robot"), "objects": observed.get("objects")}
            record["source_start_protocol"] = source.get("start_protocol")
        if "segments" in source:
            segments = source["segments"]
            if not isinstance(segments, list) or not segments or any(
                not isinstance(segment, dict) or
                type(segment.get("num_steps")) is not int or
                not 1 <= segment["num_steps"] <= 10 or
                not isinstance(segment.get("action"), list)
                for segment in segments
            ):
                raise ValueError("invalid source action segments")
        else:
            actions = reset.get("_reference_actions") if reference_index is not None else source["actions"]
            if not isinstance(actions, list) or not actions:
                raise ValueError("reference actions missing")
            segments = [{"action": action, "num_steps": 1} for action in actions]
        hold_steps = source.get("post_source_hold_steps", 0)
        if hold_steps and "segments" not in source:
            if type(hold_steps) is not int or hold_steps < 0:
                raise ValueError("invalid post-source hold budget")
            hold = [0.0] * contract["action_dim"]
            if bench in {"libero", "robocasa"}:
                hold[6] = segments[-1]["action"][6]
            elif bench == "maniskill" and len(hold) == 7:
                hold[-1] = segments[-1]["action"][-1]
            segments.extend({"action": list(hold), "num_steps": 1} for _ in range(hold_steps))
        source_steps = sum(segment["num_steps"] for segment in segments)
        if source_steps > max_control_steps and not source.get("unified_case"):
            raise ValueError(f"reference action length outside 1..{max_control_steps}: {source_steps}")
        record["source_steps"] = source_steps
        if reference_index is not None:
            record["reference_demo"] = reset.get("_reference_demo")
        task = inputs.task_description(task_id, reset.get("task") or row["task"])
        current = inputs.snapshot(observed, _save_frame(observed, directory, 0), directory,
                                  profile, "obs-0", None)
        previous = None
        history = []
        max_turns = max(60, len(segments))
        clipped_count = 0
        control_steps_used = 0
        for offset, segment in enumerate(segments):
            if control_steps_used >= max_control_steps:
                record["stop_reason"] = "control_budget_exhausted"
                break
            turn = offset + 1
            action = segment["action"]
            num_steps = min(segment["num_steps"], max_control_steps - control_steps_used)
            if source.get("action_conversion") == "maniskill_pd_ee_delta_pose_controller":
                original = np.asarray(action, dtype=float)
                action = canonicalize_pd_ee_delta_pose(action)
                clipped_count += int(np.any(np.asarray(action) != original))
            elif source.get("clip_actions_to_contract"):
                original = np.asarray(action, dtype=float)
                bounded = np.clip(original, contract["action_low"], contract["action_high"])
                clipped_count += int(np.any(bounded != original))
                action = bounded.tolist()
            parameters = {"action": action, "num_steps": num_steps}
            xml = serialize_decision("step_env", parameters, reasoning="")
            decision = parse_decision(xml, contract=contract,
                                      remaining_steps=max_control_steps - control_steps_used)["decision"]
            package = inputs.build_input(task=task, profile=profile, current=current,
                                         previous=previous, history=history, turn=turn,
                                         max_turns=max_turns, remaining_steps=max_control_steps - control_steps_used)
            assistant = {"role": "assistant", "content": xml}
            messages, images = swift_messages(copy.deepcopy(package["messages"]) + [assistant])
            receipt = transport.call_tool("step_env", {**base, **parameters, "strict": True}, timeout_s=180)
            executed_steps = receipt.get("executed_control_steps")
            if receipt.get("error") or type(executed_steps) is not int or not 1 <= executed_steps <= num_steps:
                raise RuntimeError(f"step_env at turn {turn}: {receipt.get('error') or receipt}")
            control_steps_used += executed_steps
            event = {"turn": turn, "valid": True, "validation": {"valid": True},
                     "tool": "step_env", "arguments": parameters, "model_message": assistant,
                     "tool_request": {"name": "step_env", "arguments": {**parameters, "strict": True}},
                     "executed_control_steps": executed_steps, "terminated": bool(receipt.get("terminated")),
                     "truncated": bool(receipt.get("truncated")), "official_success": _official_success(receipt, bench=bench)}
            record["turns"].append({**event, "receipt_success": event["official_success"],
                                    "input_sha256": package["messages_sha256"],
                                    "input_profile_hash": profile["hash"]})
            samples.append({"schema_version": SAMPLE_SCHEMA,
                            "sample_id": f"{bench}/{task_id}/seed-{seed}:turn-{turn:04d}",
                            "episode_id": str(directory.resolve()), "split": "train",
                            "messages": messages, "images": images,
                            "metadata": {"bench": bench, "env_id": row["env_id"], "seed": seed,
                                         "teacher_model": "artificial", "model": "artificial",
                                         "provider": "artificial", "action_source": "expert_replay",
                                         "input_protocol": inputs.INPUT_PROTOCOL,
                                         "input_profile_hash": profile["hash"],
                                         "reasoning_source": "empty", "posthoc_annotation": False,
                                         "model_protocol": PROTOCOL,
                                         "output_source": "constructed_expert_xml",
                                         "supervision": "current_decision_only",
                                         "swift_loss_scale": "last_round+ignore_empty_think",
                                         "thinking_supervision": "answer_only",
                                         "source_episode": str(source_path.resolve()),
                                         "source_sha256": record["source_sha256"],
                                         "source_turn": turn, "template_profile": "two_tool_xml_sft.v1",
                                         "target_message_index": len(messages) - 1,
                                         "official_episode_success": True}})
            if event["official_success"]:
                terminal_observation = receipt.get("observation")
                if not isinstance(terminal_observation, dict):
                    raise RuntimeError("official success receipt lacks terminal observation")
                terminal_frames = _save_frame(terminal_observation, directory, turn)
                record["terminal_frames"] = terminal_frames
                record["terminal_snapshot"] = inputs.snapshot(
                    terminal_observation, terminal_frames, directory, profile,
                    f"obs-{turn}", turn)
                record["official_success_info"] = receipt.get("info")
                record["success"] = True
                record["stop_reason"] = "official_success"
                break
            if event["terminated"] or event["truncated"]:
                record["stop_reason"] = "environment_done_without_success"
                break
            if offset + 1 < len(segments):
                previous = current
                current = inputs.snapshot(receipt["observation"],
                    _save_frame(receipt["observation"], directory, turn), directory,
                    profile, f"obs-{turn}", turn)
            history.append(event)
        if record["stop_reason"] is None:
            record["stop_reason"] = "source_exhausted_without_success"
        record["clipped_source_actions"] = clipped_count
        if record["success"]:
            record["samples"] = len(samples)
            sample_path = directory / "samples.jsonl"
            temporary = sample_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in samples),
                                 encoding="utf-8")
            temporary.replace(sample_path)
        return record
    finally:
        transport.call_tool("close_env", base, timeout_s=180)
        _write_json(directory / "episode.json", record)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "SFT_data/expert_replay_v1/sources")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "SFT_data/expert_replay_v1")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "SFT_data/configs/deepseek_xml_debug5.v1.json")
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / "SFT_data/configs/two_tool_test_holdout.v1.json")
    parser.add_argument("--sim-url", default="http://127.0.0.1:8769/sse")
    parser.add_argument("--max-control-steps", type=int, default=400)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-sources", type=int, default=0,
                        help="Limit source attempts, including infrastructure errors")
    parser.add_argument("--bench", default="", help="Replay only one benchmark")
    parser.add_argument("--task", default="", help="Replay only one task ID")
    parser.add_argument("--skip-failed", action="store_true", help="Skip already attempted unsuccessful sources")
    parser.add_argument("--target-total", type=int, default=0,
                        help="Stop when this many verified episodes exist across all benches")
    parser.add_argument("--target-bench", type=int, default=0,
                        help="Stop when this many verified episodes exist for the selected bench")
    parser.add_argument("--target-task", type=int, default=0,
                        help="Stop when this many verified episodes exist for the selected task")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    rows = {row["env_id"]: row for row in manifest["tasks"]}
    _validate_manifest(list(rows.values()), holdout, require_full=False)
    transport = SseSimulatorMcpTransport(args.sim_url)
    completed = 0
    attempted = 0
    for source_path in _source_files(args.source_root):
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if args.bench and source["bench"] != args.bench:
            continue
        if args.task and source["task_id"] != args.task:
            continue
        if args.target_total and sum(1 for _ in args.output_root.glob("episodes/*/*/seed-*/samples.jsonl")) >= args.target_total:
            break
        if args.target_bench and sum(1 for _ in (args.output_root / "episodes" / source["bench"]).glob("*/seed-*/samples.jsonl")) >= args.target_bench:
            break
        if args.target_task and sum(1 for _ in (args.output_root / "episodes" / source["bench"] / source["task_id"]).glob("seed-*/samples.jsonl")) >= args.target_task:
            break
        row = rows.get(source["env_id"])
        if row is None or source["seed"] in holdout["reserved_seeds"]:
            raise ValueError(f"source outside train manifest: {source_path}")
        episode_path = args.output_root / "episodes" / source["bench"] / source["task_id"] / f"seed-{source['seed']}" / "episode.json"
        if episode_path.is_file():
            existing = json.loads(episode_path.read_text(encoding="utf-8"))
            if existing.get("success") is True:
                if existing.get("source_sha256") != _sha256(source_path):
                    raise ValueError(f"accepted source changed after export: {source_path}")
                continue
            if args.skip_failed:
                continue
        if args.max_sources and attempted >= args.max_sources:
            break
        attempted += 1
        try:
            record = replay_one(transport, source_path, row, args.output_root, args.max_control_steps)
        except Exception as error:
            print(json.dumps({"source": str(source_path), "status": "infrastructure_or_replay_error",
                              "error": f"{type(error).__name__}: {error}"}), flush=True)
            continue
        print(json.dumps({"source": str(source_path), "success": record["success"],
                          "stop_reason": record["stop_reason"], "turns": len(record["turns"])},
                         ensure_ascii=False), flush=True)
        completed += 1
        if args.limit and completed >= args.limit:
            break
    counts = _export_samples(args.output_root)
    print(json.dumps({"status": "finished_available_sources", "exported_turns": counts}), flush=True)


if __name__ == "__main__":
    main()
