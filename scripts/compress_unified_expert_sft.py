"""Continuously compress accepted unified SFT episodes and verify real replay."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_expert_replay_success import audit_episode
from scripts.collect_unified_expert_sft import load_cases, read, write
from scripts.compress_expert_replays import candidate_source, group_actions
from scripts.expert_action_segmentation import budgeted_segments
from scripts.replay_expert_two_tool_sft import _export_samples, replay_one
from agent.tools.sim_mcp import SseSimulatorMcpTransport


def publish_verified(case, result, args):
    if args.publish_root is None:
        return
    root = args.publish_root
    if read(root / "snapshot/sft_train.json") != read(args.input_root / "snapshot/sft_train.json"):
        raise ValueError("publication root uses a different frozen SFT split")
    relative = Path("episodes") / case["bench"] / case["task_id"] / f"seed-{case['seed']}"
    source = args.output_root / relative
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not destination.is_symlink():
        raise ValueError("refusing to replace an existing episode directory")
    with (root / ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = root / "results" / (case["case_id"] + ".json")
        if previous.exists():
            existing = read(previous)
            if existing.get("success") is True and existing.get("compressed_turns", float("inf")) <= result["compressed_turns"]:
                return
        pending = destination.with_name(destination.name + f".next-{os.getpid()}")
        if pending.is_symlink():
            pending.unlink()
        pending.symlink_to(source.resolve(), target_is_directory=True)
        pending.replace(destination)
        if previous.exists():
            write(root / "result_attempts" / case["case_id"] / f"{time.time_ns()}.json", read(previous))
        write(previous, {**result, "published_from": str(args.output_root.resolve())})
        _export_samples(root)


def accepted_episodes(input_root, rows, bench):
    selected = []
    for case, task, reference in rows:
        if case["bench"] != bench:
            continue
        outcome = input_root / "results" / (case["case_id"] + ".json")
        if not outcome.exists() or read(outcome).get("success") is not True:
            continue
        path = input_root / "episodes" / bench / case["task_id"] / f"seed-{case['seed']}" / "episode.json"
        if not path.exists() or not (path.parent / "samples.jsonl").exists():
            continue
        original = read(path)
        if original.get("success") is not True or original.get("stop_reason") != "official_success":
            continue
        if original.get("unified_case") != case:
            raise ValueError(f"original episode differs from frozen SFT case: {case['case_id']}")
        selected.append((case, task, reference, path, original))
    return selected


def compress_one(case, task, reference, original_path, original, args):
    audit = audit_episode(original_path)
    if not audit["integrity_pass"]:
        raise ValueError(f"original integrity audit failed: {audit['errors']}")
    actions = [turn["arguments"]["action"] for turn in original["turns"]]
    if not actions or any(turn["arguments"].get("num_steps") != 1 for turn in original["turns"]):
        raise ValueError("input must be a successful uncompressed expert episode")
    attempts = []
    result_path = args.output_root / "results" / (case["case_id"] + ".json")
    tolerances = args.tolerances if args.segmentation == "greedy" else [0.0]
    for tolerance in tolerances:
        segments = group_actions(actions, tolerance)
        label = "ungrouped" if tolerance == -1 else f"tolerance-{tolerance:g}"
        if args.segmentation == "budgeted":
            hold_groups = (args.settle_steps + 9) // 10
            segments = budgeted_segments(actions, args.max_verified_turns - hold_groups,
                                         protection_radius=args.protection_radius)
            label = f"budgeted-{args.max_verified_turns}-radius-{args.protection_radius}-settle-{args.settle_steps}"
            offset = 0
            for segment in segments:
                window = actions[offset:offset + segment["num_steps"]]
                tolerance = max(tolerance, max(abs(value - center) for action in window
                                               for value, center in zip(action, segment["action"])))
                offset += segment["num_steps"]
        elif args.aggregation == "mean":
            offset = 0
            for segment in segments:
                window = actions[offset:offset + segment["num_steps"]]
                segment["action"] = [sum(values) / len(window) for values in zip(*window)]
                offset += segment["num_steps"]
            label = "mean-" + label
        hold_steps = args.settle_steps
        if args.max_verified_turns:
            hold_steps = min(hold_steps, max(0, args.max_verified_turns - len(segments)) * 10)
        while hold_steps > 0:
            length = min(10, hold_steps)
            segments.append({"action": [0, 0, 0, 0, 0, 0, actions[-1][-1]], "num_steps": length})
            hold_steps -= length
        if args.settle_steps and args.segmentation == "greedy":
            label += f"-settle-{args.settle_steps}"
        if args.max_verified_turns and len(segments) > args.max_verified_turns:
            attempts.append({"tolerance": tolerance, "candidate_turns": len(segments),
                             "success": False, "stop_reason": "candidate_exceeds_turn_target"})
            continue
        candidate = candidate_source(original_path, original, segments, tolerance)
        candidate["unified_case"] = case
        candidate["aggregation"] = args.aggregation
        candidate["segmentation_method"] = args.segmentation
        candidate["post_source_hold_steps"] = sum(segment["num_steps"] for segment in segments) - len(actions)
        candidate["protection_radius"] = args.protection_radius if args.segmentation == "budgeted" else None
        if reference is not None:
            candidate["unified_source_ref"] = reference
            candidate["source_integrity_verified"] = original.get("source_integrity_verified", False)
            if candidate["source_integrity_verified"] is not True:
                raise ValueError("reference source integrity was not verified")
        else:
            candidate["initial_state"] = original.get("expert_source_initial_state")
            candidate["start_protocol"] = original.get("source_start_protocol")
        candidate_path = args.output_root / "candidates" / label / case["bench"] / case["task_id"] / f"seed-{case['seed']}.json"
        write(candidate_path, candidate)
        attempt_root = args.output_root / "attempts" / label
        if case["bench"] == "metaworld":
            subprocess.run([str(ROOT / "sim/venvs/metaworld/bin/python"),
                            str(ROOT / "scripts/replay_compressed_metaworld_direct.py"),
                            "--source", str(candidate_path), "--output-root", str(attempt_root),
                            "--task", task["task"], "--max-control-steps", str(task["horizon"])],
                           check=True, timeout=1800, cwd=ROOT)
            episode_path = attempt_root / "episodes" / case["bench"] / case["task_id"] / f"seed-{case['seed']}" / "episode.json"
            record = read(episode_path)
        else:
            record = replay_one(SseSimulatorMcpTransport(args.sim_url), candidate_path,
                                task, attempt_root, task["horizon"])
            episode_path = attempt_root / "episodes" / case["bench"] / case["task_id"] / f"seed-{case['seed']}" / "episode.json"
        attempts.append({"tolerance": tolerance, "candidate_turns": len(segments),
                         "success": record.get("success", False), "stop_reason": record.get("stop_reason"),
                         "executed_turns": len(record.get("turns", [])), "episode": str(episode_path.resolve())})
        if record.get("success") is not True:
            continue
        if args.max_verified_turns and len(record["turns"]) > args.max_verified_turns:
            raise ValueError("verified episode exceeds requested turn target")
        audit = audit_episode(episode_path)
        if not audit["integrity_pass"]:
            raise ValueError(f"compressed integrity audit failed: {audit['errors']}")
        directory = args.output_root / "episodes" / case["bench"] / case["task_id"] / f"seed-{case['seed']}"
        directory.parent.mkdir(parents=True, exist_ok=True)
        if directory.exists():
            raise ValueError("compression output already exists without a recorded outcome")
        record["unified_case"] = case
        record["compression"] = {"original_episode": str(original_path.resolve()),
                                 "original_episode_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                                 "original_turns": len(actions), "verified_turns": len(record["turns"]),
                                 "tolerance": tolerance, "aggregation": args.aggregation,
                                 "segmentation_method": args.segmentation,
                                 "post_source_hold_steps": candidate["post_source_hold_steps"],
                                 "grouped": any(segment["num_steps"] > 1 for segment in segments)}
        write(episode_path, record)
        samples_path = episode_path.parent / "samples.jsonl"
        samples = [json.loads(line) for line in samples_path.read_text().splitlines()]
        for sample in samples:
            sample["metadata"].update(unified_case_id=case["case_id"], phase="sft_train",
                                      initial_state=case["initial_state"],
                                      split_revision=read(args.input_root / "snapshot/selection.v1.json").get("split_revision"),
                                      compression_tolerance=tolerance,
                                      original_episode=str(original_path.resolve()))
        temporary = samples_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples))
        temporary.replace(samples_path)
        directory.symlink_to(episode_path.parent.resolve(), target_is_directory=True)
        result = {"case_id": case["case_id"], "success": True, "original_turns": len(actions),
                  "compressed_turns": len(record["turns"]), "tolerance": tolerance,
                  "aggregation": args.aggregation,
                  "segmentation_method": args.segmentation,
                  "post_source_hold_steps": candidate["post_source_hold_steps"],
                  "grouped": record["compression"]["grouped"], "attempts": attempts}
        write(result_path, result)
        publish_verified(case, result, args)
        return result
    result = {"case_id": case["case_id"], "success": False, "original_turns": len(actions),
              "stop_reason": "all_compression_replays_unsuccessful", "attempts": attempts}
    write(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bench", choices=["metaworld", "libero", "robocasa", "maniskill"], required=True)
    parser.add_argument("--sim-url", default="http://127.0.0.1:8793/sse")
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.05, 0.02, 0.0, -1.0])
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--aggregation", choices=["first", "mean"], default="first")
    parser.add_argument("--max-verified-turns", type=int, default=0)
    parser.add_argument("--publish-root", type=Path)
    parser.add_argument("--segmentation", choices=["greedy", "budgeted"], default="greedy")
    parser.add_argument("--protection-radius", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--only-unresolved", action="store_true")
    args = parser.parse_args()
    if args.settle_steps < 0 or args.protection_radius < 0:
        parser.error("settling and protection must be nonnegative")
    if args.segmentation == "budgeted" and (args.bench != "libero" or args.max_verified_turns < 1 or args.aggregation != "mean"):
        parser.error("budgeted segmentation requires LIBERO, mean aggregation and a positive turn limit")
    if args.settle_steps and args.bench != "libero":
        parser.error("settling is currently audited only for LIBERO")
    if args.input_root.resolve() == args.output_root.resolve():
        parser.error("input and output must differ")
    if args.publish_root and args.publish_root.resolve() in {args.input_root.resolve(), args.output_root.resolve()}:
        parser.error("publication root must differ from input and working output")
    rows = load_cases(args.input_root / "snapshot")
    completed = 0
    consecutive_errors = 0
    while True:
        for case, task, reference, original_path, original in accepted_episodes(args.input_root, rows, args.bench):
            if args.only_unresolved and args.publish_root:
                published = args.publish_root / "results" / (case["case_id"] + ".json")
                if published.exists():
                    current = read(published)
                    if current.get("success") is True and current.get("compressed_turns", float("inf")) <= args.max_verified_turns:
                        continue
            result_path = args.output_root / "results" / (case["case_id"] + ".json")
            if result_path.exists():
                existing = read(result_path)
                if args.publish_root and existing.get("success") is True:
                    published = args.publish_root / "results" / (case["case_id"] + ".json")
                    if not published.exists() or read(published).get("published_from") != str(args.output_root.resolve()):
                        publish_verified(case, existing, args)
                continue
            if args.limit and completed >= args.limit:
                return
            completed += 1
            try:
                result = compress_one(case, task, reference, original_path, original, args)
                consecutive_errors = 0
            except Exception as error:
                result = {"case_id": case["case_id"], "success": False,
                          "status": "compression_error", "error": f"{type(error).__name__}: {error}"}
                write(result_path, result)
                consecutive_errors += 1
            print(json.dumps(result, ensure_ascii=False), flush=True)
            with (args.output_root / ".export.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                _export_samples(args.output_root)
            if consecutive_errors >= 3:
                raise RuntimeError("compression paused after three consecutive infrastructure errors")
        if not args.watch:
            return
        time.sleep(max(1, args.poll_seconds))


if __name__ == "__main__":
    main()
