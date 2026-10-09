#!/usr/bin/env python3
"""Compress 100 accepted expert trajectories and independently replay each one.

The original episodes are read-only. Candidates are not exported as SFT until
the OpenETA environment returns official success after compressed execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tools.sim_mcp import SseSimulatorMcpTransport
from scripts.collect_two_tool_sft import _validate_manifest
from scripts.replay_expert_two_tool_sft import _export_samples, replay_one


QUOTAS = {"metaworld": 25, "libero": 50, "robocasa": 13, "maniskill": 12}


def group_actions(actions: list[list[float]], tolerance: float) -> list[dict]:
    """Hold the first action for at most 10 nearby control steps per call."""
    if tolerance == -1:
        return [{"action": action, "num_steps": 1} for action in actions]
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    result = []
    position = 0
    while position < len(actions):
        first = actions[position]
        end = position + 1
        while end < min(position + 10, len(actions)):
            candidate = actions[end]
            if len(candidate) != len(first) or max(
                abs(float(left) - float(right)) for left, right in zip(first, candidate)
            ) > tolerance:
                break
            end += 1
        result.append({"action": first, "num_steps": end - position})
        position = end
    return result


def selected_episodes(input_root: Path, *, allow_partial: bool) -> list[tuple[Path, dict]]:
    by_bench = {bench: [] for bench in QUOTAS}
    for path in sorted(input_root.glob("episodes/*/*/seed-*/episode.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("success") is not True or not (path.parent / "samples.jsonl").is_file():
            continue
        bench = record.get("bench")
        if bench in by_bench:
            by_bench[bench].append((path, record))
    missing = {bench: quota - len(by_bench[bench]) for bench, quota in QUOTAS.items()
               if len(by_bench[bench]) < quota}
    if missing and not allow_partial:
        raise ValueError(f"100 original successful trajectories not ready; shortfall: {missing}")
    return [item for bench, quota in QUOTAS.items() for item in by_bench[bench][:quota]]


def candidate_source(path: Path, original: dict, segments: list[dict], tolerance: float) -> dict:
    source = {
        "schema_version": "openeta.compressed_expert_action_source.v1",
        "bench": original["bench"], "task_id": original["task_id"],
        "env_id": original["env_id"], "seed": original["seed"],
        "source_kind": "compressed_verified_expert_replay_candidate",
        "source_dataset": original["source_dataset"],
        "source_success": True, "segments": segments,
        "compression_tolerance_linf": tolerance,
        "original_episode": str(path.resolve()),
        "original_episode_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if original.get("reference_demo_index") is not None:
        source["reference_demo_index"] = original["reference_demo_index"]
    return source


def write_candidate(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"Existing candidate differs: {path}")
    else:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "SFT_data/expert_replay_v1")
    parser.add_argument("--output-root", type=Path, default=ROOT / "SFT_data/expert_replay_compressed_v1")
    parser.add_argument("--manifest", type=Path, default=ROOT / "SFT_data/configs/deepseek_xml_debug5.v1.json")
    parser.add_argument("--holdout", type=Path, default=ROOT / "SFT_data/configs/two_tool_test_holdout.v1.json")
    parser.add_argument("--sim-url", default="http://127.0.0.1:8769/sse")
    parser.add_argument("--maniskill-sim-url", default="",
                        help="Optional dedicated ManiSkill server with matched PhysX backend")
    parser.add_argument("--metaworld-python", type=Path,
                        default=ROOT / "sim/venvs/metaworld/bin/python",
                        help="Python with the original direct MetaWorld environment")
    parser.add_argument("--bench", default="", help="Replay only this benchmark")
    parser.add_argument("--task", default="", help="Replay only this task ID")
    parser.add_argument("--limit", type=int, default=0, help="Limit selected episode attempts")
    parser.add_argument("--max-control-steps", type=int, default=400)
    parser.add_argument("--tolerances", type=float, nargs="+", default=[0.05, 0.02, 0.0, -1.0],
                        help="Use -1 for an ungrouped diagnostic fallback")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--expected-episodes", type=int, default=100,
                        help="Require this many selected source episodes (88 for the current partial batch)")
    args = parser.parse_args()
    selected = selected_episodes(args.input_root, allow_partial=args.allow_partial)
    if len(selected) != args.expected_episodes:
        parser.error(f"Expected {args.expected_episodes} source episodes, selected {len(selected)}")
    if args.bench:
        selected = [item for item in selected if item[1]["bench"] == args.bench]
    if args.task:
        selected = [item for item in selected if item[1]["task_id"] == args.task]
    if args.limit:
        selected = selected[:args.limit]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    rows = {row["env_id"]: row for row in manifest["tasks"]}
    _validate_manifest(list(rows.values()), holdout, require_full=False)
    transport = None if args.prepare_only else SseSimulatorMcpTransport(args.sim_url)
    maniskill_transport = (SseSimulatorMcpTransport(args.maniskill_sim_url)
                           if not args.prepare_only and args.maniskill_sim_url else transport)
    summary = {"selected": len(selected), "verified": 0, "failed": 0,
               "original_turns": 0, "candidate_turns_first_tolerance": 0,
               "verified_turns": 0, "by_bench": {}}
    for original_path, original in selected:
        actions = [turn["arguments"]["action"] for turn in original["turns"]]
        if not actions or any(turn["arguments"].get("num_steps") != 1 for turn in original["turns"]):
            raise ValueError(f"Expected one control step per original turn: {original_path}")
        summary["original_turns"] += len(actions)
        bench = original["bench"]
        summary["by_bench"][bench] = summary["by_bench"].get(bench, 0) + 1
        output_episode = args.output_root / "episodes" / bench / original["task_id"] / f"seed-{original['seed']}" / "episode.json"
        if output_episode.is_file():
            existing = json.loads(output_episode.read_text(encoding="utf-8"))
            if existing.get("success") is True and (output_episode.parent / "samples.jsonl").is_file():
                summary["verified"] += 1
                summary["verified_turns"] += len(existing["turns"])
                continue
        for index, tolerance in enumerate(args.tolerances):
            segments = group_actions(actions, tolerance)
            if index == 0:
                summary["candidate_turns_first_tolerance"] += len(segments)
            candidate_label = "ungrouped" if tolerance == -1 else f"tolerance-{tolerance:g}"
            candidate_path = (args.output_root / "candidates" / candidate_label /
                              bench / original["task_id"] / f"seed-{original['seed']}.json")
            write_candidate(candidate_path, candidate_source(original_path, original, segments, tolerance))
            if args.prepare_only:
                continue
            try:
                if bench == "metaworld":
                    result = subprocess.run([
                        str(args.metaworld_python),
                        str(ROOT / "scripts/replay_compressed_metaworld_direct.py"),
                        "--source", str(candidate_path), "--output-root", str(args.output_root),
                        "--task", rows[original["env_id"]]["task"],
                        "--max-control-steps", str(args.max_control_steps),
                        "--adaptive-expert",
                    ], capture_output=True, text=True,
                       env={**os.environ, "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
                    if result.returncode:
                        raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])
                    output_episode = (args.output_root / "episodes" / bench /
                                      original["task_id"] / f"seed-{original['seed']}" / "episode.json")
                    record = json.loads(output_episode.read_text(encoding="utf-8"))
                else:
                    selected_transport = maniskill_transport if bench == "maniskill" else transport
                    record = replay_one(selected_transport, candidate_path, rows[original["env_id"]],
                                        args.output_root, args.max_control_steps)
            except Exception as error:
                print(json.dumps({"episode": str(original_path), "tolerance": tolerance,
                                  "error": f"{type(error).__name__}: {error}"}), flush=True)
                continue
            if record["success"]:
                summary["verified"] += 1
                summary["verified_turns"] += len(record["turns"])
                print(json.dumps({"episode": str(original_path), "tolerance": tolerance,
                                  "verified_turns": len(record["turns"])}), flush=True)
                break
        else:
            if not args.prepare_only:
                summary["failed"] += 1
    if not args.prepare_only:
        summary["exported_samples"] = _export_samples(args.output_root)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
