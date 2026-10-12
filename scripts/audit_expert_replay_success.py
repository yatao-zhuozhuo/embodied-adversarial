#!/usr/bin/env python3
"""Audit accepted expert-replay SFT against stored environment receipts and actions.

This is an integrity audit, not a substitute for visually inspecting terminal
states or independently re-running the official task checker.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runtime.two_tool_xml import parse_decision


def audit_episode(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    sample_path = path.with_name("samples.jsonl")
    turns = record.get("turns") or []
    errors: list[str] = []
    warnings: list[str] = []
    if record.get("success") is not True or record.get("stop_reason") != "official_success":
        errors.append("not_marked_official_success")
    if not turns or turns[-1].get("official_success") is not True:
        errors.append("last_turn_missing_official_success")
    if any(turn.get("official_success") is True for turn in turns[:-1]):
        errors.append("continued_after_first_success")
    if not sample_path.is_file():
        errors.append("samples_missing")
    source_path = Path(record.get("source_path") or "/nonexistent")
    if not source_path.is_file():
        errors.append("source_missing")
    elif hashlib.sha256(source_path.read_bytes()).hexdigest() != record.get("source_sha256"):
        errors.append("source_sha256_mismatch")
    samples = []
    if sample_path.is_file():
        try:
            samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
        except (ValueError, OSError):
            errors.append("samples_unreadable")
    if len(samples) != len(turns) or record.get("samples") != len(samples):
        errors.append("turn_sample_count_mismatch")
    for index, (turn, sample) in enumerate(zip(turns, samples), 1):
        prefix = f"turn_{index}:"
        try:
            decision = parse_decision(sample["messages"][-1]["content"],
                contract=record["action_contract"])["decision"]
            if decision["name"] != turn["tool"] or decision["parameters"] != turn["arguments"]:
                errors.append(prefix + "xml_action_differs_from_executed_action")
            if sample["metadata"].get("official_episode_success") is not True:
                errors.append(prefix + "sample_success_metadata_missing")
            if sample["metadata"].get("model") != "artificial":
                errors.append(prefix + "unexpected_action_model")
            if sum(message["content"].count("<image>") for message in sample["messages"]) != len(sample["images"]):
                errors.append(prefix + "image_placeholder_count_mismatch")
            if not all(Path(image).is_file() for image in sample["images"]):
                errors.append(prefix + "student_visible_image_missing")
        except (KeyError, ValueError, TypeError, IndexError) as exc:
            errors.append(prefix + "malformed_sample_" + type(exc).__name__)
    terminal_prefix = f"{len(turns):04d}-"
    terminal_frames = sorted(path.parent.glob("frames/" + terminal_prefix + "*.png"))
    if not terminal_frames:
        warnings.append("terminal_post_success_frame_not_captured")
    executed_control_steps = sum(turn.get("executed_control_steps", turn.get("arguments", {}).get("num_steps", 1))
                                 for turn in turns)
    if record.get("source_steps", 0) > executed_control_steps:
        warnings.append("success_before_source_demo_end")
    return {"bench": record.get("bench"), "task_id": record.get("task_id"),
            "seed": record.get("seed"), "episode": str(path.resolve()),
            "turns": len(turns), "source_steps": record.get("source_steps"),
            "source_kind": record.get("source_kind"),
            "source_demo_index": record.get("reference_demo_index"),
            "terminal_frames": [str(item.resolve()) for item in terminal_frames],
            "integrity_pass": not errors, "errors": errors, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "SFT_data/expert_replay_v1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "SFT_data/expert_replay_v1/reports/success_integrity_audit.v1.json")
    args = parser.parse_args()
    paths = sorted(args.root.glob("episodes/*/*/seed-*/samples.jsonl"))
    items = [audit_episode(path.with_name("episode.json")) for path in paths]
    by_bench = Counter(item["bench"] for item in items)
    by_task = Counter((item["bench"], item["task_id"]) for item in items)
    warning_counts = Counter(warning for item in items for warning in item["warnings"])
    report = {"schema_version": "openeta.expert_replay_success_integrity_audit.v1",
              "scope": "stored_action_receipt_and_sample_consistency_only",
              "does_not_prove": ["terminal visual correctness", "persistence after success",
                                 "independent environment re-run"],
              "episodes": len(items), "integrity_pass": sum(item["integrity_pass"] for item in items),
              "integrity_fail": sum(not item["integrity_pass"] for item in items),
              "by_bench": dict(by_bench),
              "by_task": {f"{bench}/{task}": count for (bench, task), count in by_task.items()},
              "warning_counts": dict(warning_counts), "items": items}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({key: report[key] for key in
                      ("episodes", "integrity_pass", "integrity_fail", "by_bench", "warning_counts")},
                     ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
