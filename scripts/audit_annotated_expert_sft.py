#!/usr/bin/env python3
"""Independently audit Gemini-annotated compressed expert SFT samples."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runtime.two_tool_xml import parse_decision
from scripts.collect_two_tool_sft import _write_json
from scripts.compress_expert_replays import QUOTAS, selected_episodes


def _lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=ROOT / "SFT_data/expert_replay_compressed_v1")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "SFT_data/expert_replay_compressed_reasoning_v1")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "SFT_data/expert_replay_compressed_reasoning_v1/reports/integrity_audit.v1.json")
    args = parser.parse_args()
    selected = selected_episodes(args.source_root, allow_partial=False)
    errors: list[str] = []
    bench_episodes: Counter[str] = Counter()
    bench_samples: Counter[str] = Counter()
    sample_ids: set[str] = set()
    for episode_path, episode in selected:
        bench, task, seed = episode["bench"], episode["task_id"], episode["seed"]
        bench_episodes[bench] += 1
        directory = args.output_root / "episodes" / bench / task / f"seed-{seed}"
        source_path = episode_path.parent / "samples.jsonl"
        target_path = directory / "samples.jsonl"
        report_path = directory / "report.json"
        if not target_path.is_file() or not report_path.is_file():
            errors.append(f"missing annotated episode: {bench}/{task}/{seed}")
            continue
        source = _lines(source_path)
        target = _lines(target_path)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if report.get("source_samples_sha256") != source_hash or report.get("turns") != len(source):
            errors.append(f"stale report: {bench}/{task}/{seed}")
        if len(source) != episode["samples"] or len(target) != len(source):
            errors.append(f"sample count mismatch: {bench}/{task}/{seed}")
            continue
        for index, (before, after) in enumerate(zip(source, target), 1):
            label = f"{bench}/{task}/{seed}/turn-{index}"
            if before["sample_id"] != after["sample_id"] or after["sample_id"] in sample_ids:
                errors.append(f"sample ID missing or duplicated: {label}")
            sample_ids.add(after["sample_id"])
            old = parse_decision(before["messages"][-1]["content"],
                                 contract=episode["action_contract"])["decision"]
            new = parse_decision(after["messages"][-1]["content"],
                                 contract=episode["action_contract"])["decision"]
            if {key: value for key, value in old.items() if key != "reasoning"} != {
                    key: value for key, value in new.items() if key != "reasoning"}:
                errors.append(f"fixed tool decision changed: {label}")
            if not str(new.get("reasoning", "")).strip():
                errors.append(f"empty reasoning: {label}")
            metadata = after["metadata"]
            if metadata.get("model") != "artificial" or metadata.get("reasoning_model") != "gemini-3.6-flash":
                errors.append(f"incorrect provenance: {label}")
            if metadata.get("source_sample_sha256") != hashlib.sha256(
                    json.dumps(before, ensure_ascii=False, sort_keys=True).encode()).hexdigest():
                errors.append(f"sample hash mismatch: {label}")
        bench_samples[bench] += len(target)
    for bench, quota in QUOTAS.items():
        if bench_episodes[bench] != quota:
            errors.append(f"episode quota mismatch: {bench}")
        export = args.output_root / "exports" / bench / "train.v1.jsonl"
        if not export.is_file() or len(export.read_text(encoding="utf-8").splitlines()) != bench_samples[bench]:
            errors.append(f"export count mismatch: {bench}")
    result = {
        "schema_version": "openeta.reasoning_integrity_audit.v1",
        "source_episodes": len(selected),
        "annotated_episodes_by_bench": dict(bench_episodes),
        "annotated_samples_by_bench": dict(bench_samples),
        "annotated_samples_total": sum(bench_samples.values()),
        "errors": errors,
        "pass": not errors and len(selected) == 100 and sum(bench_samples.values()) == 9385,
    }
    _write_json(args.report, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
