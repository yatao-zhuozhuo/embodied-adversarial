#!/usr/bin/env python3
"""Validate bounded-rollout and engine-lifecycle evidence from an isolated canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--per-turn-limit", type=int, default=1024)
    parser.add_argument("--trajectory-limit", type=int, default=24576)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    violations: list[str] = []
    episodes: list[dict[str, Any]] = []
    for path in sorted(args.artifact_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        info = payload.get("final_info") if isinstance(payload, dict) else None
        if not isinstance(info, dict) or info.get("role") not in {"alice", "bob"}:
            continue
        counts = [int(value) for value in info.get("turn_token_counts", [])]
        total = int(info.get("generated_tokens_total", sum(counts)))
        if counts and total != sum(counts):
            violations.append(f"{path}: cumulative token count differs from turns")
        if any(value > args.per_turn_limit for value in counts):
            violations.append(f"{path}: per-turn limit exceeded")
        if total > args.trajectory_limit:
            violations.append(f"{path}: trajectory limit exceeded")
        episodes.append({
            "path": str(path),
            "role": info.get("role"),
            "turns": len(counts),
            "generated_tokens": total,
            "termination_reason": info.get("termination_reason"),
        })

    timings: list[dict[str, Any]] = []
    for path in sorted(args.artifact_root.rglob("phase_timing.rank-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("event") == "rollout":
                timings.append(row)
                if not row.get("vllm_is_sleeping_after", False):
                    violations.append(f"{path}: vLLM was not sleeping after rollout")

    report = {
        "schema_version": "openeta.bounded_probe.v1",
        "passed": bool(episodes) and bool(timings) and not violations,
        "episodes": episodes,
        "timing_records": len(timings),
        "violations": violations,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
