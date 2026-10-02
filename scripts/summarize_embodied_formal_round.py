#!/usr/bin/env python3
"""Write a compact, machine-readable summary for one formal self-play round."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--alice-rollouts", type=Path, required=True)
    parser.add_argument("--bob-eval", type=Path, required=True)
    parser.add_argument("--bob-dataset-mode", choices=("new_alice", "bootstrap_fallback"), required=True)
    parser.add_argument("--alice-adapter", type=Path, required=True)
    parser.add_argument("--bob-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _alice_infos(root: Path) -> list[dict[str, Any]]:
    # Staged mode has an Alice-boundary trajectory artifact plus a finalized
    # proposal artifact. Deduplicate them by proposal/request ID and prefer
    # the record whose Bob receipts have already finalized the reward.
    infos_by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        info = None
        if isinstance(payload, dict):
            info = payload.get("alice_rollout_info") or payload.get("final_info")
        if isinstance(info, dict) and info.get("role") == "alice":
            identity = str(
                info.get("proposal_id") or info.get("request_id") or path.resolve()
            )
            previous = infos_by_id.get(identity)
            if previous is None or (
                bool(previous.get("reward_pending", False))
                and not bool(info.get("reward_pending", False))
            ):
                infos_by_id[identity] = info
    return list(infos_by_id.values())


def _phase_timings(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("phase_timing.rank-*.jsonl")):
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
        except (OSError, json.JSONDecodeError):
            continue
    return rows


def main() -> None:
    args = parse_args()
    infos = _alice_infos(args.alice_rollouts)
    rewards = [float(info.get("reward", 0.0)) for info in infos]
    valid = [bool((info.get("compiled_task") or {}).get("valid")) for info in infos]
    displacements = [
        float((info.get("compiled_task") or {}).get("displacement", 0.0)) for info in infos
    ]
    grasped = [
        bool((info.get("feasibility_metrics") or {}).get("ever_grasped", False)) for info in infos
    ]
    approach = [
        float((info.get("feasibility_metrics") or {}).get("approach_progress", 0.0))
        for info in infos
    ]
    budgets = [
        info["rollout_budget"]
        for info in infos
        if isinstance(info.get("rollout_budget"), dict)
    ]
    generated_tokens = [int(item.get("generated_tokens_total", 0)) for item in budgets]
    timings = _phase_timings(args.alice_rollouts.parent)
    bob_eval = json.loads(args.bob_eval.read_text(encoding="utf-8"))
    result = {
        "schema_version": "openeta.embodied_formal_round.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "round": args.round,
        "alice": {
            "episodes": len(infos),
            "valid_proposals": sum(valid),
            "valid_proposal_rate": sum(valid) / len(valid) if valid else 0.0,
            "ever_grasped_rate": sum(grasped) / len(grasped) if grasped else 0.0,
            "mean_approach_progress": statistics.fmean(approach) if approach else 0.0,
            "mean_displacement": statistics.fmean(displacements) if displacements else 0.0,
            "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
            "budget_observations": len(budgets),
            "mean_generated_tokens": (
                statistics.fmean(generated_tokens) if generated_tokens else None
            ),
            "max_generated_tokens": max(generated_tokens) if generated_tokens else None,
        },
        "bob": {
            "holdout_episodes": int(bob_eval["episodes"]),
            "holdout_success_rate": float(bob_eval["success_rate"]),
            "holdout_mean_reward": float(bob_eval["mean_reward"]),
        },
        "curriculum": {"bob_dataset_mode": args.bob_dataset_mode},
        "timing": {
            "records": len(timings),
            "mean_rollout_seconds": (
                statistics.fmean(
                    float(row.get("rollout_seconds", 0.0)) for row in timings
                )
                if timings else None
            ),
            "max_gpu_peak_bytes": max(
                (
                    int(row.get("gpu_after_rollout_peak_bytes", row.get("gpu_peak_memory_bytes", 0)))
                    for row in timings
                ),
                default=None,
            ),
        },
        "checkpoints": {
            "alice": str(args.alice_adapter.resolve()),
            "bob": str(args.bob_adapter.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
