#!/usr/bin/env python3
"""Create ms-swift JSONL entry rows for embodied Alice or Bob rollouts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from agent.runtime.embodied_selfplay_prompt import ALICE_TASK, BOB_TASK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("alice", "bob"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip this many unique valid Alice tasks before selecting Bob rows.",
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--camera-resolution", type=int, default=128)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the same Qwen thinking chat template as the rollout server.",
    )
    return parser.parse_args()


def _row(role: str, env_request: dict[str, Any], *, row_id: str) -> dict[str, Any]:
    return {
        # The scheduler replaces this bootstrap message after restoring the
        # authoritative snapshot and rendering the first observation.
        "messages": [{"role": "user", "content": "Environment bootstrap pending."}],
        "role": role,
        "row_id": row_id,
        "env_request": env_request,
        "chat_template_kwargs": {
            "enable_thinking": bool(env_request.get("enable_thinking", False)),
        },
    }


def _alice_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    snapshot_dir = (args.snapshot_dir or args.output.parent / "snapshots").resolve()
    env = ManiSkillSimulatorAdapter(
        env_id="PickCube-v1",
        camera_resolution=args.camera_resolution,
        snapshot_dir=snapshot_dir,
    )
    rows: list[dict[str, Any]] = []
    try:
        for offset in range(args.count):
            seed = args.seed_start + offset
            env.reset(seed=seed)
            snapshot = env.capture_snapshot()
            rows.append(_row("alice", {
                "role": "alice",
                "env_id": "PickCube-v1",
                "seed": seed,
                "snapshot": snapshot.to_dict(),
                "instruction": ALICE_TASK,
                "max_steps": args.max_steps,
                "camera_resolution": args.camera_resolution,
                "min_goal_displacement": 0.04,
                "goal_tolerance": 0.025,
                "replay_tolerance": 0.005,
                "enable_thinking": args.enable_thinking,
            }, row_id=f"alice-seed-{seed}"))
    finally:
        env.close()
    return rows


def _candidate_artifacts(source_run: Path) -> list[Path]:
    paths = list(source_run.rglob("alice_trajectory.json"))
    paths.extend(source_run.rglob("*.json"))
    # Preserve deterministic order while removing duplicate trajectory paths.
    return sorted({path.resolve() for path in paths})


def _task_from_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    snapshot = payload.get("initial_snapshot") or payload.get("snapshot")
    compiled = (
        payload.get("compiled")
        or (payload.get("final_info") or {}).get("compiled_task")
        or (payload.get("alice_rollout_info") or {}).get("compiled_task")
    )
    if not isinstance(snapshot, dict) or not isinstance(compiled, dict):
        return None
    if not compiled.get("valid"):
        return None
    predicate = compiled.get("goal_predicate")
    if not isinstance(predicate, dict) or not isinstance(predicate.get("position"), list):
        return None
    return snapshot, compiled


def _bob_rows_from_alice(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.source_run is None:
        raise ValueError("--source-run is required for Bob so every goal comes from Alice")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    unique_index = 0
    for path in _candidate_artifacts(args.source_run):
        parsed = _task_from_artifact(path)
        if parsed is None:
            continue
        snapshot, compiled = parsed
        key = (str(snapshot.get("state_sha256")), str(compiled.get("task_id")))
        if key in seen:
            continue
        seen.add(key)
        if unique_index < args.skip:
            unique_index += 1
            continue
        unique_index += 1
        predicate = dict(compiled["goal_predicate"])
        rows.append(_row("bob", {
            "role": "bob",
            "env_id": str((snapshot.get("metadata") or {}).get("env_id") or "PickCube-v1"),
            "seed": snapshot.get("seed"),
            "snapshot": snapshot,
            "instruction": BOB_TASK,
            "goal_predicate": predicate,
            "max_steps": args.max_steps,
            "camera_resolution": args.camera_resolution,
            "goal_tolerance": float(predicate.get("tolerance", 0.025)),
            "enable_thinking": args.enable_thinking,
        }, row_id=f"bob-{compiled.get('task_id', len(rows))}"))
        if len(rows) >= args.count:
            break
    if not rows:
        raise ValueError(f"no valid Alice tasks found below {args.source_run}")
    return rows


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.skip < 0:
        raise ValueError("--skip must be non-negative")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    rows = _alice_rows(args) if args.role == "alice" else _bob_rows_from_alice(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "role": args.role,
        "rows": len(rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
