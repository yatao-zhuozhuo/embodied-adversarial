#!/usr/bin/env python3
"""Generate deterministic all-train ManiSkill SFT manifests for both teachers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT_ROOT = REPO_ROOT / "SFT_data"
DEFAULT_CATALOG = DEFAULT_SFT_ROOT / "configs" / "task_catalog.v1.json"
TEACHERS = {
    "glm": "GLM-5.3-w8a8c8",
    "qwen": "Qwen3.8-27B",
}
TARGET_EPISODES_PER_TEACHER = 500
TIER_LIMITS = {
    "A": {"max_turns": 100, "max_tool_calls": 200, "timeout_s": 3600},
    "B": {"max_turns": 140, "max_tool_calls": 280, "timeout_s": 5400},
    "C": {"max_turns": 180, "max_tool_calls": 360, "timeout_s": 7200},
}


def _teacher_instruction(task: dict[str, Any], teacher_id: str) -> str:
    notes = [
        "Deployment note: ManiSkill native IK is available through "
        "ik_preview_check, and every move_to must use the returned ik_receipt_id; "
        "never pass target_pose directly to move_to. cuRobo endpoint collision "
        "checking is unavailable in this run, so use short observable waypoints, "
        "inspect every motion receipt, and change the waypoint after any "
        "non-convergence. For the Panda's existing top-down tool pose, preserve "
        "the current orientation: omit roll/pitch/yaw in both ik_preview_check "
        "and move_to unless the task truly requires a wrist reorientation.",
    ]
    if teacher_id == "glm":
        notes.append(
            "This endpoint is text-only. The observation objects list is the "
            "authoritative perception channel for object and goal poses; use it "
            "directly and do not call retrieve_asset_reference for visual confirmation."
        )
    if str(task.get("task_slug") or "") == "pick_cube":
        notes.append(
            "PickCube calibrated policy: call gripper_open first; approach the live "
            "cube center plus 0.10 m in z; then align the TCP to the live cube center "
            "plus exactly 0.025 m in z using position tolerance 0.003 m; close the "
            "gripper immediately; then make exactly one decisive lift, using the "
            "cube center observed immediately after closing as the fixed reference "
            "and targeting that position plus 0.18 m in z while holding it closed "
            "(a small 0.10 m lift can let the cube slip). Do not add another 0.18 m "
            "to the already-lifted live pose. After grasping, call python_exec to "
            "compute the observed TCP-minus-cube offset and all three components "
            "of final_target = goal_site + offset. Pass final_target, never the "
            "offset vector itself, to ik_preview_check. Before previewing, verify "
            "final_target_z = goal_site_z + offset_z (normally about 0.315 m for "
            "this task), then move the TCP to that final target. Do not explore "
            "lower grasp points or lateral offsets."
        )
    return f"{task['instruction']} {' '.join(notes)}"


def _load_catalog(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("task catalog requires a non-empty tasks list")
    required = {"task_slug", "env_id", "instruction", "tier", "quota"}
    seen: set[str] = set()
    total = 0
    normalized: list[dict[str, Any]] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"tasks[{index}] must be an object")
        missing = sorted(required - set(task))
        if missing:
            raise ValueError(f"tasks[{index}] is missing: {', '.join(missing)}")
        slug = str(task["task_slug"])
        if slug in seen:
            raise ValueError(f"duplicate task_slug: {slug}")
        seen.add(slug)
        tier = str(task["tier"])
        if tier not in TIER_LIMITS:
            raise ValueError(f"unsupported tier for {slug}: {tier}")
        quota = int(task["quota"])
        if quota < 1:
            raise ValueError(f"quota must be positive for {slug}")
        total += quota
        normalized.append(dict(task))
    if len(normalized) != 20:
        raise ValueError(f"expected 20 tasks, got {len(normalized)}")
    if total != TARGET_EPISODES_PER_TEACHER:
        raise ValueError(
            f"expected per-teacher quota {TARGET_EPISODES_PER_TEACHER}, got {total}"
        )
    return normalized


def build_manifest(
    tasks: list[dict[str, Any]],
    *,
    teacher_id: str,
    output_root: Path,
    seed_offset: int,
) -> dict[str, Any]:
    model = TEACHERS[teacher_id]
    workspace_parent = (output_root / "raw" / teacher_id).resolve()
    episodes: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        limits = TIER_LIMITS[str(task["tier"])]
        task_seed_base = seed_offset + task_index * 100_000
        for offset in range(int(task["quota"])):
            seed = task_seed_base + offset
            episodes.append(
                {
                    "episode_id": f"{teacher_id}-{task['task_slug']}-seed-{seed:07d}",
                    "env_id": task["env_id"],
                    "task": _teacher_instruction(task, teacher_id),
                    "seed": seed,
                    **limits,
                    "max_total_tokens": 1_000_000,
                    "metadata": {
                        "dataset_schema": "openeta.maniskill_sft_collection.v1",
                        "dataset_split": "train",
                        "teacher_id": teacher_id,
                        "teacher_model": model,
                        # Both teachers remain independent, but both receive
                        # simulator-grounded object state.  Qwen additionally
                        # receives RGB; no teacher consumes the other one's
                        # output or scores its trajectories.
                        "include_objects": True,
                        "perception_mode": (
                            "structured_state"
                            if teacher_id == "glm"
                            else "rgb_plus_structured_state"
                        ),
                        "task_slug": task["task_slug"],
                        "task_tier": task["tier"],
                        "require_official_reward": True,
                        "on_need_human": "fail",
                        "workspace_parent": str(workspace_parent),
                    },
                }
            )
    return {
        "schema_version": "openeta.maniskill_sft_manifest.v1",
        "teacher_id": teacher_id,
        "teacher_model": model,
        "dataset_split": "train",
        "target_success_count": TARGET_EPISODES_PER_TEACHER,
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    tasks = _load_catalog(args.catalog.resolve())
    manifest_dir = args.output_root.resolve() / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for teacher_id in TEACHERS:
        payload = build_manifest(
            tasks,
            teacher_id=teacher_id,
            output_root=args.output_root.resolve(),
            seed_offset=args.seed_offset,
        )
        path = manifest_dir / (
            f"{teacher_id}.train.{TARGET_EPISODES_PER_TEACHER}.v1.json"
        )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        outputs[teacher_id] = str(path)
    print(
        json.dumps(
            {
                "episodes_per_teacher": TARGET_EPISODES_PER_TEACHER,
                "outputs": outputs,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
