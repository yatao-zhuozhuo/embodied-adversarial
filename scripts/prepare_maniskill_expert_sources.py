#!/usr/bin/env python3
"""Index official ManiSkill successes for verified OpenETA replay.

These files are only candidates. A source success never becomes an accepted
SFT episode until replay_expert_two_tool_sft.py sees OpenETA official success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ("PullCube-v1", "PokeCube-v1", "LiftPegUpright-v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--source-root", type=Path,
                        default=ROOT / "SFT_data/expert_replay_v1/sources_mani_v2")
    args = parser.parse_args()
    if args.per_task < 1 or args.max_steps < 1:
        parser.error("per-task and max-steps must be positive")
    demo_root = ROOT / "SFT_data/expert_replay_v1/official_maniskill_demos"
    counts = {}
    for task_number, task_id in enumerate(TASKS):
        dataset = demo_root / task_id / "rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5"
        metadata = json.loads(dataset.with_suffix(".json").read_text(encoding="utf-8"))
        eligible = [item for item in metadata["episodes"]
                    if item.get("success") is True
                    and item.get("control_mode") == "pd_ee_delta_pose"
                    and 0 < item.get("elapsed_steps", 0) <= args.max_steps]
        eligible.sort(key=lambda item: item["episode_id"])
        selected = eligible[:args.per_task]
        for item in selected:
            episode_id = int(item["episode_id"])
            seed = 34100000 + task_number * 10000 + episode_id
            path = args.source_root / "maniskill" / task_id / f"seed-{seed}.json"
            payload = {
                "schema_version": "openeta.expert_action_source.v1",
                "bench": "maniskill", "task_id": task_id,
                "env_id": f"openeta/maniskill_{task_id}-v0", "seed": seed,
                "source_kind": "official_maniskill_rl_demonstration",
                "source_dataset": str(dataset.relative_to(ROOT)),
                "reference_demo_index": episode_id,
                "source_steps": int(item["elapsed_steps"]),
                "action_conversion": "maniskill_pd_ee_delta_pose_controller",
                "source_sim_backend": "physx_cuda",
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if json.loads(path.read_text(encoding="utf-8")) != payload:
                    raise ValueError(f"Existing source differs: {path}")
            else:
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
        counts[task_id] = {"eligible": len(eligible), "prepared": len(selected)}
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
