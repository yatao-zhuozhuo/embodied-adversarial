#!/usr/bin/env python3
"""Create a procedural Bob curriculum when no historical Alice run exists.

The generated goals are explicit cold-start seeds, not claimed model rollouts.
Each task contains a real, replayable ManiSkill snapshot and a nearby tabletop
cube target that passes the same compiler used for Alice proposals.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from agent.runtime.embodied_task_compiler import compile_cube_position_task


OFFSETS = (
    (0.08, 0.00),
    (-0.08, 0.00),
    (0.00, 0.08),
    (0.00, -0.08),
    (0.07, 0.05),
    (-0.07, 0.05),
    (0.07, -0.05),
    (-0.07, -0.05),
)


def _target(initial: list[float], offset: tuple[float, float]) -> list[float]:
    x = max(-0.30, min(0.30, float(initial[0]) + offset[0]))
    y = max(-0.30, min(0.30, float(initial[1]) + offset[1]))
    return [x, y, float(initial[2])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=29000)
    parser.add_argument("--camera-resolution", type=int, default=64)
    args = parser.parse_args()
    if args.count < 8:
        raise ValueError("cold bootstrap requires at least eight tasks")

    args.output.mkdir(parents=True, exist_ok=True)
    snapshots = args.output / "snapshots"
    existing = sorted(args.output.glob("cold_task_*.json"))
    if len(existing) >= args.count:
        print(json.dumps({"status": "already_ready", "tasks": len(existing)}))
        return

    env = ManiSkillSimulatorAdapter(
        env_id="PickCube-v1",
        camera_resolution=args.camera_resolution,
        snapshot_dir=snapshots,
    )
    written: list[str] = []
    try:
        for index in range(args.count):
            seed = args.seed_start + index
            env.reset(seed=seed)
            snapshot = env.capture_snapshot()
            initial_state = env.task_state()
            initial_cube = list(initial_state["cube_position"])
            target = _target(initial_cube, OFFSETS[index % len(OFFSETS)])
            compiled = compile_cube_position_task(
                snapshot=snapshot,
                initial_state=initial_state,
                final_state={"cube_position": target},
            )
            if not compiled.valid:
                raise RuntimeError(f"invalid procedural task for seed {seed}: {compiled.reason}")
            payload = {
                "schema_version": "openeta.embodied_cold_bootstrap.v1",
                "source": "procedural_cold_start",
                "initial_snapshot": snapshot.to_dict(),
                "initial_state": initial_state,
                "procedural_target_state": {"cube_position": target},
                "compiled": asdict(compiled),
            }
            path = args.output / f"cold_task_{index:04d}.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written.append(str(path.resolve()))
    finally:
        env.close()

    (args.output / "cold_start_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "openeta.embodied_cold_bootstrap.v1",
                "source": "procedural_cold_start",
                "tasks": written,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "created", "tasks": len(written)}))


if __name__ == "__main__":
    main()
