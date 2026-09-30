#!/usr/bin/env python3
"""Probe the spawn-isolated ManiSkill adapter from a real script entry."""
from __future__ import annotations

import argparse
import json

from adapter.maniskill_process import IsolatedManiSkillSimulatorAdapter
from adapter.protocol import EnvAction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-gpu", default="0")
    parser.add_argument("--render-backend", default="sapien_cuda:0")
    parser.add_argument("--seed", type=int, default=39000)
    parser.add_argument("--camera-resolution", type=int, default=64)
    args = parser.parse_args()

    env = IsolatedManiSkillSimulatorAdapter(
        render_backend=args.render_backend,
        physical_render_gpu=args.render_gpu,
        camera_resolution=args.camera_resolution,
        snapshot_dir="/tmp/openeta-isolated-probe-snapshots",
    )
    try:
        observation = env.reset(seed=args.seed)
        snapshot = env.capture_snapshot()
        before = env.task_state()
        result = env.step(EnvAction(action_type="DONE", code="DONE"))
        env.restore_snapshot(snapshot)
        after = env.task_state()
        print(json.dumps({
            "worker_pid": env.worker_pid,
            "render_device": env.render_device,
            "rgb_height": len(observation.cameras[0].rgb),
            "rgb_width": len(observation.cameras[0].rgb[0]),
            "step_terminated": result.terminated,
            "snapshot_sha256": snapshot.state_sha256,
            "restore_equal": before["cube_position"] == after["cube_position"],
        }, sort_keys=True))
    finally:
        env.close()


if __name__ == "__main__":
    main()
