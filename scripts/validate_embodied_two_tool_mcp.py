#!/usr/bin/env python3
"""Fail-closed MCP PickCube preflight before any two-tool GRPO training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.maniskill_mcp import McpPickCubeAdapter
from agent.runtime.embodied_two_tool import project_observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = McpPickCubeAdapter(camera_resolution=128)
    try:
        env.reset(seed=31597)
        snapshot = env.capture_snapshot(snapshot_dir=str(output.parent / "snapshots"), seed=31597)
        env.restore_snapshot(snapshot)
        initial = env.task_state()
        first = env.observe()
        visible = project_observation(first.to_dict())
        if "objects" in visible or "metadata" in visible:
            raise RuntimeError("privileged fields reached the model projection")
        invalid = env.transport.call_tool("step_env", {
            "session_id": env.session_id, "handle": env.handle,
            "action": [2.0] * 7, "num_steps": 1, "strict": True,
        }, timeout_s=30)
        if not invalid.get("error") or env.task_state() != initial:
            raise RuntimeError("invalid action changed the environment")
        result, receipt = env.step_vector([0.0] * 6 + [1.0], 2)
        if receipt["executed_control_steps"] != 2 or result.terminated or result.truncated:
            raise RuntimeError("MCP did not execute exactly two control steps")
        stepped = env.task_state()
        env.restore_snapshot(snapshot)
        if env.task_state() != initial:
            raise RuntimeError("snapshot replay changed the trusted task state")
        replay_initial = env.observe()
        if not np.array_equal(np.asarray(first.cameras[0].rgb),
                              np.asarray(replay_initial.cameras[0].rgb)):
            raise RuntimeError("repeated restore changed the initial RGB frame")
        replay_result, replay_receipt = env.step_vector([0.0] * 6 + [1.0], 2)
        replayed = env.task_state()
        if (replay_receipt["executed_control_steps"] != 2 or
            np.linalg.norm(np.asarray(stepped["cube_position"]) -
                           np.asarray(replayed["cube_position"])) > 0.005):
            raise RuntimeError("vector replay diverged from the first terminal state")
        env.restore_snapshot(snapshot)
        if len({item["restored_state_sha256"] for item in env.restore_receipts}) != 1:
            raise RuntimeError("repeat restores produced different physics hashes")
        cube = initial["cube_position"]
        goal = [float(cube[0]) - 0.04, float(cube[1]), max(float(cube[2]), 0.05)]
        env.set_goal_position(goal)
        bob = env.observe()
        rgb = np.asarray(bob.cameras[0].rgb, dtype=np.uint8)
        green = int(((rgb[:, :, 1] > rgb[:, :, 0] * 1.3) &
                     (rgb[:, :, 1] > rgb[:, :, 2] * 1.3) &
                     (rgb[:, :, 1] > 55)).sum())
        if green < 10:
            raise RuntimeError("Bob target marker is not visible")
        report = {
            "protocol_version": "openeta.two_tool_xml.v1",
            "snapshot_sha256": snapshot.state_sha256,
            "restored_state_sha256": env.restore_receipts[0]["restored_state_sha256"],
            "executed_control_steps": receipt["executed_control_steps"],
            "reward_scope": receipt["reward_scope"],
            "invalid_action_rejected": True,
            "bob_marker_green_pixels": green,
            "model_tools": ["observe_env", "step_env"],
        }
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
