"""Dummy simulator adapter used to validate the OpenETA bridge."""

from __future__ import annotations

import hashlib
import json

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState, StepResult
from adapter.sim import SimulatorAdapter
from agent.runtime.embodied_snapshot import SnapshotRef


class DummySimulatorAdapter(SimulatorAdapter):
    """Small deterministic simulator with RGBD and robot-state observations."""

    def __init__(self) -> None:
        self._task = "move the dummy robot"
        self._step_idx = 0
        self._last_action: EnvAction | None = None
        self._snapshots: dict[str, dict] = {}

    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        del seed
        self._task = task or self._task
        self._step_idx = 0
        self._last_action = None
        return self.observe()

    def observe(self) -> EnvObservation:
        rgb_value = 32 + self._step_idx * 32
        rgb = [
            [[rgb_value, 0, 0], [0, rgb_value, 0]],
            [[0, 0, rgb_value], [rgb_value, rgb_value, rgb_value]],
        ]
        depth = [[1.0, 1.1], [1.2, 1.3]]
        camera = CameraFrame(
            frame_id="dummy_front",
            rgb=rgb,
            depth=depth,
            intrinsics={"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5},
            extrinsics={"frame": "world"},
            timestamp_s=float(self._step_idx),
        )
        robot = RobotState(
            joint_positions=[0.1 * self._step_idx, 0.0, 0.0],
            joint_velocities=[0.0, 0.0, 0.0],
            end_effector_pose={"xyz": [0.0, 0.0, 0.5], "quat_xyzw": [0.0, 0.0, 0.0, 1.0]},
            gripper_state={"open": True},
        )
        return EnvObservation(
            task=self._task,
            cameras=[camera],
            robot=robot,
            objects=[{"name": "dummy_cube", "position": [0.2, 0.0, 0.0]}],
            metadata={"step_idx": self._step_idx, "last_action": self._last_action.action_type if self._last_action else None},
        )

    def step(self, action: EnvAction) -> StepResult:
        self._last_action = action
        self._step_idx += 1
        terminated = self._step_idx >= 1
        return StepResult(
            observation=self.observe(),
            reward=1.0 if terminated else 0.0,
            terminated=terminated,
            truncated=False,
            info={"accepted_action_type": action.action_type},
        )

    def capture_snapshot(self) -> SnapshotRef:
        payload = {
            "task": self._task,
            "step_idx": self._step_idx,
            "last_action": self._last_action.to_dict() if self._last_action else None,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        snapshot_id = f"dummy-{digest[:16]}"
        self._snapshots[snapshot_id] = payload
        return SnapshotRef(
            snapshot_id=snapshot_id,
            env_id="openeta/dummy_sim-v0",
            state_uri=f"memory://dummy/{snapshot_id}",
            state_sha256=digest,
            metadata={"backend": "dummy", "format": "json-v1"},
        )

    def restore_snapshot(self, snapshot: SnapshotRef) -> None:
        if snapshot.env_id != "openeta/dummy_sim-v0":
            raise ValueError(f"snapshot belongs to {snapshot.env_id!r}, not dummy simulator")
        payload = self._snapshots.get(snapshot.snapshot_id)
        if payload is None:
            raise FileNotFoundError(f"unknown in-memory snapshot: {snapshot.snapshot_id}")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if digest != snapshot.state_sha256:
            raise ValueError("snapshot hash mismatch")
        self._task = str(payload["task"])
        self._step_idx = int(payload["step_idx"])
        action = payload.get("last_action")
        self._last_action = EnvAction.from_dict(action) if isinstance(action, dict) else None
