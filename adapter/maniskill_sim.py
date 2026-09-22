"""OpenETA simulator adapter for a single ManiSkill Panda task.

The adapter intentionally exposes a small, host-owned action vocabulary for the
first Alice/Bob rollout.  Raw simulator objects remain behind this boundary.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState, StepResult
from adapter.sim import SimulatorAdapter
from agent.runtime.embodied_snapshot import SnapshotRef


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _first(value: Any) -> np.ndarray:
    arr = _array(value)
    return arr[0] if arr.ndim > 1 else arr


def _state_digest(value: Any) -> str:
    """Stable digest for nested ManiSkill tensor state dictionaries."""

    h = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                h.update(str(key).encode("utf-8"))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "detach"):
            arr = item.detach().cpu().contiguous().numpy()
            h.update(str(arr.dtype).encode("ascii"))
            h.update(repr(arr.shape).encode("ascii"))
            h.update(arr.tobytes())
        elif hasattr(item, "tobytes"):
            h.update(item.tobytes())
        else:
            h.update(json.dumps(item, sort_keys=True, default=str).encode("utf-8"))

    visit(value)
    return h.hexdigest()


class ManiSkillSimulatorAdapter(SimulatorAdapter):
    """Single-env ManiSkill adapter using Panda and RGB-D observations."""

    def __init__(
        self,
        env_id: str = "PickCube-v1",
        *,
        control_mode: str = "pd_ee_delta_pose",
        render_mode: str = "rgb_array",
        camera_resolution: int = 128,
        snapshot_dir: str | Path | None = None,
    ) -> None:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401 - registers ManiSkill tasks

        self.env_id = env_id
        self._env = gym.make(
            env_id,
            obs_mode="rgbd",
            control_mode=control_mode,
            render_mode=render_mode,
            num_envs=1,
            sensor_configs={"base_camera": {"width": camera_resolution, "height": camera_resolution}},
        )
        self._last_raw_obs: dict[str, Any] | None = None
        self._last_info: dict[str, Any] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._snapshot_dir = Path(snapshot_dir or "runs/maniskill_snapshots").resolve()
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    @property
    def action_dim(self) -> int:
        return int(self._env.action_space.shape[-1])

    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        del task
        raw, info = self._env.reset(seed=seed)
        self._last_raw_obs = raw
        self._last_info = info
        return self._observation(raw, info)

    def observe(self) -> EnvObservation:
        if self._last_raw_obs is None:
            raise RuntimeError("reset() must be called before observe()")
        return self._observation(self._last_raw_obs, self._last_info)

    def step(self, action: EnvAction) -> StepResult:
        raw_action = self._encode_action(action)
        raw_obs, reward, terminated, truncated, info = self._env.step(raw_action[None, :])
        self._last_raw_obs = raw_obs
        self._last_info = info
        return StepResult(
            observation=self._observation(raw_obs, info),
            reward=float(_first(reward).reshape(-1)[0]),
            terminated=bool(_first(terminated).reshape(-1)[0]),
            truncated=bool(_first(truncated).reshape(-1)[0]),
            info=self._plain_info(info),
        )

    def capture_snapshot(self) -> SnapshotRef:
        state = copy.deepcopy(self._env.unwrapped.get_state_dict())
        elapsed_steps = copy.deepcopy(getattr(self._env.unwrapped, "_elapsed_steps", None))
        payload = {"state": state, "elapsed_steps": elapsed_steps}
        digest = _state_digest(payload)
        snapshot_id = f"maniskill-{digest[:16]}"
        self._snapshots[snapshot_id] = payload
        state_path = self._snapshot_dir / f"{snapshot_id}.pt"
        import torch

        torch.save(payload, state_path)
        return SnapshotRef(
            snapshot_id=snapshot_id,
            env_id=f"openeta/maniskill_{self.env_id}-v0",
            state_uri=str(state_path),
            state_sha256=digest,
            metadata={"backend": "maniskill", "env_id": self.env_id, "format": "state_dict-v1"},
        )

    def restore_snapshot(self, snapshot: SnapshotRef) -> None:
        payload = self._snapshots.get(snapshot.snapshot_id)
        if payload is None:
            state_path = Path(snapshot.state_uri)
            if not state_path.is_file():
                raise FileNotFoundError(f"snapshot state not found: {state_path}")
            import torch

            payload = torch.load(state_path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict) or "state" not in payload:
                raise ValueError("invalid ManiSkill snapshot payload")
            self._snapshots[snapshot.snapshot_id] = payload
        if payload is None:
            raise FileNotFoundError(f"unknown in-memory snapshot: {snapshot.snapshot_id}")
        digest = _state_digest(payload)
        if digest != snapshot.state_sha256:
            raise ValueError("snapshot hash mismatch")
        self._env.unwrapped.set_state_dict(copy.deepcopy(payload["state"]))
        if payload["elapsed_steps"] is not None:
            self._env.unwrapped._elapsed_steps = copy.deepcopy(payload["elapsed_steps"])
        self._last_raw_obs = self._env.unwrapped.get_obs()
        self._last_info = {}

    def close(self) -> None:
        self._env.close()

    def _encode_action(self, action: EnvAction) -> np.ndarray:
        code = str(action.code or action.action_type or "DONE").upper()
        result = np.zeros(self.action_dim, dtype=np.float32)
        # ManiSkill pd_ee_delta_pose: xyz delta, rotation delta, gripper.
        if code in {"MOVE", "MOVE_Z_POS"}:
            result[2] = 0.02
            result[-1] = -1.0
        elif code == "MOVE_Z_NEG":
            result[2] = -0.02
            result[-1] = -1.0
        elif code == "MOVE_X_POS":
            result[0] = 0.02
            result[-1] = -1.0
        elif code == "MOVE_X_NEG":
            result[0] = -0.02
            result[-1] = -1.0
        elif code == "MOVE_Y_POS":
            result[1] = 0.02
            result[-1] = -1.0
        elif code == "MOVE_Y_NEG":
            result[1] = -0.02
            result[-1] = -1.0
        elif code == "GRASP":
            result[-1] = 1.0
        elif code == "RELEASE":
            result[-1] = -1.0
        elif code == "DONE":
            result[-1] = -1.0
        else:
            raise ValueError(f"unsupported embodied action: {code}")
        return result

    def _observation(self, raw: dict[str, Any], info: dict[str, Any]) -> EnvObservation:
        sensor = raw["sensor_data"]["base_camera"]
        rgb = _first(sensor["rgb"]).astype(np.uint8)
        depth = _first(sensor["depth"]).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        agent = raw.get("agent", {})
        extra = raw.get("extra", {})
        qpos = _first(agent.get("qpos", []))
        qvel = _first(agent.get("qvel", []))
        tcp = _first(extra.get("tcp_pose", []))
        goal = _first(extra.get("goal_pos", []))
        return EnvObservation(
            task=f"ManiSkill task: {self.env_id}",
            cameras=[CameraFrame(
                frame_id="base_camera",
                rgb=rgb.tolist(),
                depth=depth.tolist(),
                role="scene_primary",
            )],
            robot=RobotState(
                joint_positions=qpos.astype(float).tolist(),
                joint_velocities=qvel.astype(float).tolist(),
                end_effector_pose={"xyz": tcp[:3].astype(float).tolist()} if tcp.size >= 3 else {},
                gripper_state={"open": bool(qpos[-1] < 0.02) if qpos.size else True},
            ),
            objects=[
                {"name": "cube", "position": goal.astype(float).tolist(), "role": "goal"}
            ] if goal.size >= 3 else [],
            metadata={"env_id": self.env_id, "success": self._plain_info(info).get("success")},
        )

    @staticmethod
    def _plain_info(info: Any) -> dict[str, Any]:
        if not isinstance(info, dict):
            return {}
        result: dict[str, Any] = {}
        for key, value in info.items():
            try:
                arr = _array(value)
                result[key] = arr.tolist() if arr.ndim else arr.item()
            except Exception:
                result[key] = str(value)
        return result
