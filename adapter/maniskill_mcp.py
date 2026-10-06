"""Host-owned PickCube adapter whose model actions execute through simulator MCP."""

from __future__ import annotations

import json
import os
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from urllib import error, request
from uuid import uuid4
import numpy as np
from PIL import Image, ImageDraw

from adapter.protocol import EnvObservation, StepResult
from agent.runtime.embodied_snapshot import SnapshotRef
from agent.tools.sim_mcp import SseSimulatorMcpTransport


class McpPickCubeAdapter:
    def _observation(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("MCP observation must be an object")
        for camera in payload.get("cameras") or []:
            if isinstance(camera, dict) and camera.get("rgb_base64"):
                camera["rgb"] = np.asarray(Image.open(io.BytesIO(
                    base64.b64decode(camera["rgb_base64"])
                )).convert("RGB"), dtype=np.uint8)
        observation = EnvObservation.from_dict(payload)
        if not observation.cameras or not observation.cameras[0].rgb:
            raise ValueError("MCP returned no model-visible RGB frame")
        if self._goal_position is not None:
            self._draw_goal_marker(observation)
        return observation

    def _draw_goal_marker(self, observation):
        """Render Bob's visible target on the MCP RGB using its calibration."""
        camera = observation.cameras[0]
        intr = camera.intrinsics
        extr = camera.extrinsics
        if extr.get("camera_frame") != "ros":
            raise ValueError("PickCube camera has no ROS calibration")
        q = np.asarray(extr.get("quat_xyzw"), dtype=float)
        origin = np.asarray(extr.get("pos"), dtype=float)
        if q.shape != (4,) or origin.shape != (3,):
            raise ValueError("invalid PickCube camera calibration")
        q = q / np.linalg.norm(q)
        x, y, z, w = q
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        local = rotation.T @ (np.asarray(self._goal_position, dtype=float) - origin)
        if local[0] <= 0:
            raise ValueError("Bob goal marker is behind the camera")
        u = float(intr["fx"]) * (-local[1] / local[0]) + float(intr["cx"])
        v = float(intr["fy"]) * (-local[2] / local[0]) + float(intr["cy"])
        frame = Image.fromarray(np.asarray(camera.rgb, dtype=np.uint8))
        if not (4 <= u < frame.width - 4 and 4 <= v < frame.height - 4):
            raise ValueError("Bob goal marker is outside the primary RGB frame")
        draw = ImageDraw.Draw(frame)
        cx, cy = round(u), round(v)
        draw.ellipse((cx-5, cy-5, cx+5, cy+5), outline=(0, 0, 0), width=3)
        draw.ellipse((cx-4, cy-4, cx+4, cy+4), outline=(0, 255, 0), width=2)
        draw.line((cx-2, cy, cx+2, cy), fill=(0, 255, 0), width=1)
        draw.line((cx, cy-2, cx, cy+2), fill=(0, 255, 0), width=1)
        camera.rgb = np.asarray(frame).tolist()

    def __init__(self, *, camera_resolution: int = 128, **_ignored):
        base = os.environ.get("OPENETA_EMBODIED_MCP_URL", "").rstrip("/")
        token = os.environ.get("OPENETA_HOST_STATE_TOKEN", "")
        if not base.startswith("http://127.0.0.1:") or not token:
            raise ValueError("two-tool protocol requires local MCP URL and host state token")
        self.base = base.removesuffix("/sse")
        self.transport = SseSimulatorMcpTransport(self.base + "/sse")
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embodied-mcp")
        self.token = token
        self._goal_position = None
        self.restore_receipts = []
        self.session_id = "embodied-" + uuid4().hex
        created = self._pool.submit(self.transport.call_tool, "create_env", {
            "env_id": "openeta/maniskill_PickCube-v1-v0", "session_id": self.session_id,
            "image_width": camera_resolution, "image_height": camera_resolution,
            "include_objects": True,
        }, timeout_s=180).result()
        if created.get("error"):
            raise RuntimeError(created["error"])
        self.created = created
        self.handle = str(created["handle"])
        self.action_low = created.get("action_low")
        self.action_high = created.get("action_high")
        if (created.get("backend") != "maniskill" or created.get("action_dim") != 7
                or created.get("control_mode") != "pd_ee_delta_pose"):
            self.close()
            raise ValueError("expected seven-dimensional ManiSkill pd_ee_delta_pose control")
        if not isinstance(self.action_low, list) or not isinstance(self.action_high, list) or len(self.action_low) != 7 or len(self.action_high) != 7:
            self.close()
            raise ValueError("MCP did not report full action bounds")
        if self.action_low != [-1.0] * 7 or self.action_high != [1.0] * 7:
            self.close()
            raise ValueError("unexpected normalized PickCube action bounds")

    def _tool(self, name: str, **arguments):
        result = self._pool.submit(self.transport.call_tool, name, {
            "session_id": self.session_id, "handle": self.handle, **arguments
        }, timeout_s=180).result()
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def _host(self, operation: str, **arguments):
        url = f"{self.base}/host-state/{self.session_id}/{self.handle}"
        payload = json.dumps({"operation": operation, **arguments}).encode()
        req = request.Request(url, payload, {
            "Content-Type": "application/json", "X-OpenETA-Host-Token": self.token
        }, method="POST")
        try:
            with request.urlopen(req, timeout=180) as response:
                result = json.load(response)
        except error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode())
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = {}
            raise RuntimeError(f"MCP host {operation} failed: {payload.get('error', exc)}") from exc
        if result.get("error"):
            raise RuntimeError(result["error"])
        return result

    def reset(self, *, seed=None):
        return self._observation(self._tool("reset_env", seed=seed))

    def observe(self):
        return self._observation(self._tool("observe_env"))

    def restore_snapshot(self, snapshot: SnapshotRef):
        result = self._host("restore", snapshot=snapshot.to_dict())
        if result.get("state_sha256") != snapshot.state_sha256:
            raise RuntimeError("MCP restore returned a different state hash")
        actual = result.get("restored_state_sha256")
        if not isinstance(actual, str) or len(actual) != 64:
            raise RuntimeError("MCP restore omitted the measured state hash")
        for previous in self.restore_receipts:
            if previous["snapshot_sha256"] == snapshot.state_sha256 and previous["restored_state_sha256"] != actual:
                raise RuntimeError("repeated MCP restore produced a different state")
        self.restore_receipts.append({"snapshot_sha256": snapshot.state_sha256,
                                      "restored_state_sha256": actual})

    def capture_snapshot(self, *, snapshot_dir: str, seed=None):
        return SnapshotRef(**self._host("capture", snapshot_dir=snapshot_dir, seed=seed))

    def set_goal_position(self, position):
        self._host("set_goal", position=list(position))
        self._goal_position = list(position) if list(position) != [0.0, 0.0, -1.0] else None

    def task_state(self):
        return self._host("task_state")

    def step_vector(self, action: list[float], num_steps: int) -> tuple[StepResult, dict]:
        receipt = self._tool("step_env", action=action, num_steps=num_steps, strict=True)
        if receipt.get("reward_scope") != "last_control_step":
            raise RuntimeError("MCP step did not return reward scope")
        executed = receipt.get("executed_control_steps")
        if isinstance(executed, bool) or not isinstance(executed, int) or not 0 <= executed <= num_steps:
            raise RuntimeError("MCP step returned invalid control-step count")
        result = StepResult.from_dict({**receipt, "observation": {**receipt["observation"]}})
        result.observation = self._observation(receipt["observation"])
        return result, receipt

    def close(self):
        if getattr(self, "handle", None):
            try:
                self._tool("close_env")
            finally:
                self.handle = ""
                self._pool.shutdown(wait=False, cancel_futures=True)
