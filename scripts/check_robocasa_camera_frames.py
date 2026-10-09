"""Save real MCP camera frames for RoboCasa rendering regression checks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys
from uuid import uuid4

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tools.sim_mcp import SseSimulatorMcpTransport


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-limit", type=int, default=5)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = [row for row in manifest["tasks"] if row["bench"] == "robocasa"]
    rows = rows[args.task_offset:args.task_offset + args.task_limit]
    transport = SseSimulatorMcpTransport(args.sim_url)
    report = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for task_index, row in enumerate(rows, start=args.task_offset):
        seed = 20310000 + task_index * 10000
        session = "camera-check-" + uuid4().hex
        created = transport.call_tool("create_env", {
            "env_id": row["env_id"], "seed": seed, "session_id": session,
            "image_width": 256, "image_height": 256, "include_objects": False,
        }, timeout_s=180)
        if created.get("error") or not created.get("handle"):
            raise RuntimeError(str(created))
        base = {"handle": created["handle"], "session_id": created.get("session_id", session)}
        try:
            for stage in range(args.steps + 2):
                if stage == 0:
                    tool = "reset_env"
                    arguments = {**base, "seed": seed}
                elif stage == 1:
                    tool = "observe_env"
                    arguments = base
                else:
                    tool = "step_env"
                    action = [0.0] * 12
                    action[6] = -1.0
                    action[11] = -1.0
                    arguments = {**base, "action": action, "num_steps": 1, "strict": True}
                result = transport.call_tool(tool, arguments, timeout_s=180)
                if result.get("error"):
                    raise RuntimeError(str(result["error"]))
                observation = result.get("observation", result)
                cameras = observation.get("cameras", [])
                if {camera["frame_id"] for camera in cameras} != {"agentview_left", "agentview_right", "wrist"}:
                    raise RuntimeError("Missing or unexpected RoboCasa cameras")
                for camera in cameras:
                    encoded = base64.b64decode(camera["rgb_base64"], validate=True)
                    pixels = np.asarray(Image.open(io.BytesIO(encoded)).convert("RGB"))
                    if pixels.shape != (256, 256, 3):
                        raise RuntimeError(f"Unexpected frame shape: {pixels.shape}")
                    directory = args.output_root / row["task_id"]
                    directory.mkdir(exist_ok=True)
                    path = directory / f"{stage:04d}-{camera['frame_id']}.png"
                    path.write_bytes(encoded)
                    report.append({
                        "task": row["task_id"], "seed": seed, "stage": stage, "tool": tool,
                        "camera": camera["frame_id"], "path": str(path),
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                        "horizontal_difference": float(np.abs(np.diff(pixels.astype(float), axis=1)).mean()),
                    })
                (args.output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps({"task": row["task_id"], "stage": stage, "cameras": report[-3:]}), flush=True)
                if result.get("terminated") or result.get("truncated"):
                    break
        finally:
            transport.call_tool("close_env", base, timeout_s=180)


if __name__ == "__main__":
    main()
