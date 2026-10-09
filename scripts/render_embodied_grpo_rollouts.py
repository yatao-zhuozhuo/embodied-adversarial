#!/usr/bin/env python3
"""Replay saved embodied GRPO actions and render annotated GIFs/contact sheets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from adapter.protocol import EnvAction
from agent.runtime.embodied_snapshot import SnapshotRef


def _frame(rgb: Any, title: str, *, size: int = 512) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize(
        (size, size), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (size, size + 54), "white")
    canvas.paste(image, (0, 54))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title[:100], fill="black")
    return canvas


def _load_record(path: Path, iteration: int) -> dict[str, Any]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not records:
        raise ValueError(f"no rollout records in {path}")
    if iteration < 0:
        return records[iteration]
    for record in records:
        if int(record.get("iteration", -1)) == iteration:
            return record
    raise ValueError(f"iteration {iteration} not found in {path}")


def render(
    path: Path,
    output_dir: Path,
    iteration: int,
    fps: float,
    *,
    view: str,
) -> list[Path]:
    record = _load_record(path, iteration)
    snapshot_payload = record["snapshot"]
    snapshot = SnapshotRef(
        snapshot_id=snapshot_payload["snapshot_id"],
        env_id=snapshot_payload["env_id"],
        state_uri=snapshot_payload["state_uri"],
        state_sha256=snapshot_payload["state_sha256"],
        seed=snapshot_payload.get("seed"),
        metadata=dict(snapshot_payload.get("metadata") or {}),
    )
    env_id = str(snapshot.metadata.get("env_id") or "PickCube-v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    env = ManiSkillSimulatorAdapter(env_id=env_id, camera_resolution=128)
    rendered: list[Path] = []
    sheet_rows: list[list[Image.Image]] = []
    try:
        # Satisfy Gymnasium's OrderEnforcing wrapper before restoring the
        # authoritative physics snapshot into the newly constructed env.
        env.reset(seed=snapshot.seed or 0)
        for episode_index, episode in enumerate(record["episodes"]):
            env.restore_snapshot(snapshot)
            observation = env.observe()
            initial_rgb = (
                env.render_rgb() if view == "render" else observation.cameras[0].rgb
            )
            frames = [_frame(initial_rgb, f"episode {episode_index} | initial state")]
            for step_index, step in enumerate(episode["steps"]):
                action = str(step["action"])
                result = env.step(EnvAction(action_type=action, code=action))
                title = (
                    f"step {step_index}: {action} | reward={float(step['environment_reward']):.4f} "
                    f"| success={bool(step['success'] or step['is_obj_placed'])}"
                )
                rgb = (
                    env.render_rgb()
                    if view == "render"
                    else result.observation.cameras[0].rgb
                )
                frames.append(_frame(rgb, title))
            gif_path = output_dir / f"episode-{episode_index:02d}.gif"
            frames[0].save(
                gif_path,
                save_all=True,
                append_images=frames[1:],
                duration=max(100, round(1000.0 / fps)),
                loop=0,
            )
            rendered.append(gif_path)
            mp4_path = output_dir / f"episode-{episode_index:02d}.mp4"
            imageio.mimsave(
                mp4_path,
                [np.asarray(frame) for frame in frames],
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=2,
            )
            rendered.append(mp4_path)
            sheet_rows.append(frames)

        thumb_width, thumb_height = 256, 283
        columns = max(len(row) for row in sheet_rows)
        sheet = Image.new("RGB", (columns * thumb_width, len(sheet_rows) * thumb_height), "#dddddd")
        for row_index, row in enumerate(sheet_rows):
            for column_index, frame in enumerate(row):
                sheet.paste(
                    frame.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS),
                    (column_index * thumb_width, row_index * thumb_height),
                )
        sheet_path = output_dir / "contact-sheet.png"
        sheet.save(sheet_path)
        rendered.append(sheet_path)
    finally:
        env.close()
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration number; -1 selects latest")
    parser.add_argument("--fps", type=float, default=1.5)
    parser.add_argument(
        "--view",
        choices=("render", "agent"),
        default="render",
        help="Third-person simulator render or the RGB observation seen by the agent",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.rollouts.parent / "visualizations"
    paths = render(
        args.rollouts.resolve(),
        output_dir.resolve(),
        args.iteration,
        args.fps,
        view=args.view,
    )
    print(json.dumps({"rendered": [str(path) for path in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
