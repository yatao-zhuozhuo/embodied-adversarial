#!/usr/bin/env python3
"""Replay one self-play proposal and export Alice/Bob simulator videos."""

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


def _snapshot(payload: dict[str, Any]) -> SnapshotRef:
    return SnapshotRef(
        snapshot_id=payload["snapshot_id"],
        env_id=payload["env_id"],
        state_uri=payload["state_uri"],
        state_sha256=payload["state_sha256"],
        seed=payload.get("seed"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _annotate(rgb: Any, title: str, *, size: int = 512) -> Image.Image:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).resize(
        (size, size), Image.Resampling.LANCZOS
    )
    canvas = Image.new("RGB", (size, size + 54), "white")
    canvas.paste(image, (0, 54))
    ImageDraw.Draw(canvas).text((10, 8), title[:105], fill="black")
    return canvas


def _save_video(frames: list[Image.Image], stem: Path, fps: float) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    gif_path = stem.with_suffix(".gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(100, round(1000 / fps)),
        loop=0,
    )
    mp4_path = stem.with_suffix(".mp4")
    imageio.mimsave(
        mp4_path,
        [np.asarray(frame) for frame in frames],
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
    )
    return [gif_path, mp4_path]


def _rgb(env: ManiSkillSimulatorAdapter, observation: Any, view: str) -> Any:
    return env.render_rgb() if view == "render" else observation.cameras[0].rgb


def _select_proposal(round_dir: Path, requested: str | None) -> Path:
    proposals = sorted((round_dir / "proposals").glob("*"))
    if requested is not None:
        candidate = round_dir / "proposals" / requested
        if not candidate.is_dir():
            raise FileNotFoundError(f"proposal not found: {candidate}")
        return candidate
    metrics = json.loads((round_dir / "metrics.json").read_text(encoding="utf-8"))
    selected = set(metrics.get("selected_task_ids") or [])
    for candidate in proposals:
        trajectory = json.loads((candidate / "alice_trajectory.json").read_text(encoding="utf-8"))
        if trajectory["compiled"]["task_id"] in selected:
            return candidate
    if not proposals:
        raise FileNotFoundError(f"no proposals under {round_dir}")
    return proposals[0]


def render_round(
    round_dir: Path,
    *,
    proposal_name: str | None,
    output_dir: Path | None,
    fps: float,
    view: str,
) -> list[Path]:
    proposal = _select_proposal(round_dir, proposal_name)
    alice = json.loads((proposal / "alice_trajectory.json").read_text(encoding="utf-8"))
    bob_path = proposal / "bob_trajectories.json"
    bob = json.loads(bob_path.read_text(encoding="utf-8")) if bob_path.is_file() else None
    config = json.loads((round_dir.parent / "config.json").read_text(encoding="utf-8"))
    environment = config["environment"]
    snapshot = _snapshot(alice["initial_snapshot"])
    destination = output_dir or round_dir / "videos" / proposal.name
    env = ManiSkillSimulatorAdapter(
        env_id=environment["env_id"],
        camera_resolution=int(environment["camera_resolution"]),
        translation_step_m=float(environment["translation_step_m"]),
        fine_translation_step_m=float(environment["fine_translation_step_m"]),
        max_episode_steps=int(environment["episode_horizon"]),
        snapshot_dir=round_dir.parent / "snapshots",
    )
    rendered: list[Path] = []
    try:
        env.reset(seed=snapshot.seed or 0)
        env.restore_snapshot(snapshot)
        observation = env.observe()
        frames = [_annotate(_rgb(env, observation, view), "Alice | initial state")]
        for index, step in enumerate(alice["steps"]):
            result = env.step(EnvAction(action_type=step["action"], code=step["action"]))
            observation = result.observation
            frames.append(_annotate(
                _rgb(env, observation, view),
                f"Alice {index:03d} | {step['action']} | grasped={result.info.get('is_grasped', False)}",
            ))
        rendered.extend(_save_video(frames, destination / "alice", fps))

        if bob is not None:
            for episode_index, episode in enumerate(bob["episodes"]):
                env.restore_snapshot(snapshot)
                env.set_goal_position(bob["goal_position"])
                observation = env.observe()
                frames = [_annotate(
                    _rgb(env, observation, view), f"Bob {episode_index:02d} | initial state"
                )]
                for index, step in enumerate(episode["steps"]):
                    result = env.step(EnvAction(action_type=step["action"], code=step["action"]))
                    observation = result.observation
                    frames.append(_annotate(
                        _rgb(env, observation, view),
                        f"Bob {episode_index:02d} step {index:03d} | {step['action']} | success={result.info.get('success', False)}",
                    ))
                rendered.extend(_save_video(
                    frames, destination / f"bob-{episode_index:02d}", fps
                ))
    finally:
        env.close()
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_dir", type=Path)
    parser.add_argument("--proposal")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--view", choices=("render", "agent"), default="render")
    args = parser.parse_args()
    paths = render_round(
        args.round_dir.resolve(),
        proposal_name=args.proposal,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        fps=args.fps,
        view=args.view,
    )
    print(json.dumps({"rendered": [str(path) for path in paths]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
