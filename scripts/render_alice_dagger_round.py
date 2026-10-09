#!/usr/bin/env python3
"""Replay an Alice DAgger trajectory and export simulator video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from adapter.protocol import EnvAction
from scripts.render_embodied_selfplay_round import _annotate, _rgb, _save_video, _snapshot


def _select(records: list[dict], seed: int | None) -> dict:
    if seed is not None:
        for record in records:
            if int(record["seed"]) == seed:
                return record
        raise ValueError(f"seed {seed} is not present in this round")
    return next(
        (record for record in records if record["valid"]),
        next((record for record in records if record["grasped"]), records[0]),
    )


def render(
    round_dir: Path,
    *,
    source: str,
    seed: int | None,
    output_dir: Path | None,
    fps: float,
    view: str,
) -> list[Path]:
    records = json.loads((round_dir / source).read_text(encoding="utf-8"))
    record = _select(records, seed)
    config = json.loads((round_dir.parent / "config.json").read_text(encoding="utf-8"))
    environment = config["environment"]
    snapshot = _snapshot(record["snapshot"])
    destination = output_dir or round_dir / "videos"
    stem = destination / f"{Path(source).stem}-seed-{record['seed']}"
    env = ManiSkillSimulatorAdapter(
        env_id=environment["env_id"],
        camera_resolution=int(environment["camera_resolution"]),
        translation_step_m=float(environment["translation_step_m"]),
        fine_translation_step_m=float(environment["fine_translation_step_m"]),
        max_episode_steps=int(environment["episode_horizon"]),
        snapshot_dir=round_dir.parent / "snapshots",
    )
    try:
        env.reset(seed=int(record["seed"]))
        env.restore_snapshot(snapshot)
        observation = env.observe()
        frames = [_annotate(
            _rgb(env, observation, view),
            f"Alice model-only seed {record['seed']} | initial state",
        )]
        for index, action in enumerate(record["actions"]):
            result = env.step(EnvAction(action_type=action, code=action))
            observation = result.observation
            frames.append(_annotate(
                _rgb(env, observation, view),
                f"Alice {index:03d} | {action} | grasped={result.info.get('is_grasped', False)}",
            ))
        return _save_video(frames, stem, fps)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("round_dir", type=Path)
    parser.add_argument(
        "--source",
        choices=("model_only_eval.json", "train_trajectories.json"),
        default="model_only_eval.json",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument("--view", choices=("render", "agent"), default="render")
    args = parser.parse_args()
    paths = render(
        args.round_dir.resolve(),
        source=args.source,
        seed=args.seed,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        fps=args.fps,
        view=args.view,
    )
    print(json.dumps({"rendered": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
