#!/usr/bin/env python3
"""Render an embodied Alice proposal's embedded PNG observations as video."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np


def _short_xyz(value: object) -> str:
    if not isinstance(value, list) or len(value) < 3:
        return "n/a"
    return "[" + ", ".join(f"{float(item):+.3f}" for item in value[:3]) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--seconds-per-turn", type=float, default=1.0)
    parser.add_argument("--scale", type=int, default=8)
    args = parser.parse_args()

    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    info = proposal.get("alice_rollout_info") or {}
    images = info.get("images") or []
    if not images:
        raise ValueError(
            "proposal has no embedded images; accepted proposals were compacted to hashes"
        )

    rollout_path = args.rollout
    if rollout_path is None:
        rollout_path = (
            args.proposal.parent.parent
            / "rollouts"
            / "alice"
            / f"{info['request_id']}.json"
        )
    rollout = json.loads(rollout_path.read_text(encoding="utf-8"))
    steps = rollout.get("steps") or []

    first_payload = images[0].split(",", 1)[-1]
    first = cv2.imdecode(
        np.frombuffer(base64.b64decode(first_payload), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if first is None:
        raise ValueError("failed to decode the first embedded PNG")
    frame_height, frame_width = first.shape[:2]
    video_width = frame_width * args.scale
    image_height = frame_height * args.scale
    panel_height = 128
    video_height = image_height + panel_height

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix == ".webm":
        codec = "VP90"
    elif suffix == ".avi":
        codec = "MJPG"
    elif suffix == ".mp4":
        codec = "mp4v"
    else:
        raise ValueError("output extension must be .webm, .avi, or .mp4")
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*codec),
        args.fps,
        (video_width, video_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a {codec} writer")

    repeats = max(1, round(args.fps * args.seconds_per_turn))
    total = len(images)
    try:
        for index, encoded in enumerate(images):
            payload = encoded.split(",", 1)[-1]
            image = cv2.imdecode(
                np.frombuffer(base64.b64decode(payload), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                raise ValueError(f"failed to decode image at turn {index + 1}")
            image = cv2.resize(
                image,
                (video_width, image_height),
                interpolation=cv2.INTER_NEAREST,
            )
            canvas = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            canvas[:image_height] = image
            step = steps[index] if index < len(steps) else {}
            lines = [
                f"Alice turn {index + 1:02d}/{total:02d}   action: {step.get('action', 'n/a')}",
                f"grasped: {step.get('is_grasped', 'n/a')}   goal_score: {step.get('goal_score', 'n/a')}",
                f"cube: {_short_xyz(step.get('cube_position'))}   tcp: {_short_xyz(step.get('tcp_position'))}",
                f"proposal: {proposal.get('compiled_valid')}   end: {info.get('termination_reason')}",
            ]
            for line_index, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (10, image_height + 27 + line_index * 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
            for _ in range(repeats):
                writer.write(canvas)
    finally:
        writer.release()

    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "proposal_id": proposal.get("proposal_id"),
                "request_id": info.get("request_id"),
                "turns": total,
                "codec": codec,
                "fps": args.fps,
                "seconds": total * repeats / args.fps,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
