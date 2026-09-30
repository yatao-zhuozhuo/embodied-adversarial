#!/usr/bin/env python3
"""Return the newest complete distributed trainer checkpoint and its status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(root: Path, world_size: int) -> tuple[Path | None, bool]:
    candidates: list[tuple[int, Path, bool]] = []
    for path in root.glob("*/checkpoint-*"):
        if not path.is_dir():
            continue
        required = [
            "trainer_state.json", "adapter_config.json", "adapter_model.safetensors",
            "optimizer.pt", "scheduler.pt",
            *(f"rng_state_{rank}.pth" for rank in range(world_size)),
        ]
        if any(not (path / name).is_file() or not (path / name).stat().st_size for name in required):
            continue
        try:
            state = json.loads((path / "trainer_state.json").read_text())
            step = int(state["global_step"])
            maximum = int(state["max_steps"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if step <= 0 or maximum <= 0:
            continue
        candidates.append((step, path, step >= maximum))
    if not candidates:
        return None, False
    _, path, complete = max(candidates, key=lambda candidate: (candidate[0], candidate[1].stat().st_mtime))
    return path, complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    args = parser.parse_args()
    path, complete = inspect(args.root, args.world_size)
    print(f"{path or '-'}\t{int(complete)}")


if __name__ == "__main__":
    main()
