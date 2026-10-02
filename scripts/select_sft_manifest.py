#!/usr/bin/env python3
"""Select a deterministic subset from a generated SFT collection manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-slug", action="append", default=[])
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = payload.get("episodes")
    if not isinstance(rows, list):
        raise ValueError("input manifest requires episodes")
    selected = [
        row
        for row in rows
        if not args.task_slug
        or str((row.get("metadata") or {}).get("task_slug") or "") in args.task_slug
    ][: args.count]
    if len(selected) != args.count:
        raise ValueError(f"requested {args.count} rows, found {len(selected)}")
    result = {**payload, "target_success_count": args.count, "episodes": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "episodes": len(selected)}))


if __name__ == "__main__":
    main()
