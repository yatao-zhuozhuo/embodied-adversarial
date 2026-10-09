#!/usr/bin/env python3
"""Print verified two-tool SFT collection counts without modifying data."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_two_tool_sft import episode_provider


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("SFT_data/qwen_xml_v1"))
    args = parser.parse_args()
    counts: Counter[str] = Counter()
    attempts: Counter[str] = Counter()
    per_task: Counter[str] = Counter()
    for path in sorted(args.root.glob("*/*/attempt-*/episode.json")):
        try:
            episode = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        bench = str(episode.get("bench") or "unknown")
        if episode_provider(episode) is None:
            continue
        attempts[bench] += 1
        if episode.get("success") is not True or episode.get("stop_reason") != "official_success":
            continue
        if not any(turn.get("official_success") is True for turn in episode.get("turns", [])):
            continue
        counts[bench] += 1
        per_task[f"{bench}/{path.parents[1].name}"] += 1
    print(json.dumps({
        "root": str(args.root.resolve()),
        "official_successes": dict(counts),
        "attempts": dict(attempts),
        "per_task_successes": dict(sorted(per_task.items())),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
