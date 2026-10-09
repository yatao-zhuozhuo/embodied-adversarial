#!/usr/bin/env python3
"""Annotate all 100 reverified compressed expert episodes with Gemini Flash."""

from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.annotate_expert_sft_reasoning import annotate_episode
from scripts.compress_expert_replays import selected_episodes
from scripts.replay_expert_two_tool_sft import _export_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "SFT_data/expert_replay_compressed_v1")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "SFT_data/expert_replay_compressed_reasoning_v1")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent independent Gemini turn annotations per episode")
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Allow a verified benchmark subset while other benchmarks are replaying")
    parser.add_argument("--bench", default="", help="Annotate only this benchmark")
    parser.add_argument("--benches", nargs="+", default=[],
                        help="Annotate only the listed completed benchmarks")
    parser.add_argument("--limit", type=int, default=0, help="For an explicit small pilot only")
    parser.add_argument("--prompt-key", action="store_true",
                        help="Prompt for CodingRouter key without storing it on disk or in argv")
    args = parser.parse_args()
    selected = selected_episodes(args.input_root,
                                 allow_partial=args.allow_partial or args.expected_episodes < 100)
    if args.bench:
        selected = [item for item in selected if item[1]["bench"] == args.bench]
    if args.benches:
        selected = [item for item in selected if item[1]["bench"] in args.benches]
    if len(selected) != args.expected_episodes:
        parser.error(f"Expected {args.expected_episodes} verified compressed episodes, found {len(selected)}")
    for episode_path, episode in selected:
        if episode.get("stop_reason") != "official_success" or not episode.get("terminal_frames"):
            parser.error(f"Compressed episode lacks terminal success evidence: {episode_path}")
    key = os.environ.get("CODINGROUTER_API_KEY", "")
    if not key and args.prompt_key:
        key = getpass.getpass("CodingRouter key: ")
    if not key:
        parser.error("CODINGROUTER_API_KEY is required; set it in the shell, not in source files")
    done = 0
    for episode_path, _ in selected:
        report_path = (args.output_root / "episodes" / episode_path.parts[-4] /
                       episode_path.parts[-3] / episode_path.parts[-2] / "report.json")
        if report_path.is_file():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("success") is True and (report_path.parent / "samples.jsonl").is_file():
                done += 1
                continue
        report = annotate_episode(episode_path.parent / "samples.jsonl", args.output_root,
                                  api_key=key, timeout=args.timeout, verbose=False,
                                  workers=args.workers)
        done += 1
        print(json.dumps({"annotated_episodes": done, "episode": report["source_episode"]}), flush=True)
        if args.limit and done >= args.limit:
            break
    print(json.dumps({"annotated_episodes": done,
                      "exported_samples": _export_samples(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
