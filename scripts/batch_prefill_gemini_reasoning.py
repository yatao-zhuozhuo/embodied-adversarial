#!/usr/bin/env python3
"""Prefill per-turn Gemini caches in batches from different verified episodes.

Each request contains at most one turn from any episode, so no later frame of
the same trajectory is available when explaining its fixed current action.
The existing per-episode annotator validates actions and fills any misses.
"""

from __future__ import annotations

import argparse
from collections import deque
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runtime.two_tool_xml import parse_decision
from scripts.annotate_expert_sft_reasoning import (
    MODEL, SYSTEM, _gemini_content, _gripper_state, _reasoning_from_response,
)
from scripts.collect_two_tool_sft import PROVIDERS, _call_model, _write_json
from scripts.compress_expert_replays import selected_episodes


def _sample_hash(sample: dict) -> str:
    return hashlib.sha256(json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _parse_batch_response(response: dict, ids: list[str]) -> dict[str, str]:
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Gemini batch did not finish: {choice.get('finish_reason')}")
    content = choice["message"].get("content", "").strip()
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.I)
    payload = json.loads(content)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        raise ValueError("Gemini batch response lacks items list")
    mapped = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not isinstance(item.get("reasoning"), str):
            raise ValueError("Gemini batch item is malformed")
        if item["id"] in mapped:
            raise ValueError("Gemini batch response has duplicate IDs")
        mapped[item["id"]] = item["reasoning"]
    if set(mapped) != set(ids):
        raise ValueError("Gemini batch response IDs differ from request")
    return mapped


def _missing_turns(episode_path: Path, episode: dict, output_root: Path) -> list[tuple]:
    if episode.get("stop_reason") != "official_success" or not episode.get("terminal_frames"):
        raise ValueError(f"Episode lacks verified terminal evidence: {episode_path}")
    sample_path = episode_path.parent / "samples.jsonl"
    samples = [json.loads(line) for line in sample_path.read_text(encoding="utf-8").splitlines()]
    if len(samples) != episode.get("samples"):
        raise ValueError(f"Sample count mismatch: {episode_path}")
    result = []
    for turn, sample in enumerate(samples, 1):
        cache_path = (output_root / "episodes" / episode["bench"] / episode["task_id"] /
                      f"seed-{episode['seed']}" / "annotations" / f"turn-{turn:04d}.json")
        digest = _sample_hash(sample)
        if cache_path.is_file():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("source_sample_sha256") != digest:
                raise ValueError(f"Cached sample changed: {cache_path}")
            continue
        result.append((sample, episode, cache_path, digest))
    return result


def _wire_item(item: tuple, index: int) -> tuple[str, list[dict], str]:
    sample, episode, _, _ = item
    decision = parse_decision(sample["messages"][-1]["content"],
                              contract=episode["action_contract"])["decision"]
    action = decision["parameters"]["action"]
    gripper = action[3 if episode["bench"] == "metaworld" else 6]
    gripper_state = _gripper_state(episode["bench"], gripper)
    identifier = f"item-{index}"
    content = _gemini_content(sample, episode["action_contract"])
    content[0] = {"type": "text", "text": f"Independent example {identifier}:\n" + content[0]["text"]}
    return identifier, content, gripper_state


def run_batch(batch: list[tuple], *, api_key: str, timeout: float) -> int:
    content = [{"type": "text", "text": (
        "The following examples come from DIFFERENT successful robot episodes. "
        "For each independent example, explain only its fixed next action using its own current image and text. "
        "Do not use another example as evidence. Return only compact JSON exactly in the form "
        '{"items":[{"id":"item-1","reasoning":"one short English sentence"},...]}. '
        "Include every supplied ID once. No markdown, no future-outcome claims.\n") }]
    states = {}
    for index, item in enumerate(batch, 1):
        identifier, parts, gripper_state = _wire_item(item, index)
        states[identifier] = gripper_state
        content.extend(parts)
    wire = {"model": MODEL, "max_tokens": max(512, 180 * len(batch)),
            "provider": "gemini", "thinking_level": "minimal",
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": content}]}
    error = None
    for attempt in range(3):
        try:
            response = _call_model(wire, api_key=api_key,
                                   endpoint=PROVIDERS["gemini"]["endpoint"], timeout=timeout)
            mapped = _parse_batch_response(response, list(states))
            break
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"Gemini batch failed: {type(error).__name__}: {error}".replace(
            api_key, "[REDACTED]")) from error
    saved = 0
    for index, item in enumerate(batch, 1):
        sample, _, cache_path, digest = item
        identifier = f"item-{index}"
        try:
            reasoning, usage = _reasoning_from_response(
                {"choices": [{"finish_reason": "stop", "message": {"content": mapped[identifier]}}],
                 "usage": response.get("usage", {})},
                gripper_state=states[identifier])
        except ValueError:
            # The final per-turn pass will retry this item alone.
            continue
        _write_json(cache_path, {"schema_version": "openeta.reasoning_annotation.v1",
                                 "source_sample_sha256": digest,
                                 "sample_id": sample["sample_id"],
                                 "reasoning": reasoning,
                                 "reasoning_model": MODEL, "usage": usage,
                                 "batch_prefill": True})
        saved += 1
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path,
                        default=ROOT / "SFT_data/expert_replay_compressed_v1")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "SFT_data/expert_replay_compressed_reasoning_v1")
    parser.add_argument("--expected-episodes", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--benches", nargs="+", default=[])
    parser.add_argument("--tasks", nargs="+", default=[],
                        help="Restrict to explicit task IDs for disjoint parallel jobs")
    parser.add_argument("--seed", type=int, default=None,
                        help="Restrict to one episode seed for targeted resumable prefill")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 24:
        parser.error("batch-size must be 1..24")
    selected = selected_episodes(args.input_root, allow_partial=args.allow_partial)
    if args.benches:
        selected = [item for item in selected if item[1]["bench"] in args.benches]
    if args.tasks:
        selected = [item for item in selected if item[1]["task_id"] in args.tasks]
    if args.seed is not None:
        selected = [item for item in selected if item[1]["seed"] == args.seed]
    if len(selected) != args.expected_episodes:
        parser.error(f"Expected {args.expected_episodes} verified episodes, found {len(selected)}")
    key = os.environ.get("CODINGROUTER_API_KEY", "")
    if not key and args.prompt_key:
        key = getpass.getpass("CodingRouter key: ")
    if not key:
        parser.error("CODINGROUTER_API_KEY is required")
    queues = deque()
    for path, episode in selected:
        missing = _missing_turns(path, episode, args.output_root)
        if missing:
            queues.append((missing, 0))
    total_batches = total_saved = consecutive_failures = 0
    while queues:
        batch = []
        for _ in range(min(args.batch_size, len(queues))):
            items, position = queues.popleft()
            batch.append(items[position])
            if position + 1 < len(items):
                queues.append((items, position + 1))
        try:
            saved = run_batch(batch, api_key=key, timeout=args.timeout)
            consecutive_failures = 0
        except RuntimeError as exc:
            consecutive_failures += 1
            saved = 0
            print(json.dumps({"batch_error": str(exc), "deferred_items": len(batch),
                              "consecutive_failures": consecutive_failures}), flush=True)
            if consecutive_failures >= 3:
                raise RuntimeError("Gemini gateway failed for three consecutive batches; "
                                   "cached annotations are safe to resume") from exc
        total_batches += 1
        total_saved += saved
        print(json.dumps({"batch": total_batches, "items": len(batch),
                          "saved": saved, "total_saved": total_saved}), flush=True)
        if args.limit_batches and total_batches >= args.limit_batches:
            break


if __name__ == "__main__":
    main()
