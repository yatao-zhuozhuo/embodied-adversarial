#!/usr/bin/env python3
"""Evaluate a Swift rollout server on a fixed JSONL dataset and save metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8121")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-task", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--run-id", default="evaluation")
    return parser.parse_args()


def _find_info(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if value.get("schema_version") == "openeta.embodied_rollout.v1":
            return value
        for nested in value.values():
            found = _find_info(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in reversed(value):
            found = _find_info(nested)
            if found is not None:
                return found
    return None


def _post(url: str, payload: dict[str, Any], timeout: float) -> Any:
    endpoint = url.rstrip("/") + "/infer/"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    args = parse_args()
    if args.samples_per_task < 1:
        raise ValueError("--samples-per-task must be positive")
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line]
    receipts: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for sample_index in range(args.samples_per_task):
            uuid = f"{args.run_id}-row-{row_index:03d}-sample-{sample_index:02d}"
            payload = {
                "infer_requests": [{
                    "messages": row["messages"],
                    "data_dict": {key: value for key, value in row.items() if key != "messages"},
                    "uuid": uuid,
                }],
                "request_config": {
                    "max_tokens": 12,
                    "temperature": args.temperature,
                    "top_p": 0.9,
                    "logprobs": False,
                    "return_details": True,
                    "n": 1,
                },
                "use_tqdm": False,
            }
            outputs = _post(args.url, payload, args.timeout)
            info = _find_info(outputs)
            if info is None:
                raise RuntimeError(f"server returned no trusted rollout receipt for {uuid}")
            receipts.append(info)

    rewards = [float(item.get("reward", 0.0)) for item in receipts]
    successes = [bool(item.get("success")) for item in receipts]
    valid_actions = [float(item.get("valid_action_fraction", 0.0)) for item in receipts]
    displacements = [
        float((item.get("compiled_task") or {}).get("displacement", 0.0))
        for item in receipts
        if item.get("role") == "alice"
    ]
    result = {
        "schema_version": "openeta.embodied_eval.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": args.run_id,
        "dataset": str(args.dataset.resolve()),
        "server_url": args.url,
        "temperature": args.temperature,
        "tasks": len(rows),
        "episodes": len(receipts),
        "successes": sum(successes),
        "success_rate": sum(successes) / len(successes) if successes else 0.0,
        "mean_reward": statistics.fmean(rewards) if rewards else 0.0,
        "mean_valid_action_fraction": statistics.fmean(valid_actions) if valid_actions else 0.0,
        "mean_alice_displacement": statistics.fmean(displacements) if displacements else None,
        "receipts": receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "receipts"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
