#!/usr/bin/env python3
"""Collect successful ManiSkill SFT episodes in resumable quota-filling waves."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_openeta_sft import export_dataset
from scripts.generate_maniskill_sft_manifest import (
    DEFAULT_CATALOG,
    DEFAULT_SFT_ROOT,
    TEACHERS,
    TIER_LIMITS,
    _load_catalog,
    _teacher_instruction,
)


def _accepted_counts(path: Path) -> Counter[str]:
    if not path.is_file():
        return Counter()
    return Counter(
        str(row.get("task_slug") or "unknown")
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    )


def _used_seeds(raw_root: Path) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for manifest_path in raw_root.rglob("rollout/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict):
            continue
        slug = str(metadata.get("task_slug") or "")
        seed = metadata.get("seed")
        if slug and isinstance(seed, int):
            result.setdefault(slug, set()).add(seed)
    return result


def _next_seed(task_index: int, used: set[int]) -> int:
    seed = task_index * 100_000
    while seed in used:
        seed += 1
    return seed


def _build_wave(
    tasks: list[dict[str, Any]],
    *,
    teacher_id: str,
    raw_root: Path,
    accepted: Counter[str],
    used: dict[str, set[int]],
    batch_size: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    scheduled: Counter[str] = Counter()
    while len(rows) < batch_size:
        eligible = [
            (task_index, task)
            for task_index, task in enumerate(tasks)
            if accepted.get(str(task["task_slug"]), 0)
            + scheduled.get(str(task["task_slug"]), 0)
            < int(task["quota"])
        ]
        if not eligible:
            break
        # Spread attempts across tasks, including after a failed wave. Otherwise a
        # task with no successes can starve every later task in the catalog.
        task_index, task = min(
            eligible,
            key=lambda item: (
                len(used.get(str(item[1]["task_slug"]), ())) / int(item[1]["quota"]),
                item[0],
            ),
        )
        slug = str(task["task_slug"])
        seed = _next_seed(task_index, used.setdefault(slug, set()))
        used[slug].add(seed)
        scheduled[slug] += 1
        limits = TIER_LIMITS[str(task["tier"])]
        rows.append(
            {
                "episode_id": f"{teacher_id}-{slug}-seed-{seed:07d}",
                "env_id": task["env_id"],
                "task": _teacher_instruction(task, teacher_id),
                "seed": seed,
                **limits,
                "max_total_tokens": 1_000_000,
                "metadata": {
                    "dataset_schema": "openeta.maniskill_sft_collection.v1",
                    "dataset_split": "train",
                    "teacher_id": teacher_id,
                    "teacher_model": TEACHERS[teacher_id],
                    "include_objects": True,
                    "perception_mode": (
                        "structured_state"
                        if teacher_id == "glm"
                        else "rgb_plus_structured_state"
                    ),
                    "task_slug": slug,
                    "task_tier": task["tier"],
                    "require_official_reward": True,
                    "on_need_human": "fail",
                    "workspace_parent": str(raw_root.resolve()),
                },
            }
        )
    return rows


def _export_paths(root: Path, teacher_id: str) -> dict[str, Path]:
    return {
        "dataset": root / "exported" / teacher_id / "train.v1.jsonl",
        "report": root / "reports" / f"{teacher_id}.export.json",
        "accepted": root / "accepted" / teacher_id / "episodes.v1.jsonl",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=tuple(TEACHERS), required=True)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SFT_ROOT)
    parser.add_argument("--sim-url", default="http://127.0.0.1:8765/sse")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--provider-concurrency", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=2500)
    parser.add_argument("--max-waves", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.concurrency < 1 or args.provider_concurrency < 1:
        raise ValueError("batch and concurrency values must be positive")

    expected_model = TEACHERS[args.teacher]
    configured_model = os.environ.get("OPENETA_LLM_MODEL", "")
    if configured_model != expected_model:
        raise ValueError(
            f"OPENETA_LLM_MODEL must be {expected_model!r}, got {configured_model!r}"
        )
    if not os.environ.get("OPENETA_LLM_API_KEY"):
        raise ValueError("OPENETA_LLM_API_KEY is required")

    root = args.output_root.resolve()
    raw_root = root / "raw" / args.teacher
    paths = _export_paths(root, args.teacher)
    tasks = _load_catalog(args.catalog.resolve())
    target = sum(int(task["quota"]) for task in tasks)
    used = _used_seeds(raw_root)
    attempted = sum(len(values) for values in used.values())
    wave_index = 0

    while attempted < args.max_attempts:
        export_dataset(
            input_root=raw_root,
            output=paths["dataset"],
            report=paths["report"],
            accepted_index=paths["accepted"],
            teacher_id=args.teacher,
            teacher_model=expected_model,
        )
        accepted = _accepted_counts(paths["accepted"])
        accepted_total = sum(accepted.values())
        if accepted_total >= target and all(
            accepted.get(str(task["task_slug"]), 0) >= int(task["quota"])
            for task in tasks
        ):
            print(json.dumps({"status": "complete", "accepted": accepted_total}))
            return
        if args.max_waves and wave_index >= args.max_waves:
            print(json.dumps({"status": "wave_limit", "accepted": accepted_total}))
            return
        remaining_attempts = args.max_attempts - attempted
        wave = _build_wave(
            tasks,
            teacher_id=args.teacher,
            raw_root=raw_root,
            accepted=accepted,
            used=used,
            batch_size=min(args.batch_size, remaining_attempts),
        )
        if not wave:
            break
        wave_index += 1
        attempted += len(wave)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        wave_id = f"{args.teacher}-wave-{wave_index:04d}-{stamp}"
        manifest_path = root / "manifests" / "waves" / f"{wave_id}.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "openeta.maniskill_sft_wave.v1",
                    "teacher_id": args.teacher,
                    "episodes": wave,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        report_path = root / "reports" / "waves" / f"{wave_id}.batch.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "agent.cli.batch_eval",
            "--manifest",
            str(manifest_path),
            "--concurrency",
            str(args.concurrency),
            "--provider-concurrency",
            str(args.provider_concurrency),
            "--sim-url",
            args.sim_url,
            "--approvement",
            "standard",
            "--batch-id",
            wave_id,
            "--output",
            str(report_path.relative_to(REPO_ROOT)),
        ]
        completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
        if completed.returncode not in {0, 1}:
            raise RuntimeError(
                f"collection wave failed with process code {completed.returncode}: {wave_id}"
            )

    export_dataset(
        input_root=raw_root,
        output=paths["dataset"],
        report=paths["report"],
        accepted_index=paths["accepted"],
        teacher_id=args.teacher,
        teacher_model=expected_model,
    )
    accepted_total = sum(_accepted_counts(paths["accepted"]).values())
    raise SystemExit(
        f"attempt budget exhausted: accepted={accepted_total}, attempts={attempted}, target={target}"
    )


if __name__ == "__main__":
    main()
