#!/usr/bin/env python3
"""Audit exported OpenETA ManiSkill SFT rows and per-task episode quotas."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--accepted-index", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--require-complete-quota", action="store_true")
    args = parser.parse_args()

    samples = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    episodes = [
        json.loads(line)
        for line in args.accepted_index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))["tasks"]
    expected = {str(row["task_slug"]): int(row["quota"]) for row in catalog}
    observed = Counter(str(row.get("task_slug") or "unknown") for row in episodes)
    errors: list[str] = []
    sample_ids = [str(row.get("sample_id") or "") for row in samples]
    episode_ids = [str(row.get("episode_id") or "") for row in episodes]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("duplicate sample_id")
    if len(episode_ids) != len(set(episode_ids)):
        errors.append("duplicate episode_id")
    if any(row.get("split") != "train" for row in samples):
        errors.append("non-train sample found")
    if any(
        (row.get("metadata") or {}).get("teacher_model") != args.teacher_model
        for row in samples
    ):
        errors.append("teacher_model mismatch")
    if any((row.get("metadata") or {}).get("episode_success") is not True for row in samples):
        errors.append("sample without episode_success=true")
    quota_gaps = {
        task: quota - observed.get(task, 0)
        for task, quota in expected.items()
        if observed.get(task, 0) < quota
    }
    if args.require_complete_quota and quota_gaps:
        errors.append("task quota incomplete")
    payload = {
        "schema_version": "openeta.sft_dataset_audit.v1",
        "valid": not errors,
        "teacher_model": args.teacher_model,
        "sample_count": len(samples),
        "accepted_episode_count": len(episodes),
        "task_counts": dict(sorted(observed.items())),
        "quota_gaps": quota_gaps,
        "errors": errors,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
