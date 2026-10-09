#!/usr/bin/env python3
"""Index installed official LIBERO / RoboCasa demos for verified OpenETA replay.

Run with the LIBERO venv and LIBERO_CONFIG_PATH pointing to the local config.
Only creates source pointers; the replay script must still verify official success.
"""

import argparse
import json
from pathlib import Path

import h5py


ROOT = Path(__file__).resolve().parents[1]


def write_source(path: Path, payload: dict) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != payload:
            raise ValueError(f"Existing source differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "SFT_data/configs/deepseek_xml_debug5.v1.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "SFT_data/expert_replay_v1/sources")
    parser.add_argument("--per-task", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=400)
    args = parser.parse_args()
    from libero.libero.benchmark import get_benchmark

    rows = json.loads(args.manifest.read_text(encoding="utf-8"))["tasks"]
    counts = {}
    for row in rows:
        bench, task_id = row["bench"], row["task_id"]
        if bench == "libero":
            suite, index = task_id.rsplit("_task", 1)
            relative = get_benchmark(suite)().get_task_demonstration(int(index))
            dataset = ROOT / "sim/venvs/libero/assets/datasets" / relative
            if not dataset.is_file():
                counts[task_id] = {"status": "dataset_missing", "path": str(dataset)}
                continue
            with h5py.File(dataset, "r") as file:
                eligible = sorted((int(key.removeprefix("demo_")), len(group["actions"]))
                                  for key, group in file["data"].items()
                                  if key.startswith("demo_") and len(group["actions"]) <= args.max_steps)
            # Early episode IDs are easier to trace back to the published data.
            chosen = eligible[:args.per_task]
            for episode_index, length in chosen:
                seed = 32000000 + rows.index(row) * 10000 + episode_index
                path = args.output_root / bench / task_id / f"seed-{seed}.json"
                write_source(path, {"schema_version": "openeta.expert_action_source.v1",
                    "bench": bench, "task_id": task_id, "env_id": row["env_id"], "seed": seed,
                    "source_kind": "official_libero_human_demonstration",
                    "source_dataset": relative, "reference_demo_index": episode_index,
                    "source_steps": length})
            counts[task_id] = {"eligible": len(eligible), "selected": len(chosen)}
        elif bench == "robocasa":
            candidates = list((ROOT / "sim/venvs/robocasa/src/robocasa/datasets/v1.0/pretrain").glob(
                f"**/{task_id}/**/lerobot/meta/episodes.jsonl"))
            if len(candidates) != 1:
                counts[task_id] = {"status": "dataset_missing_or_ambiguous", "matches": len(candidates)}
                continue
            meta_file = candidates[0]
            eligible = sorted((int(item["episode_index"]), int(item["length"]))
                              for line in meta_file.read_text(encoding="utf-8").splitlines()
                              if (item := json.loads(line))["length"] <= args.max_steps)
            chosen = eligible[:args.per_task]
            for episode_index, length in chosen:
                seed = 33000000 + rows.index(row) * 10000 + episode_index
                path = args.output_root / bench / task_id / f"seed-{seed}.json"
                relative = meta_file.parent.parent.relative_to(
                    ROOT / "sim/venvs/robocasa/src/robocasa/datasets")
                write_source(path, {"schema_version": "openeta.expert_action_source.v1",
                    "bench": bench, "task_id": task_id, "env_id": row["env_id"], "seed": seed,
                    "source_kind": "official_robocasa_human_demonstration",
                    "source_dataset": f"robocasa/{relative}", "reference_demo_index": episode_index,
                    "source_steps": length})
            counts[task_id] = {"eligible": len(eligible), "selected": len(chosen)}
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
