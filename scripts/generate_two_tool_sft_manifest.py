#!/usr/bin/env python3
"""Sample 25 distinct non-holdout tasks per available OpenETA bench.

Fail closed when a reserved task cannot be resolved against a bench catalog.
The output still needs create_env/action-dimension and success-receipt canaries
before it is used for collection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_two_tool_sft import ACTION_DIMS, _normalise_task, _validate_manifest, _write_json

PYTHONS = {
    "libero": REPO_ROOT / "sim/venvs/libero/bin/python",
    "metaworld": REPO_ROOT / "sim/venvs/metaworld/bin/python",
    "maniskill": REPO_ROOT / "sim/venvs/maniskill312/bin/python",
    "robocasa": REPO_ROOT / "sim/venvs/robocasa/bin/python",
}


def _catalog(bench: str) -> list[dict[str, Any]]:
    python = PYTHONS[bench]
    if not python.is_file():
        raise RuntimeError(f"{bench} Python is missing: {python}")
    code = (
        "import json; from sim.env_registry import hot_activate,list_envs; "
        f"assert hot_activate({bench!r}), 'bench activation failed'; "
        f"rows=[x.__dict__.copy() for x in list_envs({bench!r})]; "
    )
    if bench == "maniskill":
        code += (
            "from mani_skill.utils.registration import REGISTERED_ENVS; "
            "[(row.update(supported_robots=getattr(REGISTERED_ENVS[row['task_slug']].cls, "
            "'SUPPORTED_ROBOTS', None))) for row in rows]; "
        )
    code += "print('OPENETA_CATALOG=' + json.dumps(rows))"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MUJOCO_GL", "egl")
    if bench == "libero":
        source = REPO_ROOT / "sim/venvs/_sources/LIBERO"
        env["LIBERO_DIR"] = str(source)
    completed = subprocess.run(
        [str(python), "-c", code], cwd=REPO_ROOT, env=env, text=True,
        capture_output=True, timeout=180, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"{bench} catalog failed: {completed.stderr[-1800:]}")
    marker = next((line.split("=", 1)[1] for line in completed.stdout.splitlines()
                   if line.startswith("OPENETA_CATALOG=")), None)
    if marker is None:
        raise RuntimeError(f"{bench} catalog returned no machine-readable result")
    return json.loads(marker)


def _candidate_key(bench: str, row: dict[str, Any]) -> str:
    slug = str(row.get("task_slug") or "")
    if bench == "robocasa":
        return slug.removeprefix("pretrain_").removeprefix("target_")
    return slug


def _pool(
    bench: str, rows: list[dict[str, Any]], holdout: dict[str, Any],
    *, robocasa_holdout_keys: list[str],
) -> list[dict[str, Any]]:
    if bench == "metaworld":
        rows = [row for row in rows if row.get("suite") == "metaworld_50"]
    elif bench == "robocasa":
        rows = [row for row in rows if str(row.get("task_slug") or "").startswith("pretrain_")]
    blocked_descriptions = {_normalise_task(x) for x in holdout["tasks"][bench]}
    blocked_keys = {str(x).casefold() for x in holdout.get("canonical_excluded_task_keys", {}).get(bench, [])}
    if bench == "robocasa":
        blocked_keys.update(key.casefold() for key in robocasa_holdout_keys)
        if len(blocked_keys) != 10:
            raise RuntimeError("RoboCasa requires ten audited test task class names")
    catalog_descriptions = {_normalise_task(row.get("task_description")) for row in rows}
    if bench == "libero":
        unresolved = sorted(blocked_descriptions - catalog_descriptions)
        if unresolved:
            raise RuntimeError(f"LIBERO holdout descriptions not resolved: {unresolved}")
    catalog_keys = {_candidate_key(bench, row).casefold() for row in rows}
    if blocked_keys - catalog_keys:
        raise RuntimeError(f"{bench} holdout keys not resolved: {sorted(blocked_keys - catalog_keys)}")
    eligible: list[dict[str, Any]] = []
    for row in rows:
        key = _candidate_key(bench, row)
        if key.casefold() in blocked_keys or _normalise_task(row.get("task_description")) in blocked_descriptions:
            continue
        if bench == "maniskill" and (
            _maniskill_non_panda_or_non_task(key)
            or not any(robot in (row.get("supported_robots") or []) for robot in ("panda", "panda_wristcam"))
        ):
            continue
        eligible.append(row)
    return eligible


def _maniskill_non_panda_or_non_task(key: str) -> bool:
    # This is a preliminary metadata filter, not proof of 7D control. The
    # create_env preflight must still validate every selected environment.
    blocked_fragments = (
        "SO100", "WidowX", "TwoRobot", "RotateSingleObjectInHand",
        "Empty", "Draw", "THOR", "RoboCasaKitchen", "SceneManipulation",
    )
    return any(fragment in key for fragment in blocked_fragments)


def generate(
    holdout: dict[str, Any], *, seed: int, robocasa_holdout_keys: list[str],
    benches: tuple[str, ...] = tuple(ACTION_DIMS),
    allow_maniskill_shortfall: bool = False,
    maniskill_unavailable: tuple[str, ...] = (),
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    catalog_counts: dict[str, int] = {}
    for bench in benches:
        catalog = _catalog(bench)
        pool = _pool(bench, catalog, holdout, robocasa_holdout_keys=robocasa_holdout_keys)
        if bench == "maniskill":
            pool = [row for row in pool if row.get("task_slug") not in maniskill_unavailable]
        catalog_counts[bench] = len(pool)
        if len(pool) < 25 and not (bench == "maniskill" and allow_maniskill_shortfall and pool):
            raise RuntimeError(f"{bench} has only {len(pool)} eligible registered tasks, need 25")
        # Keep a bench's random sample stable when the other benches are
        # included or omitted (important for resumable partial collection).
        selected = random.Random(seed).sample(pool, min(len(pool), 25))
        for row in selected:
            key = _candidate_key(bench, row)
            tasks.append({
                "bench": bench,
                "task_id": re.sub(r"[^a-zA-Z0-9_-]", "_", key),
                "env_id": row["id"],
                "task": row["task_description"],
                "expected_action_dim": ACTION_DIMS[bench],
            })
    shortfall = allow_maniskill_shortfall and "maniskill" in benches and catalog_counts.get("maniskill", 25) < 25
    _validate_manifest(
        tasks, holdout, require_full=len(benches) == len(ACTION_DIMS),
        allow_maniskill_shortfall=shortfall,
    )
    return {
        "schema_version": "openeta.two_tool_sft_manifest.v1",
        "selection_seed": seed,
        "target_successes_per_task": 3,
        "target_total_successes": 3 * len(tasks),
        "eligible_pool_counts": catalog_counts,
        "maniskill_shortfall": shortfall,
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / "SFT_data/configs/two_tool_test_holdout.v1.json")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "SFT_data/manifests/two_tool_100.v1.json")
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--robocasa-holdout-keys", type=Path)
    parser.add_argument("--benches", nargs="+", choices=tuple(ACTION_DIMS), default=list(ACTION_DIMS))
    parser.add_argument("--allow-maniskill-shortfall", action="store_true")
    parser.add_argument("--maniskill-unavailable", nargs="*", default=[])
    args = parser.parse_args()
    holdout = json.loads(args.holdout.read_text(encoding="utf-8"))
    keys = []
    if args.robocasa_holdout_keys:
        keys = json.loads(args.robocasa_holdout_keys.read_text(encoding="utf-8"))
    benches = tuple(dict.fromkeys(args.benches))
    manifest = generate(
        holdout, seed=args.seed, robocasa_holdout_keys=keys, benches=benches,
        allow_maniskill_shortfall=args.allow_maniskill_shortfall,
        maniskill_unavailable=tuple(args.maniskill_unavailable),
    )
    _write_json(args.output, manifest)
    print(json.dumps({"output": str(args.output), "counts": manifest["eligible_pool_counts"]}))


if __name__ == "__main__":
    main()
