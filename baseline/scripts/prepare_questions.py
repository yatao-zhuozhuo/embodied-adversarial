"""Deterministically select tasks, without looking at model outcomes."""
import hashlib
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "baseline/manifests"
SEED = 20261002


def ranked(values):
    return sorted(values, key=lambda v: hashlib.sha256(f"{SEED}:{v}".encode()).hexdigest())


def write(bench, rows, selection):
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = []
    for i, row in enumerate(rows):
        episodes.append({
            "episode_id": f"baseline-{bench}-{i:02d}",
            "env_id": row["env_id"], "task": row["task"], "seed": SEED + i,
            "max_turns": 100, "max_tool_calls": 200, "timeout_s": 3600,
            "max_total_tokens": 5000000,
            "metadata": {**row.get("metadata", {}), "benchmark": bench},
        })
    data = {"selection": selection, "episodes": episodes}
    (OUT / f"{bench}_10.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    plans = ROOT / "baseline/configs/evaluation"
    plans.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "openeta.evaluation_plan.v1", "plan_id": f"baseline-{bench}-10-v1",
        "episodes": episodes,
        "variants": [{"variant_id": "baseline", "runtime": {"visual_history": {
            "enabled": True, "recent_main_turns": 1, "include_initial_main": False,
            "include_current_wrist": True, "include_vdm": False,
        }}}],
        "repeats": 1, "schedule": {"shuffle_seed": SEED},
        "execution": {"concurrency": 1, "provider_concurrency": 1,
                      "provider_queue_timeout_s": 180, "supervision_profile": "standard",
                      "max_attempts": 2},
    }
    (plans / f"{bench}.json").write_text(json.dumps(plan, indent=2) + "\n")


def main():
    source = json.loads((ROOT / "evaluations/manifests/libero_40_pass1_seed0.json").read_text())
    rows, pools = [], {}
    for suite, count in [("libero_spatial", 3), ("libero_object", 3), ("libero_goal", 2), ("libero_10", 2)]:
        pool = {r["env_id"]: r for r in source["episodes"] if r["metadata"]["suite"] == suite}
        pools[suite] = ranked(pool)
        rows += [pool[k] for k in pools[suite][:count]]
    assert len(rows) == 10 and len({r["task"] for r in rows}) == 10
    write("libero", rows, {"algorithm": "sha256(seed:env_id) ascending, stratified", "seed": SEED, "ranked_pools": pools})
    source = json.loads((ROOT / "sim/envs/metaworld/metaworld_config.json").read_text())
    descriptions = source["TASK_DESCRIPTIONS"]
    pool = ranked(set(descriptions) - {"soccer-v3", "basketball-v3"})
    rows = [{"env_id": f"openeta/metaworld_50_{name}-v0", "task": descriptions[name],
             "metadata": {"native_task": name, "suite": "metaworld_50"}} for name in pool[:10]]
    write("metaworld", rows, {"algorithm": "sha256(seed:native_task) ascending", "seed": SEED,
                             "ranked_pool": pool, "excluded": ["soccer-v3", "basketball-v3"]})
    # Fixed compatibility pool: single Panda, Cartesian delta control,
    # tabletop tasks with procedural objects. Not the full ManiSkill suite.
    descriptions = {
        "PickCube-v1": "Pick up the cube and move it to the goal marked by the green sphere; keep the robot still at the end.",
        "PushCube-v1": "Push the cube onto the goal region.",
        "PullCube-v1": "Pull the cube onto the goal region.",
        "StackCube-v1": "Stack the red cube on the green cube and release it.",
        "PlaceSphere-v1": "Place the sphere into the shallow bin and release it; keep the robot still at the end.",
        "PegInsertionSide-v1": "Insert the peg into the side hole of the box.",
        "PlugCharger-v1": "Insert the charger into the receptacle.",
        "PullCubeTool-v1": "Use the L-shaped tool to pull the cube into the goal region.",
        "RollBall-v1": "Roll the ball into the goal region.",
        "PushT-v1": "Push the T-shaped block to align with the target T-shaped region.",
        "StackPyramid-v1": "Stack the cubes into a pyramid: two cubes below and one cube on top.",
    }
    pool = ranked(descriptions)
    rows = [{"env_id": f"openeta/maniskill_{name}-v0", "task": descriptions[name],
             "metadata": {"native_task": name, "benchmark_version": "ManiSkill3 3.0.1",
                          "robot": "panda", "control_mode": "pd_ee_delta_pose"}} for name in pool[:10]]
    write("maniskill3", rows, {"algorithm": "sha256(seed:native_task) ascending", "seed": SEED,
                              "ranked_pool": pool, "scope": "single-Panda procedural tabletop compatibility subset"})
    registry = ROOT / "baseline/.runtime/sources/robocasa/robocasa/utils/dataset_registry.py"
    if registry.exists():
        pools, horizons = {}, {}
        nodes = ast.parse(registry.read_text()).body
        for node in nodes:
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            name = getattr(node.targets[0], "id", "")
            if name not in ("ATOMIC_TASK_DATASETS", "COMPOSITE_TASK_DATASETS"):
                continue
            kind = "atomic" if name.startswith("ATOMIC") else "composite"
            pools[kind] = ranked([kw.arg for kw in node.value.keywords])
            for kw in node.value.keywords:
                horizons[kw.arg] = next(ast.literal_eval(v.value) for v in kw.value.keywords if v.arg == "horizon")
        target_node = next(n for n in nodes if isinstance(n, ast.Assign)
                           and getattr(n.targets[0], "id", "") == "TARGET_TASKS")
        target = {k.arg: ast.literal_eval(k.value) for k in target_node.value.keywords}
        pools = {"atomic": ranked(target["atomic_seen"]),
                 "composite": ranked(target["composite_seen"] + target["composite_unseen"])}
        rows = [{"env_id": f"openeta/robocasa_target_{name}-v0",
                 "task": f"Complete the {name} task according to the environment's episode language instruction.",
                 "metadata": {"native_task": name, "split": "target", "task_family": kind,
                              "native_horizon": horizons[name], "robot": "PandaOmron",
                              "instruction_source": "reset episode metadata; freeze after probe"}}
                for kind in ("atomic", "composite") for name in pools[kind][:5]]
        exclusion_file = ROOT / "baseline/configs/task_exclusions.json"
        exclusions = json.loads(exclusion_file.read_text()).get("robocasa", {}) if exclusion_file.exists() else {}
        selected = {r["metadata"]["native_task"] for r in rows}
        for row in rows:
            name = row["metadata"]["native_task"]
            if name in exclusions:
                kind = row["metadata"]["task_family"]
                replacement = next(n for n in pools[kind] if n not in selected and n not in exclusions)
                assert replacement == exclusions[name]["replacement"]
                selected.add(replacement)
                row["env_id"] = f"openeta/robocasa_target_{replacement}-v0"
                row["task"] = f"Complete the {replacement} task according to the environment's episode language instruction."
                row["metadata"].update(native_task=replacement, native_horizon=horizons[replacement], replaces=name)
        write("robocasa", rows, {"algorithm": "sha256(seed:native_task) ascending, 5 atomic + 5 composite",
                                "seed": SEED, "scope": "official target50", "ranked_pools": pools,
                                "exclusions": exclusions})


if __name__ == "__main__":
    main()
