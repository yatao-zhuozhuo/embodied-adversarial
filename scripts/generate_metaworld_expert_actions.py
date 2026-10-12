#!/usr/bin/env python3
"""Generate successful, seed-addressed MetaWorld actions with official policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.env_registry import _make_metaworld_direct

POLICIES = {
    "reach-v3": "SawyerReachV3Policy",
    "bin-picking-v3": "SawyerBinPickingV3Policy",
    "assembly-v3": "SawyerAssemblyV3Policy",
    "lever-pull-v3": "SawyerLeverPullV3Policy",
    "disassemble-v3": "SawyerDisassembleV3Policy",
    "window-open-v3": "SawyerWindowOpenV3Policy",
    "coffee-push-v3": "SawyerCoffeePushV3Policy",
    "dial-turn-v3": "SawyerDialTurnV3Policy",
    "faucet-open-v3": "SawyerFaucetOpenV3Policy",
    "hammer-v3": "SawyerHammerV3Policy",
    "peg-insert-side-v3": "SawyerPegInsertionSideV3Policy",
    "shelf-place-v3": "SawyerShelfPlaceV3Policy",
    "sweep-v3": "SawyerSweepV3Policy",
    "box-close-v3": "SawyerBoxCloseV3Policy",
    "door-unlock-v3": "SawyerDoorUnlockV3Policy",
    "button-press-topdown-wall-v3": "SawyerButtonPressTopdownWallV3Policy",
    "door-open-v3": "SawyerDoorOpenV3Policy",
    "drawer-close-v3": "SawyerDrawerCloseV3Policy",
    "handle-press-side-v3": "SawyerHandlePressSideV3Policy",
    "handle-pull-v3": "SawyerHandlePullV3Policy",
    "pick-place-wall-v3": "SawyerPickPlaceWallV3Policy",
    "push-back-v3": "SawyerPushBackV3Policy",
    "plate-slide-side-v3": "SawyerPlateSlideSideV3Policy",
    "peg-unplug-side-v3": "SawyerPegUnplugSideV3Policy",
    "stick-pull-v3": "SawyerStickPullV3Policy",
}


def collect(task: dict, seed: int, max_steps: int) -> dict:
    import metaworld
    from metaworld import policies

    task_id = task["task_id"]
    policy_name = POLICIES[task_id]
    env = _make_metaworld_direct(task_id, image_width=256, image_height=256, seed=seed)
    policy = getattr(policies, policy_name)()
    actions: list[list[float]] = []
    clipped_actions = 0
    try:
        observation, _ = env.reset(seed=seed)
        # OpenETA reset_env settles MetaWorld for five control steps with its
        # default open-gripper latch before supplying the first observation.
        # Match that exact starting point; these are host steps, not decisions.
        for _ in range(5):
            observation, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, -1.0]))
        final_info = {}
        for _ in range(max_steps):
            action = np.asarray(policy.get_action(observation), dtype=np.float64)
            if action.shape != (4,) or not np.all(np.isfinite(action)):
                raise ValueError(f"invalid expert action for {task_id}: {action}")
            # MetaWorld itself clips scripted-policy commands. Record the
            # bounded command that the environment actually executes.
            bounded = np.clip(action, -1.0, 1.0)
            clipped_actions += int(np.any(bounded != action))
            observation, _, terminated, truncated, final_info = env.step(action)
            actions.append(bounded.tolist())
            if final_info.get("success") == 1 or terminated or truncated:
                break
        return {
            "schema_version": "openeta.expert_action_source.v1",
            "source_kind": "official_metaworld_scripted_policy",
            "source_dataset": "Farama-Foundation/Metaworld",
            "source_version": getattr(metaworld, "__version__", "unreported"),
            "environment_seed_protocol": env._openeta_seed_protocol,
            "policy": policy_name,
            "bench": "metaworld",
            "task_id": task_id,
            "env_id": task["env_id"],
            "task": task["task"],
            "seed": seed,
            "source_success": final_info.get("success") == 1,
            "source_steps": len(actions),
            "clipped_policy_actions": clipped_actions,
            "action_conversion": "metaworld_native_clip_to_minus1_plus1",
            "source_start": "openeta_reset_plus_five_open_gripper_settle_steps",
            "actions": actions,
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "SFT_data/configs/deepseek_xml_debug5.v1.json")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "SFT_data/expert_replay_v1/sources/metaworld")
    parser.add_argument("--target-per-task", type=int, default=5)
    parser.add_argument("--max-attempts-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed-start", type=int, default=30000000)
    args = parser.parse_args()
    if args.target_per_task < 1 or args.max_attempts_per_task < 1 or args.max_steps < 1:
        raise ValueError("counts must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    tasks = [row for row in manifest["tasks"] if row["bench"] == "metaworld"]
    if any(row["task_id"] not in POLICIES for row in tasks):
        raise ValueError("manifest contains a MetaWorld task without an audited expert policy")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for task_index, task in enumerate(tasks):
        task_dir = args.output_root / task["task_id"]
        task_dir.mkdir(parents=True, exist_ok=True)
        accepted = len(list(task_dir.glob("seed-*.json")))
        for attempt in range(args.max_attempts_per_task):
            if accepted >= args.target_per_task:
                break
            seed = args.seed_start + task_index * 10000 + attempt
            path = task_dir / f"seed-{seed}.json"
            if path.exists():
                continue
            result = collect(task, seed, args.max_steps)
            if result["source_success"]:
                payload = json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
                path.write_text(payload, encoding="utf-8")
                accepted += 1
            print(json.dumps({"task": task["task_id"], "seed": seed,
                              "source_success": result["source_success"],
                              "steps": result["source_steps"], "accepted": accepted}), flush=True)
        if accepted < args.target_per_task:
            print(json.dumps({"task": task["task_id"], "status": "shortfall",
                              "accepted": accepted, "target": args.target_per_task}), flush=True)


if __name__ == "__main__":
    main()
