"""Check live camera/controller profiles without asking a teacher or creating SFT episodes."""

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.runtime import two_tool_input as inputs
from agent.runtime.two_tool_xml import action_contract
from agent.tools.sim_mcp import SseSimulatorMcpTransport
from scripts.collect_two_tool_sft import _save_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--sim-url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    transport = SseSimulatorMcpTransport(args.sim_url)
    rows = [row for row in json.loads(args.manifest.read_text())["tasks"] if row["bench"] == args.bench]
    for index, row in enumerate(rows):
        directory = args.output_root / args.bench / row["task_id"]
        directory.mkdir(parents=True, exist_ok=True)
        session = "input-preflight-" + uuid4().hex
        seed = 20310000 + index * 10000
        created = transport.call_tool("create_env", {"env_id": row["env_id"], "session_id": session,
                    "seed": seed, "image_width": 256, "image_height": 256}, timeout_s=180)
        if created.get("error"):
            raise RuntimeError(str(created["error"]))
        base = {"handle": created["handle"], "session_id": created.get("session_id", session)}
        try:
            profile = inputs.make_profile(args.bench, row["task_id"], created, action_contract(created))
            reset = transport.call_tool("reset_env", {**base, "seed": seed}, timeout_s=180)
            observation = transport.call_tool("observe_env", base, timeout_s=180)
            if reset.get("error") or observation.get("error"):
                raise RuntimeError(str(reset.get("error") or observation.get("error")))
            frames = _save_frame(observation, directory, 0)
            before = inputs.snapshot(observation, frames, directory, profile, "obs-0", None)
            task = inputs.task_description(row["task_id"], reset.get("task") or row["task"])
            first = inputs.build_input(task=task, profile=profile, current=before, previous=None,
                                       history=[], turn=1, max_turns=60, remaining_steps=400)
            action = [0.0] * created["action_dim"]
            action[3 if args.bench == "metaworld" else 6] = profile["native_control"]["gripper"]["open_value"]
            if args.bench == "robocasa":
                action[-1] = -1
            receipt = transport.call_tool("step_env", {**base, "action": action, "num_steps": 1, "strict": True}, timeout_s=180)
            if receipt.get("error") or receipt.get("executed_control_steps") != 1:
                raise RuntimeError(str(receipt))
            after = inputs.snapshot(receipt["observation"], _save_frame(receipt["observation"], directory, 1),
                                    directory, profile, "obs-1", 1)
            report = {"status": "passed", "kind": "host_preflight_not_teacher_trajectory", "profile": profile,
                      "initial_input": first, "after": after, "robot_delta": inputs.robot_delta(after, before)}
            (directory / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print(args.bench, row["task_id"], "passed", flush=True)
        finally:
            transport.call_tool("close_env", base, timeout_s=180)


if __name__ == "__main__":
    main()
