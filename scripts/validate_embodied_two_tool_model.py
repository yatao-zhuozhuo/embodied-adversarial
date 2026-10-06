#!/usr/bin/env python3
"""Send one real-model Swift rollout request and summarize its trusted artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request
from uuid import uuid4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--role", choices=("alice", "bob"), required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = json.loads(args.dataset.read_text(encoding="utf-8").splitlines()[0])
    if row["role"] != args.role:
        raise ValueError("role/dataset mismatch")
    row["env_request"] = {**row["env_request"], "max_steps": 3, "max_control_steps": 6}
    run_id = f"model-canary-{args.role}-{uuid4().hex[:8]}"
    payload = {
        "infer_requests": [{
            "messages": row["messages"],
            "data_dict": {key: value for key, value in row.items() if key != "messages"},
            "uuid": run_id,
        }],
        "request_config": {"max_tokens": 4096, "temperature": 0.0, "n": 1,
                           "logprobs": True, "return_details": True},
        "use_tqdm": False,
    }
    req = request.Request(
        f"http://127.0.0.1:{args.port}/infer/",
        json.dumps(payload).encode(), {"Content-Type": "application/json"}, method="POST",
    )
    with request.build_opener(request.ProxyHandler({})).open(req, timeout=900) as response:
        outputs = json.load(response)
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError("Swift returned no rollout")
    artifact_path = args.artifact_root / args.role / f"{run_id}.json"
    if not artifact_path.is_file():
        raise RuntimeError("Swift did not write the trusted rollout artifact")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact.get("protocol_version") != "openeta.two_tool_xml.v1":
        raise RuntimeError("Swift selected the wrong scheduler protocol")
    if not artifact.get("steps"):
        raise RuntimeError("model produced no scheduler turns")
    summary = {
        "role": args.role, "run_id": run_id, "turns": len(artifact["steps"]),
        "commands": [step.get("command") for step in artifact["steps"]],
        "valid_action_fraction": sum(bool(s.get("valid_action")) for s in artifact["steps"]) / len(artifact["steps"]),
        "invalid_reasons": [step.get("invalid_reason") for step in artifact["steps"]],
        "executed_control_steps": sum(int(step.get("executed_control_steps", 0)) for step in artifact["steps"]),
        "termination_reason": artifact["final_info"].get("termination_reason"),
        "reward": artifact["final_info"].get("reward"),
        "artifact": str(artifact_path),
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
