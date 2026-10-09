#!/usr/bin/env python3
"""Continue a live Qwen bench after its currently running canary exits."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def process_identity(pid: int):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        return None if fields[2] == "Z" else fields[21]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True, choices=["metaworld", "libero", "robocasa", "maniskill"])
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    identity = process_identity(args.wait_pid)
    print(f"Waiting for {args.bench} canary pid={args.wait_pid}", flush=True)
    while identity is not None and process_identity(args.wait_pid) == identity:
        time.sleep(5)
    ledger_path = root / "SFT_data/qwen_xml_v1" / f"{args.bench}.attempts.jsonl"
    if not ledger_path.exists():
        raise RuntimeError("Canary has no completed result; leaving collection stopped")
    results = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    last = results[-1]
    if last["stop_reason"] in {"infrastructure_error", "interrupted", "invalid_xml", "model_output_limit"}:
        raise RuntimeError(f"Canary requires repair before continuation: {last['stop_reason']}")
    if last.get("model_protocol") != "openeta.two_tool_xml.v1" or "step_env" not in last.get("executed_tools", []):
        raise RuntimeError("Canary has not demonstrated a real XML-selected environment action")
    count = 5 if args.bench == "maniskill" else 25
    manifest = root / f"SFT_data/manifests/two_tool_{args.bench}_{count}.v1.json"
    env = os.environ.copy()
    env["OPENETA_TWO_TOOL_MANIFEST"] = str(manifest)
    env["OPENETA_TWO_TOOL_MCP_PORT"] = str(args.port)
    print(f"Starting genuine Qwen collection for {args.bench}; no expert or annotation stage", flush=True)
    result = subprocess.run([
        "bash", "scripts/run_two_tool_sft_collection.sh", "--bench", args.bench,
        "--max-turns", "60", "--max-control-steps", "400", "--enable-thinking", "--max-tokens", "8192",
    ], cwd=root, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
