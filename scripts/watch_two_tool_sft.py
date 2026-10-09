#!/usr/bin/env python3
"""One hourly watchdog pass for the four XML collectors; never duplicate workers."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_two_tool_sft import SCHEMA, _official_success, _write_json

BENCHES = {
    "metaworld": {"port": 8766, "tasks": 25, "attempts": 20},
    "libero": {"port": 8767, "tasks": 25, "attempts": 20},
    "robocasa": {"port": 8768, "tasks": 25, "attempts": 20},
    "maniskill": {"port": 8769, "tasks": 5, "attempts": 40},
}
REPORT_ROOT = REPO_ROOT / "SFT_data/reports/qwen_xml_v1/watchdog"


def option(arguments: list[str], name: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == name and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(name + "="):
            return argument.split("=", 1)[1]
    return None


def running_processes(root: Path) -> dict[str, list[dict]]:
    found = {bench: [] for bench in BENCHES}
    for process in Path("/proc").glob("[0-9]*"):
        try:
            status = (process / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if status == "Z" or (process / "cwd").resolve() != root.resolve():
                continue
            arguments = [part.decode() for part in (process / "cmdline").read_bytes().split(b"\0") if part]
            names = {Path(argument).name for argument in arguments}
            bench = option(arguments, "--bench")
            if bench not in found:
                continue
            if "collect_two_tool_sft.py" in names:
                kind = "collector"
            elif names & {"run_two_tool_sft_collection.sh", "continue_qwen_live_sft.py"}:
                kind = "launcher"
            else:
                continue
            found[bench].append({"pid": int(process.name), "kind": kind})
        except (OSError, ValueError, IndexError):
            continue
    return found


def progress(root: Path, bench: str) -> dict:
    settings = BENCHES[bench]
    manifest = root / f"SFT_data/manifests/two_tool_{bench}_{settings['tasks']}.v1.json"
    tasks = [row for row in json.loads(manifest.read_text())["tasks"] if row["bench"] == bench]
    if not tasks:
        raise ValueError("Empty bench manifest")
    data_root = root / "SFT_data/qwen_xml_v1"
    ledger = data_root / f"{bench}.attempts.jsonl"
    records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    attempts = Counter(record["env_id"] for record in records)
    successes = Counter()
    for task in tasks:
        for path in (data_root / bench / task["task_id"]).glob("attempt-*/episode.json"):
            record = json.loads(path.read_text())
            if (record.get("schema_version") == SCHEMA and record.get("action_source") == "qwen_api"
                    and record.get("env_id") == task["env_id"] and record.get("success") is True
                    and record.get("stop_reason") == "official_success"
                    and any(turn.get("tool") == "step_env" and turn.get("official_success") is True
                            and _official_success(turn.get("receipt"), bench=bench) for turn in record.get("turns", []))):
                successes[task["env_id"]] += 1
    pending = [task for task in tasks if successes[task["env_id"]] < 3
               and attempts[task["env_id"]] < settings["attempts"]]
    return {"successful_episodes": sum(successes.values()), "completed_attempts": len(records),
            "pending_tasks": len(pending), "target_episodes": len(tasks) * 3,
            "quota_complete": all(successes[task["env_id"]] >= 3 for task in tasks),
            "last_stop_reason": records[-1].get("stop_reason") if records else None,
            "manifest": str(manifest)}


def port_in_use(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except ConnectionRefusedError:
        return False


def collector_command(bench: str) -> list[str]:
    return ["bash", "scripts/run_two_tool_sft_collection.sh", "--bench", bench,
            "--max-turns", "60", "--max-control-steps", "400", "--target-per-task", "3",
            "--max-attempts-per-task", str(BENCHES[bench]["attempts"]),
            "--enable-thinking", "--max-tokens", "8192", "--model-timeout", "300",
            "--model-max-attempts", "5", "--model-retry-backoff", "10"]


def check_bench(root: Path, bench: str, processes: list[dict], environment: dict,
                report_root: Path, dry_run: bool = False) -> dict:
    state = {"bench": bench, "processes": processes, **progress(root, bench)}
    if processes:
        state["status"] = "running" if any(process["kind"] == "collector" for process in processes) else "starting_or_stopping"
    elif state["quota_complete"]:
        state["status"] = "quota_complete"
    elif not state["pending_tasks"]:
        state["status"] = "attempt_budget_exhausted"
    elif port_in_use(BENCHES[bench]["port"]):
        state["status"] = "port_conflict"
    elif dry_run:
        state["status"] = "would_restart"
    elif not (environment.get("INF_API_KEY") or environment.get("OPENETA_LLM_API_KEY")):
        state["status"] = "missing_credentials"
    else:
        environment = environment.copy()
        environment["OPENETA_TWO_TOOL_MANIFEST"] = state["manifest"]
        environment["OPENETA_TWO_TOOL_MCP_PORT"] = str(BENCHES[bench]["port"])
        environment["OPENETA_WORKER_LOG_DIR"] = str(report_root)
        report_root.mkdir(parents=True, exist_ok=True)
        with (report_root / f"{bench}.collection.log").open("a") as stream:
            process = subprocess.Popen(collector_command(bench), cwd=root, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        state.update(status="restart_launched", launcher_pid=process.pid)
    return state


def load_environment(path: Path) -> dict:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Watchdog environment file must be owner-only (0600)")
    values = json.loads(path.read_text())
    if not isinstance(values, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in values.items()):
        raise ValueError("Invalid watchdog environment file")
    return {**os.environ, **values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    with (REPORT_ROOT / "watchdog.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another watchdog check is already running", flush=True)
            return
        environment = load_environment(args.env_file)
        processes = running_processes(REPO_ROOT)
        report = {"checked_at": datetime.now(timezone.utc).isoformat(), "dry_run": args.dry_run, "benches": []}
        for bench in BENCHES:
            try:
                state = check_bench(REPO_ROOT, bench, processes[bench], environment, REPORT_ROOT, args.dry_run)
            except Exception as error:
                state = {"bench": bench, "status": "check_error", "error_type": type(error).__name__}
            report["benches"].append(state)
        if not args.dry_run:
            _write_json(REPORT_ROOT / "latest.json", report)
            with (REPORT_ROOT / "checks.jsonl").open("a") as stream:
                stream.write(json.dumps(report, ensure_ascii=False) + "\n")
        print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
