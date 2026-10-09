"""Collect non-holdout tasks with bounded failure rounds and a verified success target."""

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_two_tool_sft import ACTION_DIMS, _normalise_task, _validate_manifest, _write_json
from scripts.export_two_tool_sft import samples_from_episode
from scripts.generate_two_tool_sft_manifest import _candidate_key, _pool

MODEL = "gemini-3.6-flash"
DOCS = ("docs/trajectory-input-unification.zh-CN.md", "docs/trajectory-xml-protocol-unification.zh-CN.md")


def prepare(catalog_root, prior_manifest, holdout):
    prior = json.loads(prior_manifest.read_text())["tasks"]
    prior_keys = {(row["bench"], row["task_id"].casefold()) for row in prior}
    prior_text = {(row["bench"], _normalise_task(row["task"])) for row in prior}
    prior_envs = {row["env_id"] for row in prior}
    for ledger in (ROOT / "SFT_data").glob("*/*.attempts.jsonl"):
        for line in ledger.read_text().splitlines():
            if not line.strip():
                continue
            attempt = json.loads(line)
            prior_envs.add(attempt["env_id"])
            bench = ledger.name.split(".")[0]
            prior_keys.add((bench, attempt["task_id"].removeprefix("pretrain_").removeprefix("target_").casefold()))
    documents = "\n".join((ROOT / name).read_text() for name in DOCS)
    tasks, skipped = [], []
    for bench in ACTION_DIMS:
        catalog = json.loads((catalog_root / f"{bench}.catalog.json").read_text())
        rows = _pool(bench, catalog, holdout, robocasa_holdout_keys=[])
        seen = set()
        for row in rows:
            key = _candidate_key(bench, row)
            description = _normalise_task(row["task_description"])
            identity = description if bench == "libero" else key.casefold()
            reason = None
            if row["id"] in prior_envs or (bench, key.casefold()) in prior_keys or (bench, description) in prior_text:
                reason = "previous_comparison_task"
            elif key in documents or (bench == "maniskill" and key.removesuffix("-v1") in documents):
                reason = "named_in_protocol_documents"
            elif bench == "maniskill" and key == "PickClutterYCB-v1":
                reason = "installed_evaluate_always_returns_false"
            elif identity in seen:
                reason = "duplicate_task_alias"
            if reason:
                skipped.append({"bench": bench, "env_id": row["id"], "reason": reason})
                continue
            seen.add(identity)
            tasks.append({"bench": bench, "task_id": re.sub(r"[^a-zA-Z0-9_-]", "_", key),
                          "env_id": row["id"], "task": row["task_description"],
                          "expected_action_dim": ACTION_DIMS[bench]})
    _validate_manifest(tasks, holdout, require_full=False)
    random.Random(20261006).shuffle(tasks)
    return {"schema_version": "openeta.two_tool_sft_manifest.v1", "tasks": tasks,
            "target_successes_per_task": 1, "target_total_successes": 100,
            "counts": dict(Counter(row["bench"] for row in tasks)), "skipped": skipped,
            "document_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in DOCS},
            "holdout": holdout, "previous_task_manifest": str(prior_manifest),
            "scope": "New tasks only; no old repeated successes count toward target; finite pool may fall short"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, default=ROOT / "SFT_data/reports/flash_expansion")
    parser.add_argument("--prior-manifest", type=Path, default=ROOT / "SFT_data/configs/deepseek_xml_debug5.v1.json")
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--port-base", type=int, default=9566)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-rounds", type=int, default=2)
    parser.add_argument("--adopt-existing", action="store_true")
    args = parser.parse_args()
    if args.target < 1 or args.retry_rounds < 0:
        parser.error("target must be positive")
    if args.concurrency < 1 or not 1 <= args.port_base <= 65536 - args.concurrency:
        parser.error("concurrency must be positive and all worker ports must be valid")
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lockfile = (output / "campaign.lock").open("a")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    holdout_path = ROOT / "SFT_data/configs/two_tool_test_holdout.v1.json"
    holdout = json.loads(holdout_path.read_text())
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        _write_json(manifest_path, prepare(args.catalog_root, args.prior_manifest, holdout))
    manifest = json.loads(manifest_path.read_text())
    _validate_manifest(manifest["tasks"], holdout, require_full=False)
    print(json.dumps({"event": "pool", "counts": manifest["counts"], "target": args.target}), flush=True)
    if args.prepare_only:
        return
    if not os.environ.get("CODINGROUTER_API_KEY"):
        parser.error("CODINGROUTER_API_KEY required")
    allowed = {row["env_id"]: row for row in manifest["tasks"]}
    report = output / "reports"
    report.mkdir(exist_ok=True)
    state_path = output / "campaign.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"scheduled": {}, "successes": [], "completed": {}}
    for entry in state["scheduled"].values():
        process_path = Path(f"/proc/{entry.get('pid', -1)}/cmdline")
        if process_path.exists() and str(output).encode() in process_path.read_bytes() and not args.adopt_existing:
            raise RuntimeError("An existing campaign child is still active; refusing concurrent recovery")
    state.update(target=args.target, model=MODEL, thinking_mode="provider_default", status="running",
                 concurrency=args.concurrency, retry_rounds=args.retry_rounds, scheduling="shared_pool_round_barrier")
    history = state.setdefault("attempt_history", {})
    previous = dict(state["scheduled"])
    state["successes"] = []
    for path in output.glob("*/*/attempt-*/episode.json"):
        record = json.loads(path.read_text())
        identity = record["env_id"]
        attempt_index = int(path.parent.name.split("-")[1])
        job_key = f"{identity}::round={attempt_index}"
        state["scheduled"].setdefault(identity, {"episode": str(path), "recovered": True})
        if record.get("stop_reason") and identity not in state.get("active", []):
            history.setdefault(job_key, {"episode": str(path), "stop_reason": record["stop_reason"],
                                         "success": False, "recovered": True})
        if record.get("success") and samples_from_episode(path, allowed, holdout):
            state["successes"].append(identity)
            history[job_key] = {"episode": str(path), "success": True, "recovered": True}
    state["successes"] = sorted(set(state["successes"]))
    condition = threading.Condition()
    active = set()
    children = {}
    interrupted = threading.Event()

    def persist():
        state.update(active=sorted(active), success_count=len(state["successes"]), updated_at=time.time())
        _write_json(state_path, state)

    def stop(signum, frame):
        interrupted.set()
        with condition:
            for process in list(children.values()):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
            condition.notify_all()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    persist()

    def lane(index, jobs, round_index):
        while True:
            with condition:
                while len(state["successes"]) + len(active) >= args.target and len(state["successes"]) < args.target and not interrupted.is_set():
                    condition.wait(timeout=10)
                if not jobs or interrupted.is_set() or len(state["successes"]) >= args.target or state.get("api_blocked"):
                    return
                row = jobs.popleft()
                identity, bench = row["env_id"], row["bench"]
                job_key = f"{identity}::round={round_index}"
                active.add(identity)
                old = previous.get(identity, {}) if args.adopt_existing and round_index == 0 else {}
                old_process = Path(f"/proc/{old.get('pid', -1)}/cmdline")
                adopt_command = old_process.read_bytes() if old_process.exists() else b""
                adopt = str(output).encode() in adopt_command and identity.encode() in adopt_command
                state["scheduled"][identity] = {"bench": bench, "task_id": row["task_id"], "started_at": time.time(),
                                                "round": round_index, "adopted": adopt}
                if adopt:
                    state["scheduled"][identity]["pid"] = old["pid"]
                persist()
            env = os.environ.copy()
            env.update(OPENETA_TWO_TOOL_MANIFEST=str(manifest_path), OPENETA_TWO_TOOL_MCP_LOG_DIR=str(report / f"worker-{index}"),
                       OPENETA_TWO_TOOL_MCP_PORT=str(args.port_base + index))
            command = ["bash", "scripts/run_two_tool_sft_collection.sh", "--provider", "gemini", "--model", MODEL,
                       "--bench", bench, "--env-id", identity, "--output-root", str(output), "--export-root", str(output / "exports"),
                       "--task-limit", "1", "--attempt-limit", "1", "--max-attempts-per-task", str(round_index + 1), "--target-per-task", "1",
                       "--retain-failures", "--unified-input", "--max-turns", "60", "--max-control-steps", "400",
                       "--max-tokens", "8192", "--model-timeout", "300", "--model-max-attempts", "5"]
            result = {}
            try:
                if adopt:
                    while old_process.exists():
                        try:
                            if old_process.read_bytes() != adopt_command:
                                break
                        except FileNotFoundError:
                            break
                        if interrupted.wait(2):
                            os.killpg(old["pid"], signal.SIGINT)
                    returncode = None
                else:
                    with (report / f"{bench}.{row['task_id']}.round-{round_index}.log").open("a") as stream:
                        with condition:
                            if interrupted.is_set():
                                return
                            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                            children[identity] = process
                            state["scheduled"][identity]["pid"] = process.pid
                            persist()
                    returncode = process.wait()
                paths = sorted((output / bench / row["task_id"]).glob("attempt-*/episode.json"))
                path = paths[-1] if paths else output / bench / row["task_id"] / f"attempt-{round_index:04d}/episode.json"
                episode = json.loads(path.read_text()) if path.exists() else {}
                verified = bool(episode.get("success") and samples_from_episode(path, allowed, holdout))
                result = {"returncode": returncode, "success": verified, "stop_reason": episode.get("stop_reason"),
                          "error": episode.get("error"), "episode": str(path), "finished_at": time.time()}
                for call in episode.get("model_calls", []):
                    if call.get("retry_exhausted") or any(attempt.get("http_status") in {401, 402, 403} for attempt in call.get("attempts", [])):
                        result["api_blocked"] = True
            except Exception as error:
                result = {"success": False, "error": f"{type(error).__name__}: {error}", "stop_reason": "supervisor_error"}
            finally:
                with condition:
                    active.discard(identity)
                    children.pop(identity, None)
                    state["completed"][identity] = result
                    history[job_key] = result
                    if result.get("success"):
                        state["successes"].append(identity)
                    if result.get("api_blocked"):
                        state["api_blocked"] = True
                    persist()
                    condition.notify_all()
                print(json.dumps({"event": "task_finished", "env_id": identity, **result}), flush=True)

    for round_index in range(args.retry_rounds + 1):
        state["round"] = round_index
        rows = [row for row in manifest["tasks"] if row["env_id"] not in state["successes"]
                and f"{row['env_id']}::round={round_index}" not in history]
        if round_index == 0 and args.adopt_existing:
            rows.sort(key=lambda row: 0 if row["env_id"] in previous else 1)
        jobs = deque(rows)
        _write_json(report / f"round-{round_index}.tasks.json", {"round": round_index, "tasks": rows})
        persist()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(lane, index, jobs, round_index) for index in range(args.concurrency)]
            for future in futures:
                future.result()
        if interrupted.is_set() or state.get("api_blocked") or len(state["successes"]) >= args.target:
            break
    state["status"] = ("interrupted" if interrupted.is_set() else "target_reached" if len(state["successes"]) >= args.target
                       else "api_blocked" if state.get("api_blocked") else "pool_exhausted_below_target")
    persist()
    print(json.dumps({"status": state["status"], "successes": len(state["successes"]), "target": args.target}), flush=True)


if __name__ == "__main__":
    main()
