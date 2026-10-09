"""Preflight CodingRouter teachers, then collect the original twenty tasks each."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import collect_two_tool_sft as collector
from agent.runtime.two_tool_xml import parse_decision


MODELS = ("gemini-3.6-flash", "gemini-3.1-pro-preview")
BENCHES = ("metaworld", "libero", "maniskill", "robocasa")


def wait_for_previous_lane(report: Path, bench: str, model: str):
    path = report / f"{bench}.current.json"
    if not path.exists():
        raise ValueError(f"Missing previous lane state: {path}")
    state = json.loads(path.read_text())
    if state["model"] != model:
        raise ValueError("Previous lane model does not match continuation")
    expected = state["command"]
    print(json.dumps({"event": "waiting_for_previous_lane", "bench": bench,
                      "model": model, "pid": state["pid"]}), flush=True)
    while True:
        try:
            command = Path(f"/proc/{state['pid']}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except FileNotFoundError:
            return
        if command != expected:
            return
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--port-base", type=int, default=8966)
    parser.add_argument("--provider", choices=("gemini", "codingrouter"), default="gemini")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--attempts-per-task", type=int, default=1)
    parser.add_argument("--report-tag", default="")
    parser.add_argument("--wait-for-report", type=Path)
    args = parser.parse_args()
    if args.attempts_per_task < 1:
        parser.error("attempts-per-task must be positive")
    selected_models = args.models or (MODELS if args.provider == "gemini" else ("gpt-5.6-luna",))
    api_key = os.environ["CODINGROUTER_API_KEY"]
    for name in ("NO_PROXY", "no_proxy"):
        os.environ[name] = ",".join(filter(None, [os.environ.get(name), "api.codingrouter.ai", "localhost", "127.0.0.1"]))
    report = ROOT / "SFT_data/reports" / (args.provider + "_comparison_" + args.stamp)
    if args.report_tag:
        report = report.with_name(report.name + "_" + args.report_tag)
    report.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "SFT_data/configs/deepseek_xml_debug5.v1.json"
    source = next((ROOT / "SFT_data/deepseek_input_v1_nothink_20261006_065047/metaworld").glob("*/attempt-*/episode.json"))
    source_record = json.loads(source.read_text())
    messages = source_record["model_calls"][0]["input_package"]["messages"]
    contract = source_record.get("action_contract")

    def preflight(model):
        payload = collector._model_request(args.provider, model, messages, 8192, True)
        evidence = {"model": model, "thinking_mode": "provider_default", "request": payload,
                    "source_episode": str(source), "executed": False}
        path = report / (model + ".preflight.json")
        try:
            response = collector._call_model_with_retries(
                collector._call_model, payload, api_key=api_key, endpoint=collector.PROVIDERS[args.provider]["endpoint"],
                timeout=300, max_attempts=5, backoff=10, evidence=evidence,
                persist=lambda: collector._write_json(path, evidence))
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop" or choice["message"].get("tool_calls"):
                raise ValueError("Preflight did not return a complete text decision")
            parse_decision(choice["message"]["content"], contract=contract, remaining_steps=400)
            evidence["ok"] = True
        except Exception as error:
            evidence.update(ok=False, error=f"{type(error).__name__}: {error}".replace(api_key, "[REDACTED]"))
        collector._write_json(path, evidence)
        print(json.dumps({"event": "preflight", "model": model, "ok": evidence["ok"],
                          "error": evidence.get("error")}), flush=True)
        return evidence["ok"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        checks = list(pool.map(preflight, selected_models))
    if not all(checks):
        collector._write_json(report / "campaign.status.json", {
            "status": "preflight_failed", "models": dict(zip(selected_models, checks)),
            "episodes_launched": 0, "finished_at": time.time()})
        raise SystemExit("Preflight failed; no simulator campaign launched")

    def lane(index):
        bench = BENCHES[index]
        models = selected_models if index % 2 == 0 else tuple(reversed(selected_models))
        results = []
        for model in models:
            if args.wait_for_report:
                wait_for_previous_lane(args.wait_for_report, bench, model)
            output = ROOT / "SFT_data" / f"{args.provider}_input_v1_{model}_{args.stamp}"
            env = os.environ.copy()
            env.update(OPENETA_TWO_TOOL_MANIFEST=str(manifest),
                       OPENETA_TWO_TOOL_MCP_LOG_DIR=str(report / model),
                       OPENETA_TWO_TOOL_MCP_PORT=str(args.port_base + index))
            command = ["bash", "scripts/run_two_tool_sft_collection.sh", "--provider", args.provider,
                       "--bench", bench, "--model", model,
                       "--output-root", str(output), "--export-root", str(output / "exports"),
                       "--task-limit", "5", "--attempt-limit", str(5 * args.attempts_per_task),
                       "--max-attempts-per-task", str(args.attempts_per_task),
                       "--target-per-task", str(args.attempts_per_task), "--retain-failures", "--unified-input",
                       "--max-turns", "60", "--max-control-steps", "400", "--max-tokens", "8192",
                       "--model-timeout", "300", "--model-max-attempts", "5", "--model-retry-backoff", "10"]
            log = report / f"{model}.{bench}.log"
            with log.open("a") as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                           stdout=stream, stderr=subprocess.STDOUT)
            state = {"model": model, "bench": bench, "pid": process.pid, "started_at": time.time(),
                     "output_root": str(output), "log": str(log), "command": command}
            collector._write_json(report / f"{bench}.current.json", state)
            print(json.dumps({"event": "started", **state}), flush=True)
            state.update(returncode=process.wait(), finished_at=time.time())
            results.append(state)
            collector._write_json(report / f"{bench}.results.json", results)
            if state["returncode"]:
                break
        return results

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lane, range(4)))
    collector._write_json(report / "campaign.results.json", results)


if __name__ == "__main__":
    main()
