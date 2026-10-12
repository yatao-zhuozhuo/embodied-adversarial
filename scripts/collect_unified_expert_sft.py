"""Execute frozen unified SFT cases using host-side experts, never evaluation cases."""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_case(case, task, reference):
    if case["phase"] != "sft_train" or task["role"] != "shared":
        raise ValueError("only frozen shared-task SFT cases are permitted")
    for field in ("task_key", "bench", "task_id", "env_id"):
        if case[field] != task[field]:
            raise ValueError(f"case/task mismatch: {field}")
    selector = case["initial_state"]
    if selector["kind"] == "reference_demo":
        if not reference or reference["source_ref_id"] != selector["source_ref_id"]:
            raise ValueError("missing frozen source reference")
        if reference["task_key"] != case["task_key"] or reference["phase"] != "sft_train":
            raise ValueError("reference phase/task mismatch")
        for field in ("reference_demo_index", "state_fingerprint"):
            if selector[field] != reference[field]:
                raise ValueError(f"reference selector mismatch: {field}")
    elif selector["kind"] != "native_seed" or reference is not None:
        raise ValueError("unsupported initial state selector")


def verify_reference(reference):
    import h5py
    import numpy as np
    from data_unified.build import fingerprint

    dataset = (ROOT / reference["dataset_uri"]).resolve()
    index = reference["reference_demo_index"]
    actual = {}
    digest = lambda value: hashlib.sha256(value).hexdigest()
    if reference["kind"] == "libero_demo_state":
        with h5py.File(dataset, "r") as handle:
            group = handle[f"data/demo_{index}"]
            state = np.asarray(group.attrs["init_state"])
            xml = group.attrs["model_file"]
            xml = xml.encode() if isinstance(xml, str) else xml
            actual["action_dim"] = group["actions"].shape[-1]
        actual["model_xml_sha256"] = digest(xml)
    elif reference["kind"] == "robocasa_demo_state":
        extra = dataset / "extras" / f"episode_{index:06d}"
        with np.load(extra / "states.npz", allow_pickle=False) as archive:
            state = archive["states"][0]
        with gzip.open(extra / "model.xml.gz", "rb") as handle:
            actual["model_xml_sha256"] = digest(handle.read())
        actual["episode_meta_sha256"] = digest((extra / "ep_meta.json").read_bytes())
    elif reference["kind"] == "maniskill_demo_state":
        arrays = {}
        with h5py.File(dataset, "r") as handle:
            group = handle[f"traj_{index}"]
            def visit(name, item):
                if isinstance(item, h5py.Dataset):
                    arrays[name] = item[0]
            group["env_states"].visititems(visit)
            actual["action_dim"] = group["actions"].shape[-1]
        actual["state_fingerprint"] = fingerprint(arrays)
        actual["metadata_sha256"] = digest(dataset.with_suffix(".json").read_bytes())
    else:
        raise ValueError("unknown source kind")
    if reference["kind"] != "maniskill_demo_state":
        actual["state_sha256"] = digest(np.asarray(state, dtype="<f8").tobytes())
        actual["state_fingerprint"] = fingerprint({"state": state}, mujoco=True)
    for field, value in actual.items():
        if reference[field] != value:
            raise ValueError(f"source integrity mismatch: {field}")


def load_cases(snapshot):
    manifest = read(snapshot / "sft_train.json")
    if manifest["phase"] != "sft_train":
        raise ValueError("not an SFT manifest")
    tasks = {task["task_key"]: task for task in manifest["tasks"]}
    refs = {item["source_ref_id"]: item for item in
            map(json.loads, (snapshot / "source_refs.jsonl").read_text().splitlines())}
    rows = []
    for case in manifest["cases"]:
        task = tasks[case["task_key"]]
        reference = refs.get(case["initial_state"].get("source_ref_id"))
        validate_case(case, task, reference)
        rows.append((case, task, reference))
    return rows


def reuse_successes(source_root, output_root, rows):
    from scripts.audit_expert_replay_success import audit_episode

    counts = {}
    identity_fields = ("task_key", "bench", "task_id", "env_id", "seed",
                       "seed_role", "cached_episode_seed", "initial_state")
    previous = {case["case_id"]: (case, task) for case, task, _ in load_cases(source_root / "snapshot")}
    task_fields = ("env_id", "task", "expected_action_dim", "robot", "horizon", "environment_config")
    for case, task, reference in rows:
        result_path = output_root / "results" / (case["case_id"] + ".json")
        if result_path.exists():
            continue
        old_case, old_task = previous.get(case["case_id"], (None, None))
        if old_case is None or any(old_case.get(field) != case.get(field) for field in identity_fields):
            continue
        if any(old_task.get(field) != task.get(field) for field in task_fields):
            continue
        relative = Path("episodes") / case["bench"] / case["task_id"] / f"seed-{case['seed']}"
        old_directory = source_root / relative
        old_record_path = old_directory / "episode.json"
        if not old_record_path.exists() or read(old_record_path).get("success") is not True:
            continue
        audit = audit_episode(old_record_path)
        if not audit["integrity_pass"]:
            raise ValueError(f"reused episode failed integrity audit: {case['case_id']}: {audit['errors']}")
        directory = output_root / relative
        directory.mkdir(parents=True, exist_ok=True)
        def link_frame(source, destination):
            destination = Path(destination)
            if destination.exists():
                destination.unlink()
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)
            return str(destination)
        if (old_directory / "frames").exists():
            shutil.copytree(old_directory / "frames", directory / "frames",
                            dirs_exist_ok=True, copy_function=link_frame)
        def relocate(value):
            if isinstance(value, str) and value.startswith(str(old_directory.resolve()) + "/"):
                return str(directory.resolve()) + value[len(str(old_directory.resolve())):]
            if isinstance(value, list):
                return [relocate(item) for item in value]
            if isinstance(value, dict):
                return {key: relocate(item) for key, item in value.items()}
            return value
        record = relocate(read(old_record_path))
        record["unified_case"] = case
        if reference is not None:
            record["unified_source_ref"] = reference
        record["reused_from_episode"] = str(old_record_path.resolve())
        record["reused_from_episode_sha256"] = hashlib.sha256(old_record_path.read_bytes()).hexdigest()
        write(directory / "episode.json", record)
        write(directory / "unified_case.json", {"case": case, "source_ref": reference})
        samples = [relocate(json.loads(line)) for line in (old_directory / "samples.jsonl").read_text().splitlines()]
        for sample in samples:
            sample["episode_id"] = str(directory.resolve())
            sample["metadata"].update(unified_case_id=case["case_id"], phase="sft_train",
                                      initial_state=case["initial_state"], split_snapshot="snapshot",
                                      reused_from_episode=str(old_record_path.resolve()))
        temporary = directory / "samples.jsonl.tmp"
        temporary.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples))
        temporary.replace(directory / "samples.jsonl")
        audit = audit_episode(directory / "episode.json")
        if not audit["integrity_pass"]:
            raise ValueError(f"copied episode failed integrity audit: {case['case_id']}: {audit['errors']}")
        write(result_path, {"case_id": case["case_id"], "success": True,
                            "stop_reason": record["stop_reason"], "samples": record["samples"],
                            "reused_from_episode": str(old_record_path.resolve())})
        counts[case["bench"]] = counts.get(case["bench"], 0) + 1
    from scripts.replay_expert_two_tool_sft import _export_samples
    with (output_root / ".export.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _export_samples(output_root)
    return counts


def execute(case, task, reference, args):
    from scripts.replay_expert_two_tool_sft import replay_one
    directory = args.output_root / "episodes" / case["bench"] / case["task_id"] / f"seed-{case['seed']}"
    write(directory / "unified_case.json", {"case": case, "source_ref": reference})
    if case["bench"] == "metaworld":
        from scripts.collect_metaworld_artificial_sft import collect_one
        record = collect_one({**task, "unified_case": case}, case["seed"],
                             args.output_root, task["horizon"])
    elif reference is None and case["bench"] == "maniskill":
        from agent.tools.sim_mcp import SseSimulatorMcpTransport
        spec = args.output_root / "native_expert_specs" / case["task_id"] / f"seed-{case['seed']}.json"
        write(spec, {"case": case, "task": task})
        source_path = args.output_root / "sources" / case["bench"] / case["task_id"] / f"seed-{case['seed']}.json"
        if not source_path.exists() or read(source_path).get("source_success") is not True:
            command = [str(ROOT / "sim/venvs/maniskill312/bin/python"),
                       str(ROOT / "scripts/generate_maniskill_native_expert.py"),
                       "--case-file", str(spec), "--output", str(source_path)]
            source_environment = dict(os.environ)
            if case["task_id"] in {"FMBAssembly1Easy-v1", "PegInsertionSide-v1", "StackPyramid-v1"}:
                compatibility = ROOT / "SFT_data/runtime/maniskill_planner_compat"
                if not (compatibility / "mplib").exists():
                    raise RuntimeError("isolated ManiSkill planner dependencies are missing")
                source_environment["PYTHONPATH"] = str(compatibility) + os.pathsep + str(ROOT)
                command.append("--motion-planning")
            subprocess.run(command, env=source_environment, check=True, timeout=1800, cwd=ROOT)
        source = read(source_path)
        if source.get("source_success") is not True:
            return {"case_id": case["case_id"], "success": False,
                    "status": "native_expert_unsuccessful", "source_steps": source.get("source_steps"),
                    "stop_reason": source.get("source_stop_reason"), "source_path": str(source_path)}
        record = replay_one(SseSimulatorMcpTransport(args.sim_url), source_path,
                            task, args.output_root, task["horizon"])
    else:
        subprocess.run([str(ROOT / "sim/venvs/libero/bin/python"), __file__,
                        "--output-root", str(args.output_root), "--verify-case", case["case_id"]],
                       check=True, timeout=180, cwd=ROOT)
        from agent.tools.sim_mcp import SseSimulatorMcpTransport
        source = {"schema_version": "openeta.expert_action_source.v1",
                  "bench": case["bench"], "task_id": case["task_id"], "env_id": case["env_id"],
                  "seed": case["seed"], "source_kind": reference["kind"],
                  "source_dataset": reference["dataset_uri"],
                  "reference_demo_index": reference["reference_demo_index"],
                  "unified_case": case, "unified_source_ref": reference,
                  "source_integrity_verified": True}
        source["post_source_hold_steps"] = args.reference_settle_steps
        if case["bench"] == "maniskill":
            source["action_conversion"] = "maniskill_pd_ee_delta_pose_controller"
        source_path = args.output_root / "sources" / case["bench"] / case["task_id"] / f"seed-{case['seed']}.json"
        write(source_path, source)
        record = replay_one(SseSimulatorMcpTransport(args.sim_url), source_path,
                            task, args.output_root, task["horizon"])
    sample_path = directory / "samples.jsonl"
    if record.get("success") and sample_path.exists():
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        for sample in samples:
            sample["metadata"].update(unified_case_id=case["case_id"], phase="sft_train",
                                      initial_state=case["initial_state"], split_snapshot="snapshot")
        temporary = sample_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples))
        temporary.replace(sample_path)
    return {"case_id": case["case_id"], "success": record.get("success", False),
            "stop_reason": record.get("stop_reason"), "samples": record.get("samples", 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bench", choices=["metaworld", "libero", "robocasa", "maniskill"])
    parser.add_argument("--sim-url", default="http://127.0.0.1:8791/sse")
    parser.add_argument("--worker-case")
    parser.add_argument("--verify-case")
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-native-failed", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reference-settle-steps", type=int, default=0)
    parser.add_argument("--case-ids-file", type=Path)
    args = parser.parse_args()
    rows = load_cases(args.output_root / "snapshot")
    selected_ids = set(read(args.case_ids_file)) if args.case_ids_file else None
    if selected_ids is not None and selected_ids - {case["case_id"] for case, _, _ in rows}:
        raise ValueError("case selection includes IDs outside the frozen SFT split")
    if args.reuse_root:
        if args.reuse_root.resolve() == args.output_root.resolve():
            raise ValueError("source and destination must differ")
        print(json.dumps(reuse_successes(args.reuse_root, args.output_root, rows)), flush=True)
        return
    if args.verify_case or args.worker_case:
        case, task, reference = next(row for row in rows if row[0]["case_id"] == (args.verify_case or args.worker_case))
        if args.verify_case:
            verify_reference(reference)
        else:
            result = execute(case, task, reference, args)
            write(args.output_root / "results" / (case["case_id"] + ".json"), result)
        return
    if not args.bench:
        parser.error("--bench required for collection")
    attempted = 0
    consecutive_errors = 0
    for case, task, reference in rows:
        if case["bench"] != args.bench:
            continue
        if selected_ids is not None and case["case_id"] not in selected_ids:
            continue
        result_path = args.output_root / "results" / (case["case_id"] + ".json")
        if result_path.exists():
            previous_result = read(result_path)
            status = previous_result.get("status")
            if previous_result.get("success") is True:
                continue
            if status != "blocked_missing_native_seed_expert" and not (
                    args.retry_native_failed and reference is None and args.bench == "maniskill"
                    and status in {"native_expert_unsuccessful", "execution_error"}) and not args.retry_failed:
                continue
            if status != "blocked_missing_native_seed_expert":
                archive = args.output_root / "result_attempts" / case["case_id"] / f"{time.time_ns()}.json"
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(result_path, archive)
                episode = args.output_root / "episodes" / case["bench"] / case["task_id"] / f"seed-{case['seed']}"
                if episode.exists():
                    destination = args.output_root / "failed_episodes" / case["bench"] / case["task_id"] / f"seed-{case['seed']}" / str(time.time_ns())
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    episode.rename(destination)
        if reference is None and args.bench not in {"metaworld", "maniskill"}:
            write(result_path, {"case_id": case["case_id"], "success": False,
                                "status": "blocked_missing_native_seed_expert"})
            continue
        if args.limit and attempted >= args.limit:
            break
        if shutil.disk_usage(args.output_root).free < 20 * 1024 ** 3:
            raise RuntimeError("collection stopped: less than 20 GiB free disk space")
        attempted += 1
        python = ROOT / ("sim/venvs/metaworld/bin/python" if args.bench == "metaworld" else ".venv/bin/python")
        log = args.output_root / "logs" / (case["case_id"].replace("/", "__") + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log.open("a") as handle:
            try:
                subprocess.run([str(python), __file__, "--output-root", str(args.output_root),
                                "--worker-case", case["case_id"], "--sim-url", args.sim_url,
                                "--reference-settle-steps", str(args.reference_settle_steps)],
                               cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=7200)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                write(result_path, {"case_id": case["case_id"], "success": False,
                                    "status": "execution_error", "error": str(error), "log": str(log)})
        result = read(result_path)
        result["elapsed_seconds"] = round(time.time() - started, 2)
        write(result_path, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        consecutive_errors = consecutive_errors + 1 if result.get("status") == "execution_error" else 0
        from scripts.replay_expert_two_tool_sft import _export_samples
        with (args.output_root / ".export.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _export_samples(args.output_root)
        if consecutive_errors >= 3:
            raise RuntimeError("collection paused after three consecutive execution errors; inspect logs")


if __name__ == "__main__":
    main()
