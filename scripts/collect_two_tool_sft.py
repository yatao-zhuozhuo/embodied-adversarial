#!/usr/bin/env python3
"""Collect live teacher XML decisions executed through OpenETA's two MCP tools."""

from __future__ import annotations

import argparse
import base64
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any
from urllib import error as urlerror, request
from urllib.parse import urlparse
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.tools.sim_mcp import SseSimulatorMcpTransport
from agent.runtime import two_tool_input
from scripts import gemini_sft_provider
from agent.runtime.two_tool_xml import (
    INVALID_POLICY, PROMPT_VERSION, PROTOCOL, action_contract, parse_decision, protocol_prompt,
)

ACTION_DIMS = {"libero": 7, "metaworld": 4, "maniskill": 7, "robocasa": 12}
SCHEMA = "openeta.qwen_live_two_tool_xml.v1"
DEFAULT_ENDPOINT = "https://cqhbod8bjjjbcoakk8pmeebgkaq9akcq.openapi-sj.sii.edu.cn/v1/chat/completions"
PROVIDERS = {
    "codingrouter": {"schema": "openeta.codingrouter_live_two_tool_xml.v1",
                     "sample_schema": "openeta.sft.codingrouter_xml_decision.v1",
                     "source": "codingrouter_api", "model": "gpt-5.6-luna",
                     "endpoint": "https://api.codingrouter.ai/v1/chat/completions"},
    "gemini": {"schema": "openeta.gemini_live_two_tool_xml.v1",
               "sample_schema": "openeta.sft.gemini_xml_decision.v1",
               "source": "gemini_api", "model": "gemini-3.6-flash",
               "endpoint": "https://api.codingrouter.ai/v1beta/models/{model}:generateContent"},
    "qwen": {"schema": SCHEMA, "sample_schema": "openeta.sft.qwen_xml_decision.v1",
             "source": "qwen_api", "model": "Qwen3.8-27B", "endpoint": DEFAULT_ENDPOINT},
    "deepseek": {"schema": "openeta.deepseek_live_two_tool_xml.v1",
                 "sample_schema": "openeta.sft.deepseek_xml_decision.v1",
                 "source": "deepseek_api", "model": "deepseek-flash",
                 "endpoint": "https://api.deepseek.com/chat/completions"},
}


def episode_provider(record: dict) -> str | None:
    for name, config in PROVIDERS.items():
        if (record.get("schema_version") == config["schema"] and record.get("action_source") == config["source"]
                and record.get("provider", name) == name):
            return name
    return None
def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _official_success(payload: Any, *, bench: str) -> bool:
    if not isinstance(payload, dict):
        return False
    info = payload.get("info")
    if not isinstance(info, dict):
        return False
    # Fail closed. The per-bench canary must establish that this is emitted by
    # its official task checker; never infer success from reward or model prose.
    success = info.get("success")
    if bench == "maniskill" and isinstance(success, list):
        if len(success) != 1:
            return False
        success = success[0]
    return type(success) in (bool, int, float) and success == 1


def _robot_projection(observation: dict[str, Any]) -> dict[str, Any]:
    robot = observation.get("robot")
    robot = robot if isinstance(robot, dict) else {}
    return {
        key: robot[key]
        for key in ("joint_positions", "joint_velocities", "end_effector_pose", "gripper_state", "base_pose")
        if key in robot
    }


def _camera_items(observation: dict[str, Any], *, image_limit: int) -> list[dict[str, Any]]:
    cameras = observation.get("cameras")
    if not isinstance(cameras, list):
        return []
    return [c for c in cameras if isinstance(c, dict) and c.get("rgb_base64")][:image_limit]


def _save_frame(observation: dict[str, Any], directory: Path, turn: int) -> list[str]:
    paths = []
    for index, camera in enumerate(_camera_items(observation, image_limit=3)):
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(camera.get("frame_id") or index))
        path = directory / "frames" / f"{turn:04d}-{safe_name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(camera["rgb_base64"], validate=True))
        paths.append(str(path.relative_to(directory)))
    return paths


def _action_layout(bench: str) -> str:
    layouts = {
        "libero": "[dx,dy,dz,droll,dpitch,dyaw,gripper], gripper -1=open,+1=close",
        "metaworld": "[dx,dy,dz,gripper]",
        "maniskill": "[dx,dy,dz,droll,dpitch,dyaw,gripper], gripper +1=open,-1=close",
        "robocasa": (
            "[arm_dx,arm_dy,arm_dz,arm_droll,arm_dpitch,arm_dyaw,gripper,"
            "base_forward,base_left,base_yaw,torso,mode], gripper -1=open,+1=close, "
            "mode -1=arm,+1=base"
        ),
    }
    return layouts[bench]


def _normalise_task(value: object) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def _validate_manifest(
    tasks: list[dict[str, Any]], holdout: dict[str, Any], *, require_full: bool = True,
    allow_maniskill_shortfall: bool = False,
) -> None:
    if require_full and len(tasks) != 100 and not allow_maniskill_shortfall:
        raise ValueError(f"manifest must contain exactly 100 distinct tasks, got {len(tasks)}")
    if not tasks:
        raise ValueError("manifest must contain at least one task")
    counts = {bench: 0 for bench in ACTION_DIMS}
    seen_envs: set[str] = set()
    seen_task_ids: set[tuple[str, str]] = set()
    reserved = set(holdout["reserved_seeds"])
    for index, row in enumerate(tasks):
        bench = str(row.get("bench") or "")
        if bench not in counts:
            raise ValueError(f"task {index} has unsupported bench {bench!r}")
        counts[bench] += 1
        env_id = str(row.get("env_id") or "")
        if not env_id.startswith(f"openeta/{bench}_") or env_id in seen_envs:
            raise ValueError(f"task {index} has invalid or duplicate env_id: {env_id}")
        seen_envs.add(env_id)
        task_key = (bench, str(row.get("task_id") or ""))
        if not task_key[1] or task_key in seen_task_ids:
            raise ValueError(f"task {index} has missing or duplicate task_id: {task_key}")
        seen_task_ids.add(task_key)
        description = _normalise_task(row.get("task"))
        blocked_descriptions = {_normalise_task(x) for x in holdout["tasks"][bench]}
        if description in blocked_descriptions:
            raise ValueError(f"test holdout task in training manifest: {env_id}")
        blocked_keys = holdout.get("canonical_excluded_task_keys", {}).get(bench, [])
        if any(key.casefold() in env_id.casefold() for key in blocked_keys):
            raise ValueError(f"test holdout environment in training manifest: {env_id}")
        if row.get("seed") in reserved:
            raise ValueError(f"reserved test seed in training manifest: {env_id}")
    if require_full:
        if allow_maniskill_shortfall:
            valid = all(counts[bench] == 25 for bench in ACTION_DIMS if bench != "maniskill") and 1 <= counts["maniskill"] <= 25
        else:
            valid = all(count == 25 for count in counts.values())
        if not valid:
            raise ValueError(f"manifest requires 25 per bench except an audited ManiSkill shortfall, got {counts}")


def _system_prompt(bench: str, task: str, contract: dict, model: str = "Qwen3.8-27B") -> str:
    return (
        f"You are {model} controlling a real simulation through OpenETA. Solve the task yourself. "
        "The host supplies the initial observation. You may observe again or act. "
        "Choose every action using the actual camera images, robot state and tool feedback. "
        "Do not invent observations, use code, request hidden object coordinates, or other tools. "
        "Only the environment can confirm success. "
        "step_env repeats the same normalized action vector for num_steps (1-10) control steps. "
        "These are normalized commands, not absolute coordinates. Use longer repetitions for "
        "coarse motion and shorter ones for careful contact. Inspect the returned images to adjust. "
        f"Bench={bench}; layout={_action_layout(bench)}. Task: {task}\n" + protocol_prompt(contract)
        + "\nThe outermost element must literally be <decision>...</decision>, never <tool_call>. "
        "tool_call is only the text inside <kind>. Do not wrap the decision in any tool_call tag."
    )


def _request_messages(messages: list[dict], history_turns: int) -> list[dict]:
    starts = [index for index, message in enumerate(messages) if message["role"] == "assistant"]
    if len(starts) <= history_turns:
        return copy.deepcopy(messages)
    return copy.deepcopy(messages[:1] + messages[starts[-history_turns] - 1:])


def _provider_messages(messages: list[dict]) -> list[dict]:
    converted = copy.deepcopy(messages)
    for message in converted:
        if not isinstance(message.get("content"), list):
            continue
        for item in message["content"]:
            if item.get("type") == "image_url":
                path = Path(item["image_url"]["url"])
                item["image_url"]["url"] = "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")
    return converted


def _model_request(provider: str, model: str, messages: list[dict], max_tokens: int, enable_thinking: bool) -> dict:
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if provider == "deepseek":
        payload["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
        if enable_thinking:
            payload["reasoning_effort"] = "high"
    elif provider == "qwen":
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    elif provider == "gemini":
        if not enable_thinking:
            raise ValueError("Gemini adapter uses provider-default thinking; disabling is not implemented")
        payload["provider"] = "gemini"
    elif provider == "codingrouter":
        if not enable_thinking:
            raise ValueError("CodingRouter uses provider-default thinking")
    else:
        raise ValueError("Unsupported model provider")
    return payload


def _provider_wire(payload: dict) -> dict:
    if payload.get("provider") == "gemini":
        return gemini_sft_provider.render_request(payload)
    wire = copy.deepcopy(payload)
    wire["messages"] = _provider_messages(payload["messages"])
    return wire


def _call_model(payload: dict, *, api_key: str, endpoint: str, timeout: float) -> dict:
    wire = _provider_wire(payload)
    headers = {"Content-Type": "application/json"}
    if payload.get("provider") == "gemini":
        endpoint = gemini_sft_provider.endpoint_for_model(endpoint, payload["model"])
        headers["x-goog-api-key"] = api_key
    else:
        headers["Authorization"] = "Bearer " + api_key
    req = request.Request(endpoint, data=json.dumps(wire).encode("utf-8"),
                          headers=headers,
                          method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        raw = json.load(response)
    return gemini_sft_provider.normalize_response(raw) if payload.get("provider") == "gemini" else raw


def _retryable_model_error(error: Exception) -> bool:
    if isinstance(error, urlerror.HTTPError):
        return error.code in {408, 429, 500, 502, 503, 504, 524}
    if isinstance(error, urlerror.URLError):
        return isinstance(error.reason, (TimeoutError, ConnectionError))
    return isinstance(error, (TimeoutError, ConnectionError))


def _call_model_with_retries(caller, payload: dict, *, api_key: str, endpoint: str,
                            timeout: float, max_attempts: int, backoff: float,
                            evidence: dict, persist) -> dict:
    evidence["attempts"] = []
    for attempt_number in range(1, max_attempts + 1):
        attempt = {"attempt": attempt_number, "started_at": time.time(), "status": "pending"}
        evidence["attempts"].append(attempt)
        persist()
        started = time.monotonic()
        try:
            response = caller(copy.deepcopy(payload), api_key=api_key, endpoint=endpoint, timeout=timeout)
        except Exception as error:
            retryable = _retryable_model_error(error)
            attempt.update(status="error", error=f"{type(error).__name__}: {error}",
                           retryable=retryable, elapsed_seconds=time.monotonic() - started)
            if isinstance(error, urlerror.HTTPError):
                attempt["http_status"] = error.code
                body = error.read(8192)
                if body:
                    attempt["http_error_body"] = body.decode("utf-8", errors="replace").replace(api_key, "[REDACTED]")
                error.close()
            if not retryable or attempt_number == max_attempts:
                evidence["retry_exhausted"] = retryable
                persist()
                raise
            delay = min(backoff, 60.0)
            for _ in range(attempt_number - 1):
                delay = min(delay * 2, 60.0)
            attempt["retry_delay_seconds"] = delay
            persist()
            print(json.dumps({"event": "model_request_retry", "attempt": attempt_number,
                              "max_attempts": max_attempts, "http_status": attempt.get("http_status"),
                              "error_type": type(error).__name__, "retry_delay_seconds": delay}), flush=True)
            time.sleep(delay)
        else:
            attempt.update(status="success", elapsed_seconds=time.monotonic() - started)
            evidence["response"] = response
            persist()
            return response
    raise RuntimeError("Model request retry budget must be positive")


def _visible_message(observation: dict, frames: list[str], directory: Path, image_limit: int) -> dict:
    content = [{"type": "text", "text": json.dumps({"robot": _robot_projection(observation)}, ensure_ascii=False)}]
    content.extend({"type": "image_url", "image_url": {"url": str((directory / relative).resolve())}}
                   for relative in frames[:image_limit])
    if len(content) == 1:
        raise RuntimeError("Actual tool observation has no RGB image")
    return {"role": "user", "content": content}


def _receipt(payload, directory: Path, frames: list[str]):
    frame_by_name = {Path(path).stem.split("-", 1)[1]: str((directory / path).resolve()) for path in frames}
    def convert(value):
        if isinstance(value, list):
            return [convert(item) for item in value]
        if not isinstance(value, dict):
            return value
        result = {key: convert(item) for key, item in value.items() if key != "rgb_base64"}
        if value.get("rgb_base64"):
            name = re.sub(r"[^a-zA-Z0-9_-]", "_", str(value.get("frame_id") or ""))
            result["rgb_file"] = frame_by_name.get(name)
            result["rgb_sha256"] = hashlib.sha256(base64.b64decode(value["rgb_base64"])).hexdigest()
        return result
    return convert(payload)


def collect_attempt(transport, row: dict, *, directory: Path, seed: int, api_key: str,
                    endpoint: str, model: str, max_turns: int = 60, max_control_steps: int = 300,
                    image_limit: int = 2, timeout: float = 180, enable_thinking: bool = True,
                    history_turns: int = 5, max_tokens: int = 8192, model_call=None,
                    model_timeout: float = 300, model_max_attempts: int = 5,
                    model_retry_backoff: float = 10, provider: str = "qwen",
                    retain_failures: bool = False, unified_input: bool = False) -> dict:
    if provider not in PROVIDERS:
        raise ValueError("Unsupported model provider")
    if type(model_max_attempts) is not int or model_max_attempts < 1:
        raise ValueError("model_max_attempts must be a positive integer")
    if not math.isfinite(model_timeout) or model_timeout <= 0 or not math.isfinite(model_retry_backoff) or model_retry_backoff < 0:
        raise ValueError("Invalid model timeout or retry backoff")
    bench = row["bench"]
    directory.mkdir(parents=True, exist_ok=False)
    record = {"schema_version": PROVIDERS[provider]["schema"], "bench": bench, "env_id": row["env_id"], "task": row["task"],
              "seed": seed, "model": model, "provider": provider, "endpoint": endpoint,
              "action_source": PROVIDERS[provider]["source"], "success": False,
              "initial_frames": [], "initial_robot": {}, "turns": [], "model_calls": [],
              "session_id": provider + "-xml-" + uuid4().hex,
              "model_protocol": PROTOCOL, "output_source": "original_model_xml",
              "prompt_version": PROMPT_VERSION,
              "run_config": {"enable_thinking": None if provider in {"gemini", "codingrouter"} else enable_thinking,
                             "thinking_mode": "provider_default" if provider in {"gemini", "codingrouter"} else ("enabled" if enable_thinking else "disabled"),
                             "max_tokens": max_tokens,
                             "retain_failures": retain_failures,
                             "max_turns": max_turns, "max_control_steps": max_control_steps,
                             "history_turns": history_turns, "image_limit": image_limit,
                             "model_retry": {"timeout_seconds": model_timeout, "max_attempts": model_max_attempts,
                                             "backoff_seconds": model_retry_backoff, "backoff_cap_seconds": 60,
                                             "http_statuses": [408, 429, 500, 502, 503, 504, 524],
                                             "scope": "model_request_only_same_observation"},
                             "observation_version": "rgb256_proprioception.v1",
                             "initial_observation": "host_observe", "xml_repair": False,
                             "invalid_output_policy": INVALID_POLICY,
                             "invalid_turn_cost": 1, "invalid_retries": INVALID_POLICY["max_consecutive_invalid"] - 1,
                             "supervision": "current_decision_only", "thinking_supervision": "answer_only"}}
    record["code_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    record["source_sha256"] = {str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (Path(__file__), REPO_ROOT / "scripts/gemini_sft_provider.py", REPO_ROOT / "agent/runtime/two_tool_xml.py",
                                            REPO_ROOT / "agent/runtime/planner_xml.py", REPO_ROOT / "sim/bench_worker.py",
                                            REPO_ROOT / "sim/envs/robocasa/direct_env.py",
                                            REPO_ROOT / "sim/envs/robocasa/rendering.py",
                                            REPO_ROOT / "sim/two_tool_contract.py",
                                            REPO_ROOT / "agent/runtime/two_tool_input.py")}
    handle = ""
    session = record["session_id"]
    control_steps = 0
    requested_steps = 0
    consecutive_invalid = 0
    consecutive_observe = 0
    messages = []
    caller = model_call or _call_model
    try:
        created = transport.call_tool("create_env", {
            "env_id": row["env_id"], "task": row["task"], "seed": seed, "session_id": session,
            "render_mode": "rgb_array", "image_width": 256, "image_height": 256,
            "include_objects": False,
        }, timeout_s=timeout)
        handle = str(created.get("handle") or "")
        session = str(created.get("session_id") or session)
        if created.get("error") or not handle:
            raise RuntimeError(str(created.get("error") or "Missing environment handle"))
        record["create_receipt"] = created
        contract = action_contract(created)
        record["action_contract"] = contract
        record["control_spec"] = created.get("control_spec")
        reset = transport.call_tool("reset_env", {"handle": handle, "seed": seed, "session_id": session}, timeout_s=timeout)
        if reset.get("error"):
            raise RuntimeError(str(reset["error"]))
        task = str(reset.get("task") or row["task"])
        record["task"] = task
        initial = transport.call_tool("observe_env", {"handle": handle, "session_id": session}, timeout_s=timeout)
        if initial.get("error"):
            raise RuntimeError(str(initial["error"]))
        observation = initial.get("observation", initial)
        frames = _save_frame(observation, directory, 0)
        record["initial_frames"] = frames
        record["initial_robot"] = _robot_projection(observation)
        record["initial_receipt"] = _receipt(initial, directory, frames)
        visible = _visible_message(observation, frames, directory, image_limit)
        visible["content"].insert(0, {"type": "text", "text": json.dumps({"task": task, "control_steps_remaining": max_control_steps})})
        messages = [{"role": "system", "content": _system_prompt(bench, task, contract, model)}, visible]
        if unified_input:
            profile = two_tool_input.make_profile(bench, row["task_id"], created, contract)
            task = two_tool_input.task_description(row["task_id"], task)
            record.update(input_protocol=two_tool_input.INPUT_PROTOCOL, input_profile=profile, task=task)
            record["run_config"].update(history_turns=4, image_limit=profile["image_limit"],
                                        observation_version=two_tool_input.INPUT_PROTOCOL)
            current_snapshot = two_tool_input.snapshot(observation, frames, directory, profile, "obs-0000", None)
            previous_snapshot = None
            record["observations"] = [current_snapshot]
        _write_json(directory / "episode.json", record)
        for turn in range(1, max_turns + 1):
            input_package = None
            if unified_input:
                input_package = two_tool_input.build_input(
                    task=task, profile=profile, current=current_snapshot, previous=previous_snapshot,
                    history=record["turns"], turn=turn, max_turns=max_turns,
                    remaining_steps=max_control_steps - control_steps)
                messages = copy.deepcopy(input_package["messages"])
                previous_snapshot = current_snapshot
            model_request = _model_request(provider, model, copy.deepcopy(messages) if unified_input else _request_messages(messages, history_turns),
                                           max_tokens, enable_thinking)
            call_index = len(record["model_calls"])
            evidence = {"request": model_request, "started_at": time.time()}
            if input_package is not None:
                evidence["input_package"] = input_package
                wire = _provider_wire(model_request)
                evidence["provider_wire_sha256"] = hashlib.sha256(json.dumps(wire).encode("utf-8")).hexdigest()
            record["model_calls"].append(evidence)
            _write_json(directory / "episode.json", record)
            try:
                response = _call_model_with_retries(
                    caller, model_request, api_key=api_key, endpoint=endpoint, timeout=model_timeout,
                    max_attempts=model_max_attempts, backoff=model_retry_backoff, evidence=evidence,
                    persist=lambda: _write_json(directory / "episode.json", record))
            except Exception as error:
                evidence["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                evidence["elapsed_seconds"] = time.time() - evidence["started_at"]
            evidence["response"] = response
            evidence["token_usage"] = response.get("usage")
            choice = response["choices"][0]
            assistant = copy.deepcopy(choice["message"])
            messages.append(assistant)
            event = {"turn": turn, "model_message": assistant, "model_call_index": call_index,
                     "valid": False, "before_frames": frames,
                     "control_steps_remaining": max_control_steps - control_steps,
                     "validation": {"valid": False, "repaired": False}}
            record["turns"].append(event)
            if choice.get("finish_reason") == "length":
                event["error"] = "Truncated model output is not executable"
                record["stop_reason"] = "model_output_limit"
                break
            try:
                if assistant.get("tool_calls") or assistant.get("function_call") or choice.get("finish_reason") != "stop":
                    raise ValueError("Expected a complete XML answer, not API tool calls")
                parsed = parse_decision(assistant.get("content"), contract=contract,
                                        remaining_steps=max_control_steps - control_steps)
                if parsed["decision"]["name"] == "observe_env" and consecutive_observe >= INVALID_POLICY["max_consecutive_observe"]:
                    raise ValueError("Too many consecutive observe calls without movement")
            except (ValueError, TypeError, KeyError) as error:
                event["error"] = str(error)
                event["validation"]["error"] = str(error)
                record["stop_reason"] = "invalid_xml"
                consecutive_invalid += 1
                messages.append({"role": "user", "content": json.dumps({"error": str(error), "executed": False,
                                                                         "required_format": "One <decision> with kind, name, reasoning and parameters. The outer tag is decision, not tool_call.",
                                                                         "control_steps_remaining": max_control_steps - control_steps})})
                _write_json(directory / "episode.json", record)
                if consecutive_invalid >= INVALID_POLICY["max_consecutive_invalid"] or turn == max_turns:
                    break
                continue
            consecutive_invalid = 0
            record.pop("stop_reason", None)
            decision = parsed["decision"]
            name, arguments = decision["name"], decision["parameters"]
            event.update(parsed=parsed, decision=decision, validation={"valid": True, "repaired": False})
            if decision["kind"] == "response":
                event["valid"] = True
                record["stop_reason"] = "model_finish"
                break
            event["tool"] = name
            consecutive_observe = consecutive_observe + 1 if name == "observe_env" else 0
            event["arguments"] = arguments
            tool_request = {"handle": handle, "session_id": session, **arguments}
            if name == "step_env":
                tool_request["strict"] = True
            event["tool_request"] = {"name": name, "arguments": tool_request}
            _write_json(directory / "episode.json", record)
            payload = transport.call_tool(name, tool_request, timeout_s=timeout)
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            observation = payload.get("observation", payload)
            frames = _save_frame(observation, directory, turn)
            if unified_input:
                current_snapshot = two_tool_input.snapshot(observation, frames, directory, profile,
                                                          f"obs-{turn:04d}", turn)
                record["observations"].append(current_snapshot)
            visible = _visible_message(observation, frames, directory, image_limit)
            if name == "step_env":
                executed = payload.get("executed_control_steps")
                if type(executed) is not int or not 0 <= executed <= arguments["num_steps"]:
                    event["receipt"] = _receipt(payload, directory, frames)
                    raise RuntimeError("Missing or invalid actual control step receipt")
                event["executed_control_steps"] = executed
                control_steps += executed
                requested_steps += arguments["num_steps"]
            else:
                event["executed_control_steps"] = 0
            success = name == "step_env" and _official_success(payload, bench=bench)
            event.update(valid=True, frames=frames, robot=_robot_projection(observation),
                         reward=payload.get("reward"), official_info=payload.get("info"),
                         official_success=success, receipt=_receipt(payload, directory, frames),
                         terminated=bool(payload.get("terminated")), truncated=bool(payload.get("truncated")))
            feedback = {"ok": True, "tool": name, "terminated": event["terminated"],
                        "truncated": event["truncated"], "control_steps_remaining": max_control_steps - control_steps}
            visible["content"].insert(0, {"type": "text", "text": json.dumps(feedback)})
            messages.append(visible)
            record["messages"] = messages
            record["control_steps_requested"] = requested_steps
            record["executed_control_steps"] = control_steps
            if success:
                record["success"] = True
                record["stop_reason"] = "official_success"
            _write_json(directory / "episode.json", record)
            if success:
                break
            if event["terminated"] or event["truncated"]:
                record["stop_reason"] = "environment_ended"
                break
            if control_steps >= max_control_steps:
                record["stop_reason"] = "control_step_limit"
                break
        else:
            record["stop_reason"] = "turn_limit"
    except KeyboardInterrupt:
        record["stop_reason"] = "interrupted"
    except Exception as error:
        record["stop_reason"] = "infrastructure_error"
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        record["messages"] = messages
        if handle:
            try:
                transport.call_tool("close_env", {"handle": handle, "session_id": session}, timeout_s=30)
            except Exception as error:
                record["close_error"] = f"{type(error).__name__}: {error}"
        _write_json(directory / "episode.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bench", choices=tuple(ACTION_DIMS), required=True)
    parser.add_argument("--env-id")
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / "SFT_data/configs/two_tool_test_holdout.v1.json")
    parser.add_argument("--provider", choices=tuple(PROVIDERS), default="qwen")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--export-root", type=Path)
    parser.add_argument("--retain-failures", action="store_true")
    parser.add_argument("--unified-input", action="store_true")
    parser.add_argument("--continue-on-infrastructure-error", action="store_true")
    parser.add_argument("--sim-url", default="http://127.0.0.1:8766/sse")
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--max-turns", type=int, default=60)
    parser.add_argument("--max-control-steps", type=int, default=300)
    parser.add_argument("--max-attempts-per-task", type=int, default=20)
    parser.add_argument("--target-per-task", type=int, default=3)
    parser.add_argument("--task-limit", type=int, default=0)
    parser.add_argument("--attempt-limit", type=int, default=0)
    parser.add_argument("--history-turns", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--image-limit", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--model-timeout", type=float, default=300)
    parser.add_argument("--model-max-attempts", type=int, default=5)
    parser.add_argument("--model-retry-backoff", type=float, default=10)
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    config = PROVIDERS[args.provider]
    args.endpoint = args.endpoint or config["endpoint"]
    args.model = args.model or config["model"]
    if args.provider == "codingrouter" and not args.enable_thinking:
        parser.error("CodingRouter currently supports provider-default thinking only")
    if args.provider == "gemini":
        if not args.enable_thinking:
            parser.error("Gemini currently supports provider-default thinking only")
        args.endpoint = gemini_sft_provider.endpoint_for_model(args.endpoint, args.model)
    args.output_root = args.output_root or REPO_ROOT / f"SFT_data/{args.provider}_xml_v1"
    if args.continue_on_infrastructure_error and (not args.retain_failures or args.attempt_limit < 1):
        parser.error("Continuing infrastructure errors requires --retain-failures and a positive --attempt-limit")
    if min(args.max_turns, args.max_control_steps, args.max_attempts_per_task, args.target_per_task, args.history_turns, args.max_tokens) < 1:
        parser.error("Budgets must be positive")
    if (args.model_max_attempts < 1 or not math.isfinite(args.model_timeout) or args.model_timeout <= 0
            or not math.isfinite(args.model_retry_backoff) or args.model_retry_backoff < 0):
        parser.error("Model attempts/timeout must be positive; backoff must be finite and nonnegative")
    api_key = (os.environ.get("CODINGROUTER_API_KEY") if args.provider in {"gemini", "codingrouter"}
               else os.environ.get("DEEPSEEK_API_KEY") if args.provider == "deepseek"
               else os.environ.get("INF_API_KEY") or os.environ.get("OPENETA_LLM_API_KEY"))
    if not api_key:
        parser.error("CODINGROUTER_API_KEY is required" if args.provider in {"gemini", "codingrouter"}
                     else "DEEPSEEK_API_KEY is required" if args.provider == "deepseek"
                     else "INF_API_KEY or OPENETA_LLM_API_KEY is required")
    host = urlparse(args.endpoint).hostname
    for key in ("NO_PROXY", "no_proxy"):
        os.environ[key] = ",".join(filter(None, [os.environ.get(key, ""), host, "localhost", "127.0.0.1"]))
    rows = [row for row in json.loads(args.manifest.read_text())["tasks"] if row["bench"] == args.bench]
    if args.env_id:
        rows = [row for row in rows if row["env_id"] == args.env_id]
    holdout = json.loads(args.holdout.read_text())
    _validate_manifest(rows, holdout, require_full=False)
    if args.task_limit:
        rows = rows[:args.task_limit]
    ledger_path = args.output_root / (args.bench + ".attempts.jsonl")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        stream.seek(0)
        ledger = [json.loads(line) for line in stream.read().splitlines()]
    transport = SseSimulatorMcpTransport(args.sim_url)
    launched = 0
    for wave in range(args.max_attempts_per_task):
        pending = False
        for task_index, row in enumerate(rows):
            previous = [item for item in ledger if item["env_id"] == row["env_id"]]
            task_root = args.output_root / args.bench / row["task_id"]
            successes = sum(json.loads(path.read_text()).get("success") is True for path in task_root.glob("attempt-*/episode.json"))
            if successes >= args.target_per_task or len(previous) >= args.max_attempts_per_task:
                continue
            pending = True
            existing = [int(path.name.split("-")[1]) for path in task_root.glob("attempt-*")]
            attempt = max(existing + [item["attempt"] for item in previous], default=-1) + 1
            seed = 20310000 + task_index * 10000 + attempt
            if seed in holdout["reserved_seeds"]:
                raise ValueError("Reserved test seed")
            directory = task_root / f"attempt-{attempt:04d}"
            record = collect_attempt(transport, row, directory=directory, seed=seed, api_key=api_key,
                                     endpoint=args.endpoint, model=args.model, max_turns=args.max_turns,
                                     max_control_steps=args.max_control_steps, image_limit=args.image_limit,
                                     timeout=args.timeout, enable_thinking=args.enable_thinking,
                                     history_turns=args.history_turns, max_tokens=args.max_tokens,
                                     model_timeout=args.model_timeout, model_max_attempts=args.model_max_attempts,
                                     model_retry_backoff=args.model_retry_backoff, provider=args.provider,
                                     retain_failures=args.retain_failures, unified_input=args.unified_input)
            record["images_retained"] = record["success"] or args.retain_failures
            record["run_config"]["continue_on_infrastructure_error"] = args.continue_on_infrastructure_error
            _write_json(directory / "episode.json", record)
            summary = {"env_id": row["env_id"], "task_id": row["task_id"], "attempt": attempt, "seed": seed,
                       "success": record["success"], "stop_reason": record.get("stop_reason"),
                       "turns": len(record["turns"]), "error": record.get("error"),
                       "model_protocol": PROTOCOL, "provider": args.provider, "model": args.model,
                       "images_retained": record["images_retained"], "episode_path": str(directory / "episode.json"),
                       "executed_tools": [turn["tool"] for turn in record["turns"] if turn.get("valid") is True and turn.get("tool")]}
            with ledger_path.open("a") as stream:
                fcntl.flock(stream, fcntl.LOCK_EX)
                stream.write(json.dumps(summary, ensure_ascii=False) + "\n")
            ledger.append(summary)
            if record["success"]:
                from scripts.export_two_tool_sft import export_dataset
                export_path = (args.export_root / args.bench / "train.v1.jsonl" if args.export_root
                               else REPO_ROOT / f"SFT_data/exported/{args.provider}_xml_{args.bench}/train.v1.jsonl")
                export_dataset(args.output_root, args.manifest, args.holdout,
                               export_path, args.bench)
            elif not args.retain_failures:
                audit = copy.deepcopy(record)
                audit["images_retained"] = False
                _write_json(args.output_root / "audit" / args.bench / row["task_id"] / f"attempt-{attempt:04d}.json", audit)
                shutil.rmtree(directory)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            launched += 1
            if record.get("stop_reason") == "infrastructure_error" and not args.continue_on_infrastructure_error:
                raise RuntimeError("Collector stopped for infrastructure repair: " + str(record.get("error")))
            if record.get("stop_reason") == "interrupted":
                raise KeyboardInterrupt
            if args.attempt_limit and launched >= args.attempt_limit:
                return
        if not pending:
            break


if __name__ == "__main__":
    main()
