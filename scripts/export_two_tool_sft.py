#!/usr/bin/env python3
"""Export authentic XML requests and replies from successful live teacher episodes."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_two_tool_sft import ACTION_DIMS, PROVIDERS, episode_provider, _official_success, _validate_manifest
from agent.runtime.two_tool_xml import PROTOCOL, parse_decision, split_answer
from agent.runtime.two_tool_input import INPUT_PROTOCOL, digest


def swift_messages(source: list[dict]) -> tuple[list[dict], list[str]]:
    messages, images = [], []
    for index, original in enumerate(source):
        message = {"role": original["role"]}
        content = original.get("content")
        if original["role"] == "assistant":
            _, content = split_answer(content)
            message["loss"] = index == len(source) - 1
        elif isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(item["text"])
                elif item.get("type") == "image_url":
                    image = item["image_url"]["url"]
                    if not Path(image).is_file():
                        raise ValueError("Model input image is missing")
                    images.append(image)
                    parts.append("<image>")
                else:
                    raise ValueError("Unsupported multimodal input")
            content = "\n".join(parts)
        if message["role"] not in {"system", "user", "assistant"} or not isinstance(content, str):
            raise ValueError("Unexpected XML conversation message")
        message["content"] = content
        messages.append(message)
    return messages, images


def samples_from_episode(path: Path, allowed: dict, holdout: dict) -> list[dict]:
    record = json.loads(path.read_text())
    provider = episode_provider(record)
    if provider is None:
        return []
    if record.get("model_protocol") != PROTOCOL or record.get("output_source") != "original_model_xml":
        return []
    if record.get("success") is not True or record.get("stop_reason") != "official_success":
        return []
    row = allowed.get(record.get("env_id"))
    if not row or record.get("bench") != row["bench"] or record.get("seed") in holdout["reserved_seeds"]:
        return []
    if record.get("qwen_retrospective") or any(turn.get("qwen_think") for turn in record.get("turns", [])):
        return []
    turns = record.get("turns", [])
    if not any(turn.get("tool") == "step_env" and turn.get("official_success") is True
               and _official_success(turn.get("receipt"), bench=record["bench"]) for turn in turns):
        return []
    result = []
    for turn in turns:
        if turn.get("valid") is not True:
            continue
        evidence = record["model_calls"][turn["model_call_index"]]
        if record.get("input_protocol") == INPUT_PROTOCOL:
            package = evidence["input_package"]
            if package["messages"] != evidence["request"]["messages"] or digest(package["messages"]) != package["messages_sha256"]:
                raise ValueError("Saved conditioning input differs from actual model request")
            import hashlib
            for image in package["images"]:
                if hashlib.sha256(Path(image["path"]).read_bytes()).hexdigest() != image["file_sha256"]:
                    raise ValueError("Model input image changed after collection")
        assistant = evidence["response"]["choices"][0]["message"]
        if assistant != turn["model_message"]:
            raise ValueError("Model output does not match executed turn")
        if assistant.get("tool_calls") or evidence["request"].get("tools") or evidence["response"]["choices"][0].get("finish_reason") != "stop":
            raise ValueError("Not a complete native XML reply")
        parsed = parse_decision(assistant["content"], contract=record["action_contract"],
                                remaining_steps=turn["control_steps_remaining"])
        decision = parsed["decision"]
        arguments = decision["parameters"]
        if decision != turn["decision"] or decision["name"] != turn["tool"] or arguments != turn["arguments"]:
            raise ValueError("Executed tool differs from model output")
        actual = turn["tool_request"]
        host_keys = {"handle", "session_id", "strict"}
        if actual["name"] != turn["tool"] or {key: value for key, value in actual["arguments"].items() if key not in host_keys} != arguments:
            raise ValueError("Actual MCP request differs from model output")
        if turn["tool"] == "step_env" and actual["arguments"].get("strict") is not True:
            raise ValueError("Execution did not enforce strict action validation")
        messages, images = swift_messages(copy.deepcopy(evidence["request"]["messages"]) + [copy.deepcopy(assistant)])
        result.append({
            "schema_version": PROVIDERS[provider]["sample_schema"],
            "sample_id": f"{record['bench']}/{path.parents[1].name}/{path.parent.name}:turn-{turn['turn']:04d}",
            "episode_id": str(path.parent.resolve()), "split": "train",
            "messages": messages, "images": images,
            "metadata": {"bench": record["bench"], "env_id": record["env_id"], "seed": record["seed"],
                         "teacher_model": record["model"], "action_source": record["action_source"], "provider": provider,
                         "input_protocol": record.get("input_protocol", "legacy"),
                         "input_profile_hash": record.get("input_profile", {}).get("hash"),
                         "reasoning_source": "original_model_response", "posthoc_annotation": False,
                         "model_protocol": PROTOCOL, "output_source": "original_model_xml",
                         "supervision": "current_decision_only", "swift_loss_scale": "last_round+ignore_empty_think",
                         "thinking_supervision": "answer_only", "source_episode": str(path.resolve()),
                         "source_turn": turn["turn"], "template_profile": "two_tool_xml_sft.v1",
                         "target_message_index": len(messages) - 1, "official_episode_success": True},
        })
    return result


def export_dataset(root: Path, manifest_path: Path, holdout_path: Path, output: Path, bench: str) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _export_dataset_locked(root, manifest_path, holdout_path, output, bench)


def _export_dataset_locked(root: Path, manifest_path: Path, holdout_path: Path, output: Path, bench: str) -> dict:
    rows = [row for row in json.loads(manifest_path.read_text())["tasks"] if row["bench"] == bench]
    holdout = json.loads(holdout_path.read_text())
    _validate_manifest(rows, holdout, require_full=False)
    allowed = {row["env_id"]: row for row in rows}
    samples = []
    episodes = 0
    for path in sorted((root / bench).glob("*/attempt-*/episode.json")):
        group = samples_from_episode(path, allowed, holdout)
        if group:
            samples.extend(group)
            episodes += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w") as stream:
        for sample in samples:
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return {"bench": bench, "episodes": episodes, "samples": len(samples), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, default=REPO_ROOT / "SFT_data/configs/two_tool_test_holdout.v1.json")
    parser.add_argument("--input-root", type=Path, default=REPO_ROOT / "SFT_data/qwen_xml_v1")
    parser.add_argument("--bench", choices=tuple(ACTION_DIMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_dataset(args.input_root, args.manifest, args.holdout, args.output, args.bench)))


if __name__ == "__main__":
    main()
