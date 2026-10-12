#!/usr/bin/env python3
"""Add Gemini post-hoc reasoning to verified artificial SFT without changing actions."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.runtime.two_tool_xml import parse_decision
from scripts.collect_two_tool_sft import PROVIDERS, _call_model, _write_json

MODEL = "gemini-3.6-flash"
SCHEMA = "openeta.sft.artificial_action_gemini_reasoning.v1"
PROMPT_VERSION = "step_purpose.v2"
SYSTEM = (
    "You annotate an already-executed robot demonstration for supervised training. "
    "Explain the immediate purpose of this FIXED next step in progressing toward the task, "
    "rather than merely describing the action or its numeric direction. "
    "State the local subgoal this action is intended to achieve, grounded in the current observation. "
    "Mention motion or gripper behavior only when necessary to explain that purpose. "
    "Do not infer a successful grasp or object possession from a close-gripper command alone. "
    "Write one concise English sentence. "
    "This is a post-hoc explanation, not your own action choice. Do not change or propose "
    "an action, cite hidden simulator state, or refer to any future observation or success. "
    "Do not claim a movement has succeeded before it happens. No XML, JSON or markdown. "
    "If evidence is ambiguous, say what the action is intended to do rather than inventing facts. "
    "Do not guess an object's color, shape or identity from ambiguous pixels; use the task's object name. "
    "Use world-axis language only when the action mapping makes it unambiguous. "
    "Do not mention official implementations, expert policies, demonstrations, replay, "
    "annotation, source code or dataset construction in the explanation."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gripper_state(bench: str, value: float) -> str:
    if abs(value) < 0.2:
        return "neutral"
    return ("open" if value < 0 else "close") if bench != "maniskill" else (
        "open" if value > 0 else "close")


def _gemini_content(sample: dict, action_contract: dict) -> list[dict]:
    visible = json.loads(sample["messages"][-2]["content"].split("\nImage evidence:", 1)[0])
    compact = {key: visible.get(key) for key in
               ("turn", "observation_id", "execution", "last_action", "terminated",
                "truncated", "robot", "robot_delta")}
    bench = sample["metadata"]["bench"]
    slots = {
        "metaworld": "[dx, dy, dz, gripper]; -1=open, +1=close",
        "libero": "[dx, dy, dz, droll, dpitch, dyaw, gripper]; -1=open, +1=close",
        "maniskill": "[dx, dy, dz, droll, dpitch, dyaw, gripper]; +1=open, -1=close",
        "robocasa": "[arm_dx, arm_dy, arm_dz, droll, dpitch, dyaw, gripper, base_forward, base_left, base_yaw, torso, mode]; -1=open, +1=close",
    }[bench]
    decision = parse_decision(sample["messages"][-1]["content"], contract=action_contract)["decision"]
    gripper = decision["parameters"]["action"][3 if bench == "metaworld" else 6]
    gripper_state = _gripper_state(bench, gripper)
    task = sample["messages"][0]["content"].split("Task: ", 1)[-1].split("\n", 1)[0]
    text = (f"Task: {task}\nAction slots: {slots}\n"
            f"Fixed gripper command: {gripper_state} (numeric value {gripper}). "
            "Do not reverse this polarity.\n"
            f"Current student-visible state: {json.dumps(compact, ensure_ascii=False)}\n"
            f"Fixed expert decision: {sample['messages'][-1]['content']}\n"
            "What immediate task-relevant subgoal is this fixed step intended to achieve? "
            "Ground the purpose in the current image and visible state, not merely the action numbers. "
            "It has not been executed yet; do not claim its outcome or assume an object is held.")
    if not sample["images"]:
        raise ValueError("Reasoning preview requires the current RGB observation")
    frame_prefix = f"{int(visible['turn']) - 1:04d}-"
    current_images = [path for path in sample["images"]
                      if Path(path).name.startswith(frame_prefix)]
    if not current_images:
        raise ValueError("No current image matched the visible decision turn")
    return [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": path}} for path in current_images]


def _reasoning_from_response(response: dict, *, gripper_state: str) -> tuple[str, dict]:
    choice = response["choices"][0]
    if choice.get("finish_reason") != "stop":
        raise ValueError(f"Gemini did not finish: {choice.get('finish_reason')}")
    value = choice["message"].get("content", "").strip()
    value = re.sub(r"\s+", " ", value).strip().strip('"')
    if not value or len(value) > 350 or any(mark in value for mark in ("<decision", "```", "<reasoning")):
        raise ValueError("Gemini reasoning is empty, overlong, or contains markup")
    if "<" in value or ">" in value:
        raise ValueError("Gemini reasoning contains angle brackets")
    if re.search(r'\b(?:official|implementation|expert policy|replay|demonstration|annotation|source code|dataset)\b', value, re.I):
        raise ValueError("Gemini reasoning refers to data construction instead of the current action")
    def gripper_verb(verb: str) -> bool:
        forms = (r"open(?:s|ed|ing)?" if verb == "open" else
                 r"clos(?:e|es|ed|ing)")
        target = r"(?:gripper|fingers|claw)"
        return bool(re.search(rf"\b{forms}\s+(?:the\s+)?(?:robot(?:'s)?\s+)?{target}\b", value, re.I)
                    or re.search(rf"\b{target}\s+(?:is|are|remains|stay|stays|keeps|kept)?\s*{forms}\b", value, re.I))

    if gripper_state == "open" and gripper_verb("close"):
        raise ValueError("Gemini reasoning claims closing while fixed gripper command is open")
    if gripper_state == "close" and gripper_verb("open"):
        raise ValueError("Gemini reasoning claims opening while fixed gripper command is close")
    return value, response.get("usage", {})


def annotate_episode(path: Path, output_root: Path, *, api_key: str, timeout: float,
                     verbose: bool = True, workers: int = 1, model: str = MODEL) -> dict:
    episode_path = path.parent / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    if episode.get("success") is not True or episode.get("stop_reason") != "official_success":
        raise ValueError(f"Not an officially verified successful episode: {episode_path}")
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not samples or len(samples) != episode.get("samples"):
        raise ValueError("Episode/sample count mismatch")
    bench, task_id, seed = episode["bench"], episode["task_id"], episode["seed"]
    directory = output_root / "episodes" / bench / task_id / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(path)
    if workers < 1:
        raise ValueError("workers must be positive")

    def load_annotation(item: tuple[int, dict]) -> dict:
        turn, sample = item
        original = sample["messages"][-1]["content"]
        if original.count("<reasoning />") != 1:
            raise ValueError(f"Expected exactly one empty reasoning at turn {turn}")
        cache_path = directory / "annotations" / f"turn-{turn:04d}.json"
        if cache_path.is_file():
            annotation = json.loads(cache_path.read_text(encoding="utf-8"))
            if annotation.get("source_sample_sha256") != hashlib.sha256(
                    json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()).hexdigest():
                raise ValueError(f"Cached annotation does not match source turn {turn}")
        else:
            decision = parse_decision(original, contract=episode["action_contract"])["decision"]
            action = decision["parameters"]["action"]
            gripper = action[3 if bench == "metaworld" else 6]
            gripper_state = _gripper_state(bench, gripper)
            wire = {"model": model, "max_tokens": 8192, "provider": "gemini",
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": _gemini_content(sample, episode["action_contract"])}]}
            error = None
            for attempt in range(3):
                try:
                    response = _call_model(wire, api_key=api_key,
                        endpoint=PROVIDERS["gemini"]["endpoint"], timeout=timeout)
                    reasoning, usage = _reasoning_from_response(response, gripper_state=gripper_state)
                    error = None
                    break
                except Exception as exc:
                    error = exc
                    if attempt < 2:
                        time.sleep(2 ** attempt)
            if error is not None:
                raise RuntimeError(f"Gemini failed on turn {turn}: {type(error).__name__}: {error}".replace(api_key, "[REDACTED]")) from error
            annotation = {"schema_version": "openeta.reasoning_annotation.v1",
                          "source_sample_sha256": hashlib.sha256(
                              json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                          "sample_id": sample["sample_id"], "reasoning": reasoning,
                          "reasoning_model": model, "usage": usage,
                          "thinking_mode": "provider_default", "prompt_version": PROMPT_VERSION}
            _write_json(cache_path, annotation)
        return annotation

    if workers == 1:
        annotations = list(map(load_annotation, enumerate(samples, 1)))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            annotations = list(pool.map(load_annotation, enumerate(samples, 1)))
    decorated = []
    for turn, (sample, annotation) in enumerate(zip(samples, annotations), 1):
        original = sample["messages"][-1]["content"]
        new_sample = copy.deepcopy(sample)
        new_xml = original.replace("<reasoning />", f"<reasoning>{html.escape(annotation['reasoning'])}</reasoning>")
        before = parse_decision(original, contract=episode["action_contract"])["decision"]
        after = parse_decision(new_xml, contract=episode["action_contract"])["decision"]
        if {key: value for key, value in before.items() if key != "reasoning"} != {
                key: value for key, value in after.items() if key != "reasoning"}:
            raise ValueError(f"Annotated turn {turn} changed the fixed decision")
        new_sample["schema_version"] = SCHEMA
        new_sample["messages"][-1]["content"] = new_xml
        new_sample["metadata"].update({"reasoning_source": "gemini_posthoc_annotation",
                                       "reasoning_model": annotation['reasoning_model'],
                                       "reasoning_prompt_version": PROMPT_VERSION,
                                       "posthoc_annotation": True,
                                       "source_sample_sha256": annotation["source_sample_sha256"]})
        decorated.append(new_sample)
        if verbose:
            print(json.dumps({"sample_id": sample["sample_id"], "reasoning": annotation["reasoning"]},
                             ensure_ascii=False), flush=True)
    output = directory / "samples.jsonl"
    temporary = output.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in decorated), encoding="utf-8")
    temporary.replace(output)
    report = {"schema_version": "openeta.reasoning_preview_episode.v1",
              "source_episode": str(episode_path.resolve()), "source_samples_sha256": source_hash,
              "success": True, "model": model,
              "annotation_models": sorted({annotation['reasoning_model'] for annotation in annotations}),
              "action_source": "artificial", "prompt_version": PROMPT_VERSION,
              "reasoning_source": "gemini_posthoc_annotation", "turns": len(decorated),
              "output": str(output.resolve())}
    _write_json(directory / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path, help="Successful source episode.json paths")
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "SFT_data/expert_replay_v1/reasoning_preview_gemini_flash")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    key = os.environ.get("CODINGROUTER_API_KEY", "")
    if not key:
        parser.error("CODINGROUTER_API_KEY is required")
    for episode_path in args.episodes:
        path = episode_path.parent / "samples.jsonl"
        print(json.dumps(annotate_episode(path, args.output_root, api_key=key,
                                          timeout=args.timeout), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
