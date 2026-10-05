"""Live inference checks through the repository planner backend; no simulator needed."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.backends.planner import (  # noqa: E402
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    PlannerBackendRequest,
)
from agent.runtime.actions import PipelineStatus  # noqa: E402
from agent.runtime.planner import _build_planner_decision, _parse_backend_payload  # noqa: E402

SYSTEM_PROMPT = (
    "You are running an inference contract test. Follow the task exactly. "
    "Return a single XML <decision> with kind, name, reasoning and parameters. "
    "For an answer use kind=response, name=talk and parameters/message. "
    "For a requested tool use kind=tool_call, the exact tool name and typed parameters. "
    "Do not execute tools, add markdown, or claim that a simulator was tested."
)


def cases(image_dir: Path, vision: bool) -> list[dict]:
    result = [
        {"id": "text", "task": "计算 17 + 25，仅在 message 中回答数字。", "expected": "42"},
        {"id": "multiturn", "task": "What is the code I asked you to remember? Reply only with the code.",
         "history": [{"role": "user", "content": "Remember the code BASELINE-731."},
                     {"role": "assistant", "content": "I will remember it."}],
         "expected": "BASELINE-731"},
        {"id": "tool_xml", "task": "Call inspect_object exactly once with object_id='cube_7' and include_depth=true.",
         "tools": [{"name": "inspect_object", "parameters": {
             "type": "object", "properties": {"object_id": {"type": "string"},
             "include_depth": {"type": "boolean"}}, "required": ["object_id", "include_depth"]}}]},
    ]
    if vision:
        from PIL import Image
        image_dir.mkdir(parents=True, exist_ok=True)
        # Paired images: the text prompt and filenames contain no answer clue.
        for index, (rgb, answer) in enumerate([((255, 0, 0), "red"), ((0, 0, 255), "blue")]):
            path = image_dir / f"probe_{index}.png"
            Image.new("RGB", (224, 224), rgb).save(path)
            result.append({"id": f"vision_{index}", "task": "What is the dominant color of the attached image? Reply with one lowercase English color word.",
                           "image": str(path), "expected": answer})
    return result


def validate_result(case: dict, result) -> tuple[bool, list[str]]:
    errors = []
    if result.status != PipelineStatus.PLANNED:
        errors.append("provider_failed")
    if not isinstance(result.payload, str) or not result.payload.strip():
        return False, errors + ["missing_text_content"]
    if result.details.get("finish_reason") != "stop":
        errors.append("incomplete_or_unexpected_finish_reason")
    payload, parse_errors = _parse_backend_payload(result.payload)
    errors.extend(parse_errors)
    if not parse_errors:
        _, build_errors = _build_planner_decision(payload)
        errors.extend(build_errors)
        parameters = payload.get("parameters", {})
        if case["id"] == "tool_xml":
            if (payload.get("kind") != "tool_call" or payload.get("name") != "inspect_object"
                    or parameters != {"object_id": "cube_7", "include_depth": True}
                    or parameters.get("include_depth") is not True):
                errors.append("wrong_tool_or_typed_parameters")
        else:
            message = str(parameters.get("message", "")).strip()
            answer_correct = message == case["expected"]
            if case["id"] == "multiturn":
                # Recall correctness is separate from "only the code" formatting.
                # Reject a wrong code, multiple answers, and substring collisions.
                answer_correct = re.findall(r"\bBASELINE-\d+\b", message) == [case["expected"]]
            if payload.get("kind") != "response" or payload.get("name") != "talk" or not answer_correct:
                errors.append("wrong_answer")
    if "image" in case and not any(x.get("attached") for x in result.details.get("vision_attachments", [])):
        errors.append("image_not_attached")
    return not errors, errors


def redact(value, secrets: list[str]):
    if isinstance(value, dict):
        return {k: redact(v, secrets) for k, v in value.items()
                if k.lower() not in {"authorization", "api_key", "api-key"}}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value


def run_model(name: str, entry: dict, args, test_cases: list[dict]) -> dict:
    key = os.environ.get(entry["api_key_env"], "")
    if not key and entry.get("local"):
        key = "local-no-auth"
    record = {"name": name, "model": entry["model"], "api_base": entry["api_base"], "cases": []}
    if not key:
        return {**record, "text_passed": False, "vision_passed": None,
                "error": f"Missing environment variable {entry['api_key_env']}"}
    config = OpenAICompatiblePlannerBackendConfig(
        model=entry["model"], api_base=entry["api_base"], api_key=key,
        max_tokens=args.max_tokens, temperature=0.0, enable_thinking=False,
        thinking_format=entry.get("thinking_format", "chat_template_kwargs"),
        use_json_response_format=False, timeout_s=args.timeout, max_attempts=1,
        enable_vision=True, fallback=None,
    )
    backend = OpenAICompatiblePlannerBackend(config)
    record["generation"] = {"max_tokens": args.max_tokens, "temperature": 0,
                            **config.thinking_request_fields(), "max_attempts": 1}
    for case in test_cases:
        started = time.monotonic()
        context = {"task": case["task"], "available_tools": case.get("tools", [])}
        if "image" in case:
            context["vision_image_paths"] = [case["image"]]
        try:
            response = backend.decide(PlannerBackendRequest(
                tool_context=context, system_prompt=SYSTEM_PROMPT,
                conversation_messages=case.get("history", []),
            ))
            passed, errors = validate_result(case, response)
            row = {"id": case["id"], "passed": passed, "errors": errors,
                   "content": response.payload, "details": response.details,
                   "exchange": response.rollout_exchange}
            if "expected" in case:
                parsed, _ = _parse_backend_payload(response.payload)
                row["exact_answer_match"] = (
                    str(parsed.get("parameters", {}).get("message", "")).strip() == case["expected"])
        except Exception as exc:
            row = {"id": case["id"], "passed": False, "errors": [f"{type(exc).__name__}: {exc}"]}
        row["elapsed_s"] = round(time.monotonic() - started, 3)
        record["cases"].append(redact(row, [key]))
        print(f"{name}/{case['id']}: {'PASS' if row['passed'] else 'FAIL'} ({row['elapsed_s']}s)", flush=True)
    text_rows = [r for r in record["cases"] if not r["id"].startswith("vision_")]
    vision_rows = [r for r in record["cases"] if r["id"].startswith("vision_")]
    record["text_passed"] = bool(text_rows) and all(r["passed"] for r in text_rows)
    record["vision_passed"] = all(r["passed"] for r in vision_rows) if vision_rows else None
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "baseline/configs/models.json")
    parser.add_argument("--models", nargs="+", help="Default: all four models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--vision", action="store_true", help="Probe paired images without requiring vision for text success")
    parser.add_argument("--require-vision", action="store_true", help="Fail the command if any paired image check fails")
    args = parser.parse_args()
    entries = json.loads(args.config.read_text())
    names = args.models or list(entries)
    if any(n not in entries for n in names) or len(names) != len(set(names)):
        parser.error("Unknown or duplicate model selection")
    if args.timeout <= 0 or args.max_tokens <= 0:
        parser.error("timeout and max-tokens must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("Output already exists; choose a new filename to preserve evidence")
    test_cases = cases(args.output.parent / "images", args.vision or args.require_vision)
    report = {"schema_version": "baseline.inference_smoke.v1", "started_at": datetime.now(timezone.utc).isoformat(),
              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "python": sys.version, "hostname": platform.node(),
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "models": []}
    for name in names:
        report["models"].append(run_model(name, entries[name], args, test_cases))
        report["passed"] = len(report["models"]) == len(names) and all(
            r["text_passed"] and (not args.require_vision or r["vision_passed"] is True)
            for r in report["models"])
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)
    print(f"Report: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
