#!/usr/bin/env python3
"""Export successful OpenETA ManiSkill rollout bundles to turn-level SFT JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping
from xml.sax.saxutils import escape


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.runtime.rollout import validate_rollout_bundle


SFT_SCHEMA_VERSION = "openeta.sft.tool_call.v1"
MAIN_PLANNER_MARKER = "OpenETA closed-loop embodied planner"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain an object")
        rows.append(value)
    return rows


def discover_bundles(root: Path) -> list[Path]:
    return sorted(path.parent for path in root.rglob("rollout/manifest.json"))


def _explicit_success(transition: Mapping[str, Any]) -> bool:
    info = transition.get("info")
    if not isinstance(info, Mapping):
        return False
    if any(
        info.get(key) is True
        for key in ("environment_success", "task_success", "checker_success")
    ):
        return True
    receipt = info.get("environment_receipt")
    return isinstance(receipt, Mapping) and receipt.get("task_success") is True


def _episode_result(bundle: Path) -> dict[str, Any] | None:
    for row in reversed(_jsonl(bundle / "episodes.jsonl")):
        if row.get("event") == "result" and isinstance(row.get("result"), dict):
            return dict(row["result"])
    return None


def _episode_assisted(result: Mapping[str, Any]) -> bool:
    metadata = result.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    assistance = metadata.get("assistance")
    assistance = assistance if isinstance(assistance, Mapping) else {}
    return bool(
        assistance.get("assisted")
        or assistance.get("human_intervention_count")
        or assistance.get("guidance_intervention_count")
        or metadata.get("human_intervention_count")
        or metadata.get("guidance_intervention_count")
    )


def audit_bundle(
    bundle: Path,
    *,
    teacher_model: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    validation = validate_rollout_bundle(bundle)
    if validation.get("valid") is not True:
        return None, ["bundle_invalid"]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    if str(metadata.get("teacher_model") or "") != teacher_model:
        reasons.append("teacher_model_mismatch")
    env_id = str(metadata.get("env_id") or "")
    if "maniskill" not in env_id.lower():
        reasons.append("not_maniskill")
    result = _episode_result(bundle)
    if result is None:
        reasons.append("episode_result_missing")
    elif _episode_assisted(result):
        reasons.append("episode_assisted")
    transitions = _jsonl(bundle / "transitions.jsonl")
    if not any(_explicit_success(row) for row in transitions):
        reasons.append("explicit_task_success_missing")
    accepted_calls = [
        row
        for row in _jsonl(bundle / "model_calls.jsonl")
        if _is_accepted_main_planner_call(row, teacher_model=teacher_model)
    ]
    if not accepted_calls:
        reasons.append("accepted_main_planner_calls_missing")
    summary = {
        "bundle": str(bundle.resolve()),
        "session_id": manifest.get("session_id"),
        "episode_id": metadata.get("episode_id"),
        "env_id": env_id,
        "seed": metadata.get("seed"),
        "task_slug": metadata.get("task_slug"),
        "teacher_model": metadata.get("teacher_model"),
        "accepted_call_count": len(accepted_calls),
        "transition_count": len(transitions),
        "artifact_count": validation.get("artifact_count", 0),
        "artifact_bytes": validation.get("artifact_bytes", 0),
    }
    return (summary if not reasons else None), reasons


def _is_accepted_main_planner_call(
    row: Mapping[str, Any],
    *,
    teacher_model: str,
) -> bool:
    validation = row.get("validation")
    result = row.get("result")
    semantic = row.get("semantic_request")
    decision = row.get("parsed_decision")
    if not isinstance(validation, Mapping) or validation.get("accepted") is not True:
        return False
    if not isinstance(result, Mapping) or str(result.get("model") or "") != teacher_model:
        return False
    if not isinstance(semantic, Mapping) or MAIN_PLANNER_MARKER not in str(
        semantic.get("system_prompt") or ""
    ):
        return False
    metadata = semantic.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("isolated_context") is True:
        return False
    return bool(
        isinstance(decision, Mapping)
        and decision.get("kind") in {"tool_call", "response"}
        and decision.get("name")
    )


def _last_successful_exchange(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    exchange = row.get("provider_exchange")
    attempts = exchange.get("attempts") if isinstance(exchange, Mapping) else None
    if not isinstance(attempts, list):
        return None
    for attempt in reversed(attempts):
        if isinstance(attempt, Mapping) and isinstance(attempt.get("response"), Mapping):
            return attempt
    return None


def _cdata(value: str) -> str:
    suffix = "]]" + ">"
    return "<![CDATA[" + value.replace("]]>", "]]]]><![CDATA[>") + suffix


def _xml_value(name: str, value: Any) -> str:
    tag = escape(name)
    if value is None:
        return f'<{tag} type="null"/>'
    if isinstance(value, bool):
        return f'<{tag} type="boolean">{str(value).lower()}</{tag}>'
    if isinstance(value, int) and not isinstance(value, bool):
        return f'<{tag} type="integer">{value}</{tag}>'
    if isinstance(value, float):
        return f'<{tag} type="number">{value}</{tag}>'
    if isinstance(value, Mapping):
        body = "".join(_xml_value(str(key), child) for key, child in value.items())
        return f"<{tag}>{body}</{tag}>"
    if isinstance(value, list):
        body = "".join(_xml_value("item", child) for child in value)
        return f'<{tag} type="array">{body}</{tag}>'
    text = str(value)
    encoded = _cdata(text) if "\n" in text or any(char in text for char in "<>&") else escape(text)
    return f"<{tag}>{encoded}</{tag}>"


def canonical_decision_xml(decision: Mapping[str, Any]) -> str:
    return (
        "<decision>"
        + _xml_value("kind", decision.get("kind"))
        + _xml_value("name", decision.get("name"))
        + _xml_value("reasoning", decision.get("reasoning") or "")
        + _xml_value("parameters", decision.get("parameters") or {})
        + "</decision>"
    )


def _materialize_message_artifacts(
    value: Any,
    *,
    bundle: Path,
    images: list[str],
) -> Any:
    if isinstance(value, list):
        return [
            _materialize_message_artifacts(item, bundle=bundle, images=images)
            for item in value
        ]
    if not isinstance(value, dict):
        return value
    if isinstance(value.get("bundle_path"), str):
        path = (bundle / value["bundle_path"]).resolve()
        if path.is_file() and str(value.get("mime_type") or "").startswith("image/"):
            path_text = str(path)
            if path_text not in images:
                images.append(path_text)
            return path_text
    return {
        str(key): _materialize_message_artifacts(child, bundle=bundle, images=images)
        for key, child in value.items()
    }


def samples_from_bundle(
    bundle: Path,
    *,
    teacher_id: str,
    teacher_model: str,
) -> Iterable[dict[str, Any]]:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    episode_id = str(metadata.get("episode_id") or manifest.get("session_id") or "episode")
    for row in _jsonl(bundle / "model_calls.jsonl"):
        if not _is_accepted_main_planner_call(row, teacher_model=teacher_model):
            continue
        exchange = _last_successful_exchange(row)
        request_body = exchange.get("request_body") if isinstance(exchange, Mapping) else None
        messages = request_body.get("messages") if isinstance(request_body, Mapping) else None
        if not isinstance(messages, list):
            continue
        images: list[str] = []
        prepared_messages = _materialize_message_artifacts(
            messages,
            bundle=bundle,
            images=images,
        )
        decision = row["parsed_decision"]
        prepared_messages.append(
            {"role": "assistant", "content": canonical_decision_xml(decision)}
        )
        request_kwargs = (
            request_body.get("chat_template_kwargs")
            if isinstance(request_body, Mapping)
            else None
        )
        yield {
            "schema_version": SFT_SCHEMA_VERSION,
            "sample_id": f"{episode_id}:model-call-{int(row.get('seq') or 0):04d}",
            "episode_id": episode_id,
            "split": "train",
            "messages": prepared_messages,
            "images": images,
            "metadata": {
                "env_id": metadata.get("env_id"),
                "seed": metadata.get("seed"),
                "task_slug": metadata.get("task_slug"),
                "teacher_id": teacher_id,
                "teacher_model": teacher_model,
                "thinking_enabled": (
                    request_kwargs.get("enable_thinking")
                    if isinstance(request_kwargs, Mapping)
                    else None
                ),
                "tool_name": decision.get("name"),
                "episode_success": True,
                "source_bundle": str(bundle.resolve()),
                "source_model_call_seq": row.get("seq"),
            },
        }


def export_dataset(
    *,
    input_root: Path,
    output: Path,
    report: Path,
    accepted_index: Path,
    teacher_id: str,
    teacher_model: str,
) -> dict[str, Any]:
    bundles = discover_bundles(input_root)
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    tool_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    for bundle in bundles:
        summary, reasons = audit_bundle(bundle, teacher_model=teacher_model)
        if reasons:
            rejected.append({"bundle": str(bundle.resolve()), "reasons": reasons})
            continue
        assert summary is not None
        bundle_samples = list(
            samples_from_bundle(
                bundle,
                teacher_id=teacher_id,
                teacher_model=teacher_model,
            )
        )
        if not bundle_samples:
            rejected.append(
                {"bundle": str(bundle.resolve()), "reasons": ["no_exportable_samples"]}
            )
            continue
        accepted.append(summary)
        samples.extend(bundle_samples)
        task_counts[str(summary.get("task_slug") or "unknown")] += 1
        tool_counts.update(
            str(sample["metadata"].get("tool_name") or "unknown")
            for sample in bundle_samples
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples),
        encoding="utf-8",
    )
    accepted_index.parent.mkdir(parents=True, exist_ok=True)
    accepted_index.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted),
        encoding="utf-8",
    )
    payload = {
        "schema_version": "openeta.sft_export_report.v1",
        "teacher_id": teacher_id,
        "teacher_model": teacher_model,
        "input_root": str(input_root.resolve()),
        "output": str(output.resolve()),
        "bundle_count": len(bundles),
        "accepted_episode_count": len(accepted),
        "rejected_episode_count": len(rejected),
        "sample_count": len(samples),
        "task_counts": dict(sorted(task_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "rejected": rejected,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--accepted-index", type=Path, required=True)
    parser.add_argument("--teacher-id", choices=("glm", "qwen"), required=True)
    parser.add_argument("--teacher-model", required=True)
    args = parser.parse_args()
    payload = export_dataset(
        input_root=args.input_root,
        output=args.output,
        report=args.report,
        accepted_index=args.accepted_index,
        teacher_id=args.teacher_id,
        teacher_model=args.teacher_model,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
