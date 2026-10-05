"""Guard against false-positive inference smoke results and credential leaks."""
import importlib.util
from pathlib import Path

import pytest

from agent.backends.planner import PlannerBackendResult
from agent.runtime.actions import PipelineStatus

_PATH = Path(__file__).resolve().parents[1] / "baseline/scripts/inference_smoke.py"
_SPEC = importlib.util.spec_from_file_location("baseline_inference_smoke", _PATH)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


def response(text="42", **details):
    return PlannerBackendResult(
        payload=f"<decision><kind>response</kind><name>talk</name>"
                f"<reasoning>test</reasoning><parameters><message>{text}</message>"
                "</parameters></decision>",
        details={"finish_reason": "stop", **details},
    )


def test_valid_answer_and_truncation_are_distinguished():
    case = {"id": "text", "expected": "42"}
    assert smoke.validate_result(case, response())[0]
    assert not smoke.validate_result(case, response(finish_reason="length"))[0]
    assert not smoke.validate_result(case, response("43"))[0]


def test_failed_provider_and_empty_reasoning_only_do_not_pass():
    case = {"id": "text", "expected": "42"}
    result = response()
    result.status = PipelineStatus.FAILED
    assert not smoke.validate_result(case, result)[0]
    result.payload = ""
    assert "missing_text_content" in smoke.validate_result(case, result)[1]


def test_image_must_be_attached_and_answer_must_match():
    case = {"id": "vision_0", "image": "x.png", "expected": "red"}
    assert not smoke.validate_result(case, response("red"))[0]
    assert smoke.validate_result(case, response("red", vision_attachments=[{"attached": True}]))[0]
    assert not smoke.validate_result(case, response("blue", vision_attachments=[{"attached": True}]))[0]


def test_recall_accepts_explanation_but_rejects_wrong_or_ambiguous_codes():
    case = {"id": "multiturn", "expected": "BASELINE-731"}
    assert smoke.validate_result(case, response("The code is BASELINE-731."))[0]
    for answer in ("BASELINE-732", "BASELINE-7310", "BASELINE-731 or BASELINE-732", "unknown"):
        assert not smoke.validate_result(case, response(answer))[0]


def test_tool_parameter_types_are_checked():
    result = PlannerBackendResult(
        payload='<decision><kind>tool_call</kind><name>inspect_object</name>'
                '<reasoning>test</reasoning><parameters><object_id>cube_7</object_id>'
                '<include_depth>true</include_depth></parameters></decision>',
        details={"finish_reason": "stop"},
    )
    assert smoke.validate_result({"id": "tool_xml"}, result)[0]
    result.payload = result.payload.replace('<include_depth>', '<include_depth type="string">')
    assert not smoke.validate_result({"id": "tool_xml"}, result)[0]


def test_nested_credentials_and_echoed_secrets_are_redacted():
    raw = {"Authorization": "Bearer test-secret", "cases": [{"error": "echo test-secret"}],
           "api_key": "test-secret", "model": "Qwen"}
    assert smoke.redact(raw, ["test-secret"]) == {
        "cases": [{"error": "echo [REDACTED]"}], "model": "Qwen"}


@pytest.mark.parametrize("thinking_format,enabled,expected", [
    ("chat_template_kwargs", False, {"chat_template_kwargs": {"enable_thinking": False}}),
    ("thinking", False, {"thinking": {"type": "disabled"}}),
    ("thinking", True, {"thinking": {"type": "enabled"}}),
    ("thinking", None, {}),
])
def test_live_request_contract_uses_production_backend_without_fallback(thinking_format, enabled, expected):
    captured = {}

    def transport(url, body, headers, timeout):
        captured.update(url=url, body=body, headers=headers)
        return {"choices": [{"message": {"content": response().payload}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}}

    config = smoke.OpenAICompatiblePlannerBackendConfig(
        model="test", api_base="http://localhost:8101/v1", api_key="private-test-key",
        max_tokens=1024, enable_thinking=enabled, thinking_format=thinking_format,
        use_json_response_format=False, max_attempts=1,
    )
    backend = smoke.OpenAICompatiblePlannerBackend(config, transport=transport)
    result = backend.decide(smoke.PlannerBackendRequest(
        tool_context={"task": "17+25"}, system_prompt=smoke.SYSTEM_PROMPT))
    assert smoke.validate_result({"id": "text", "expected": "42"}, result)[0]
    assert captured["url"] == "http://localhost:8101/v1/chat/completions"
    assert {k: captured["body"][k] for k in ("thinking", "chat_template_kwargs") if k in captured["body"]} == expected
    assert captured["body"]["max_tokens"] == 1024
    assert "response_format" not in captured["body"]
    assert config.fallback is None
    assert "private-test-key" not in str(smoke.redact(result.rollout_exchange, ["private-test-key"]))


def test_thinking_format_validation_and_provider_metadata():
    from agent.backends.provider_config import PlannerProviderConfig

    with pytest.raises(ValueError, match="thinking_format"):
        smoke.OpenAICompatiblePlannerBackendConfig(thinking_format="typo")
    config = smoke.OpenAICompatiblePlannerBackendConfig.from_provider_config(
        PlannerProviderConfig(metadata={"enable_thinking": False, "thinking_format": "thinking"}))
    assert config.thinking_request_fields() == {"thinking": {"type": "disabled"}}
    assert config.redacted()["thinking_format"] == "thinking"
