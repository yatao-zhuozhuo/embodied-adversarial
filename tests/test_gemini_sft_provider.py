import base64
import copy
import io
import json
from urllib.error import HTTPError

import pytest

from scripts import collect_two_tool_sft as collector
from scripts import gemini_sft_provider as gemini
from scripts.export_two_tool_sft import samples_from_episode
from tests.test_two_tool_sft import HOLDOUT, ROW, run_episode


def test_native_mapping_preserves_order_and_default_thinking(tmp_path):
    picture = tmp_path / "frame.png"
    picture.write_bytes(b"png-fixture")
    messages = [{"role": "system", "content": "contract"},
                {"role": "user", "content": "task"},
                {"role": "assistant", "content": "<decision/>"},
                {"role": "user", "content": [{"type": "text", "text": "current wrist"},
                 {"type": "image_url", "image_url": {"url": str(picture)}}]}]
    payload = collector._model_request("gemini", "gemini-3.6-flash", messages, 8192, True)
    original = copy.deepcopy(payload)
    wire = collector._provider_wire(payload)
    assert payload == original
    assert wire["systemInstruction"] == {"parts": [{"text": "contract"}]}
    assert [item["role"] for item in wire["contents"]] == ["user", "model", "user"]
    assert wire["contents"][-1]["parts"] == [{"text": "current wrist"}, {"inlineData": {
        "mimeType": "image/png", "data": base64.b64encode(b"png-fixture").decode()}}]
    assert wire["generationConfig"] == {"maxOutputTokens": 8192}
    assert "tools" not in wire
    with pytest.raises(ValueError):
        collector._model_request("gemini", "model", messages, 8192, False)


@pytest.mark.parametrize("reason,expected", [("STOP", "stop"), ("MAX_TOKENS", "length"), ("SAFETY", "safety")])
def test_raw_response_and_thinking_preserved(reason, expected):
    raw = {"candidates": [{"finishReason": reason, "content": {"parts": [
        {"text": "hidden summary", "thought": True},
        {"text": "<decision>", "thoughtSignature": "original-signature"},
        {"text": "</decision>"}]}}], "modelVersion": "returned-model",
        "usageMetadata": {"thoughtsTokenCount": 42, "candidatesTokenCount": 10}}
    response = gemini.normalize_response(raw)
    assert response["provider_response"] == raw
    assert response["provider_response"] is not raw
    assert response["choices"][0]["message"]["content"] == "<decision></decision>"
    assert response["choices"][0]["message"]["reasoning_content"] == "hidden summary"
    assert response["choices"][0]["finish_reason"] == expected
    assert response["usage"]["reasoning_tokens"] == 42


def test_blocked_and_native_tools_cannot_be_executed():
    blocked = gemini.normalize_response({"promptFeedback": {"blockReason": "SAFETY"}})
    assert blocked["choices"][0]["finish_reason"] != "stop"
    assert blocked["usage"]["reasoning_tokens"] is None
    native = gemini.normalize_response({"candidates": [{"finishReason": "STOP", "content": {
        "parts": [{"functionCall": {"name": "step_env", "args": {}}}]}}]})
    assert native["choices"][0]["message"]["tool_calls"]


def test_http_auth_is_not_in_url_or_body(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"candidates": []}'

    def urlopen(req, timeout):
        captured.update(url=req.full_url, headers=dict(req.header_items()), body=json.loads(req.data))
        return Response()

    monkeypatch.setattr(collector.request, "urlopen", urlopen)
    payload = collector._model_request("gemini", "gemini-3.6-flash", [{"role": "user", "content": "hello"}], 8192, True)
    collector._call_model(payload, api_key="test-secret", endpoint=collector.PROVIDERS["gemini"]["endpoint"], timeout=3)
    assert captured["url"].endswith("/gemini-3.6-flash:generateContent")
    assert captured["headers"]["X-goog-api-key"] == "test-secret"
    assert "test-secret" not in captured["url"] + json.dumps(captured["body"])
    with pytest.raises(ValueError):
        gemini.endpoint_for_model("https://example.invalid?key=secret", "model")


@pytest.mark.parametrize("provider", ["gemini", "codingrouter"])
def test_gemini_success_can_be_exported(tmp_path, provider):
    record, transport, requests, path = run_episode(tmp_path, provider=provider)
    assert record["success"] is True
    assert record["run_config"]["thinking_mode"] == "provider_default"
    assert record["run_config"]["enable_thinking"] is None
    assert collector.episode_provider(record) == provider
    assert len(samples_from_episode(path, {ROW["env_id"]: ROW}, HOLDOUT)) == 2


def test_codingrouter_keeps_default_thinking():
    payload = collector._model_request("codingrouter", "gpt-5.6-luna", [{"role": "user", "content": "task"}], 8192, True)
    assert collector._provider_wire(payload) == payload
    assert set(payload) == {"model", "messages", "max_tokens"}
    with pytest.raises(ValueError):
        collector._model_request("codingrouter", "gpt-5.6-luna", [], 8192, False)


def test_gateway_error_details_are_recorded_without_key():
    evidence = {}

    def model(*args, **kwargs):
        raise HTTPError("https://example.invalid", 503, "Unavailable", {},
                        io.BytesIO(b'{"code":"model_not_found","echo":"test-secret"}'))

    with pytest.raises(HTTPError):
        collector._call_model_with_retries(model, {}, api_key="test-secret", endpoint="https://example.invalid",
                                          timeout=1, max_attempts=1, backoff=0, evidence=evidence, persist=lambda: None)
    assert "model_not_found" in evidence["attempts"][0]["http_error_body"]
    assert "test-secret" not in json.dumps(evidence)
    assert evidence["retry_exhausted"] is True
