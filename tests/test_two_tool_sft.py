import copy
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

from scripts import collect_two_tool_sft as collector
from scripts.export_two_tool_sft import samples_from_episode
from agent.runtime.two_tool_xml import parse_decision, serialize_decision

HOLDOUT_PATH = Path(__file__).resolve().parents[1] / "SFT_data/configs/two_tool_test_holdout.v1.json"
HOLDOUT = json.loads(HOLDOUT_PATH.read_text())
ROW = {"bench": "metaworld", "task_id": "reach-v3", "env_id": "openeta/metaworld_50_reach-v3-v0", "task": "Reach a goal position"}


class Transport:
    def __init__(self, success=True):
        self.calls = []
        self.success = success

    def call_tool(self, name, arguments, timeout_s):
        self.calls.append((name, arguments))
        observation = {"robot": {}, "cameras": [{"frame_id": "view", "rgb_base64": "aW1hZ2U="}]}
        if name == "create_env":
            return {"handle": "test", "action_dim": 4, "action_low": [-1] * 4, "action_high": [1] * 4}
        if name == "reset_env":
            return {}
        if name == "observe_env":
            return observation
        if name == "step_env":
            return {"observation": observation, "info": {"success": self.success}, "reward": 1,
                    "executed_control_steps": arguments["num_steps"]}
        if name == "close_env":
            return {}
        raise AssertionError(f"Unexpected tool {name}")


def reply(name, arguments, identifier):
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": serialize_decision(name, arguments, "这是当轮原始解释。"),
        "reasoning_content": "保留原始推理字段。",
    }}]}


def run_episode(tmp_path, success=True, provider="qwen"):
    transport = Transport(success)
    answers = [reply("observe_env", {}, "observe-call"),
               reply("step_env", {"action": [0.1, 0, 0, -1], "num_steps": 5}, "action-call")]
    requests = []

    def model(payload, **kwargs):
        requests.append(copy.deepcopy(payload))
        return answers[len(requests) - 1]

    directory = tmp_path / "metaworld/reach-v3/attempt-0000"
    record = collector.collect_attempt(transport, ROW, directory=directory, seed=20310000,
                                       api_key="test-secret", endpoint="https://example.invalid", model=collector.PROVIDERS[provider]["model"],
                                       model_call=model, max_turns=2, provider=provider)
    return record, transport, requests, directory / "episode.json"


def test_real_decision_and_tool_chain_are_preserved(tmp_path):
    record, transport, requests, path = run_episode(tmp_path)
    assert record["success"] is True
    assert [name for name, _ in transport.calls] == ["create_env", "reset_env", "observe_env", "observe_env", "step_env", "close_env"]
    assert record["control_steps_requested"] == 5
    assert len(requests[0]["messages"]) == 2
    assert requests[0]["chat_template_kwargs"]["enable_thinking"] is True
    assert requests[0]["max_tokens"] == 8192
    assert "tools" not in requests[0] and "tool_choice" not in requests[0]
    assert requests[1]["messages"][-2]["role"] == "assistant"
    assert requests[1]["messages"][-1]["content"][-1]["type"] == "image_url"
    assert "test-secret" not in path.read_text()
    samples = samples_from_episode(path, {ROW["env_id"]: ROW}, HOLDOUT)
    assert len(samples) == 2
    assert len(samples[0]["images"]) == 1
    assert samples[0]["messages"][-1]["content"] == record["model_calls"][0]["response"]["choices"][0]["message"]["content"]
    assert samples[1]["messages"][2]["loss"] is False
    assert samples[1]["messages"][-1]["loss"] is True
    assert sum(message["content"].count("<image>") for message in samples[1]["messages"]) == len(samples[1]["images"])
    assert record["turns"][1]["tool_request"]["arguments"]["strict"] is True
    assert record["executed_control_steps"] == 5
    assert '"reward"' not in json.dumps(requests)
    assert 'official_success' not in json.dumps(requests)
    assert samples[1]["metadata"]["posthoc_annotation"] is False
    assert not any("qwen_think" in turn for turn in record["turns"])


def test_failure_is_not_exported(tmp_path):
    record, _, _, path = run_episode(tmp_path, success=False)
    assert not record["success"]
    assert samples_from_episode(path, {ROW["env_id"]: ROW}, HOLDOUT) == []


def test_deepseek_native_thinking_vision_and_export(tmp_path):
    record, transport, requests, path = run_episode(tmp_path, provider="deepseek")
    assert record["success"] is True
    assert record["schema_version"] == "openeta.deepseek_live_two_tool_xml.v1"
    assert record["action_source"] == "deepseek_api"
    assert record["provider"] == "deepseek"
    assert "deepseek-flash" in requests[0]["messages"][0]["content"]
    assert "Qwen" not in requests[0]["messages"][0]["content"]
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in requests[0]
    assert requests[0]["max_tokens"] == 8192
    assert requests[0]["messages"][1]["content"][-1]["type"] == "image_url"
    assert record["model_calls"][0]["response"]["choices"][0]["message"]["reasoning_content"] == "保留原始推理字段。"
    samples = samples_from_episode(path, {ROW["env_id"]: ROW}, HOLDOUT)
    assert len(samples) == 2
    assert samples[0]["schema_version"] == "openeta.sft.deepseek_xml_decision.v1"
    assert samples[0]["metadata"]["provider"] == "deepseek"
    assert samples[0]["metadata"]["action_source"] == "deepseek_api"
    record["action_source"] = "qwen_api"
    path.write_text(json.dumps(record))
    assert samples_from_episode(path, {ROW["env_id"]: ROW}, HOLDOUT) == []


def test_deepseek_disabled_thinking_uses_provider_switch():
    payload = collector._model_request("deepseek", "deepseek-flash", [], 8192, False)
    assert payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in payload
    assert "chat_template_kwargs" not in payload


def test_deepseek_cli_never_uses_qwen_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("INF_API_KEY", "qwen-only-secret")
    monkeypatch.setattr("sys.argv", ["collector", "--provider", "deepseek", "--manifest", str(tmp_path / "unused.json"), "--bench", "metaworld"])
    with pytest.raises(SystemExit) as error:
        collector.main()
    assert error.value.code == 2


@pytest.mark.parametrize("change", ["expert", "posthoc", "action"])
def test_synthetic_or_modified_actions_cannot_be_exported(tmp_path, change):
    record, _, _, path = run_episode(tmp_path)
    if change == "expert":
        record["action_source"] = "metaworld.official_scripted_policy"
    elif change == "posthoc":
        record["turns"][0]["qwen_think"] = "补写解释"
    else:
        record["turns"][1]["arguments"]["action"][0] = 0.9
    path.write_text(json.dumps(record))
    if change == "action":
        with pytest.raises(ValueError, match="differs"):
            samples_from_episode(path, {ROW["env_id"]: ROW}, HOLDOUT)
    else:
        assert samples_from_episode(path, {ROW["env_id"]: ROW}, HOLDOUT) == []


def test_host_initial_observation_allows_first_model_action(tmp_path):
    transport = Transport()
    record = collector.collect_attempt(transport, ROW, directory=tmp_path / "attempt", seed=20310000,
                                       api_key="test", endpoint="", model="Qwen", max_turns=1,
                                       model_call=lambda *args, **kwargs: reply("step_env", {"action": [0] * 4, "num_steps": 1}, "early"))
    assert [name for name, _ in transport.calls] == ["create_env", "reset_env", "observe_env", "step_env", "close_env"]
    assert record["turns"][0]["valid"] is True
    assert record["initial_frames"]


@pytest.mark.parametrize("value,expected", [([True], True), ([False], False), ([True, False], False), ("true", False)])
def test_success_is_environment_only(value, expected):
    assert collector._official_success({"info": {"success": value}}, bench="maniskill") is expected


def test_failed_attempt_keeps_only_summary(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [ROW]}))
    output = tmp_path / "data"
    monkeypatch.setenv("INF_API_KEY", "test")
    monkeypatch.setattr(collector, "SseSimulatorMcpTransport", lambda url: Transport())
    monkeypatch.setattr(collector, "_call_model", lambda *args, **kwargs: reply("task_complete", {}, "finish"))
    monkeypatch.setattr("sys.argv", ["collector", "--manifest", str(manifest), "--bench", "metaworld",
                                   "--output-root", str(output), "--holdout", str(HOLDOUT_PATH), "--attempt-limit", "1"])
    collector.main()
    assert not list(output.glob("*/*/attempt-*"))
    summary = json.loads((output / "metaworld.attempts.jsonl").read_text())
    assert summary["stop_reason"] == "model_finish"
    audit = json.loads(next((output / "audit").glob("*/*/*.json")).read_text())
    assert audit["images_retained"] is False
    assert audit["model_calls"][0]["response"]["choices"][0]["message"]["content"].startswith("<decision>")


def test_context_window_keeps_observation_before_each_answer():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "task"}]
    for index in range(4):
        messages.extend([{"role": "assistant", "content": str(index)},
                         {"role": "user", "content": f"observation-{index}"}])
    window = collector._request_messages(messages, 2)
    assert window[0] == messages[0]
    assert window[1]["content"] == "observation-1"
    assert window[2]["content"] == "2"
    assert window[-1]["content"] == "observation-3"


@pytest.mark.parametrize("content,finish", [("<think><decision></decision>", "stop"),
                                           ("<decision><kind>tool_call</kind></decision>", "stop"),
                                           (serialize_decision("step_env", {"action": [0] * 4, "num_steps": 1}), "length")])
def test_invalid_or_truncated_xml_never_executes(tmp_path, content, finish):
    transport = Transport()
    record = collector.collect_attempt(transport, ROW, directory=tmp_path / "attempt", seed=20310000,
                                       api_key="test", endpoint="", model="Qwen", max_turns=1,
                                       model_call=lambda *args, **kwargs: {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}]})
    assert "step_env" not in [name for name, _ in transport.calls]
    assert not record["success"]
    assert record["stop_reason"] in {"invalid_xml", "model_output_limit"}


def test_actual_steps_not_requested_steps_control_budget(tmp_path):
    class EarlyTransport(Transport):
        def call_tool(self, name, arguments, timeout_s):
            payload = super().call_tool(name, arguments, timeout_s)
            if name == "step_env":
                payload["executed_control_steps"] = 2
            return payload
    record = collector.collect_attempt(EarlyTransport(), ROW, directory=tmp_path / "attempt", seed=20310000,
                                       api_key="test", endpoint="", model="Qwen", max_turns=1,
                                       model_call=lambda *args, **kwargs: reply("step_env", {"action": [0] * 4, "num_steps": 5}, "step"))
    assert record["control_steps_requested"] == 5
    assert record["executed_control_steps"] == 2


@pytest.mark.parametrize("http_status", [504, 524])
def test_api_retries_preserve_context_and_execute_action_once(tmp_path, monkeypatch, http_status):
    transport = Transport()
    requests, delays = [], []
    monkeypatch.setattr(collector.time, "sleep", delays.append)

    def model(payload, **kwargs):
        requests.append(copy.deepcopy(payload))
        assert kwargs["timeout"] == 300
        if len(requests) == 1:
            return reply("observe_env", {}, "observe")
        if len(requests) in (2, 3):
            assert "step_env" not in [name for name, _ in transport.calls]
            raise HTTPError("https://example.invalid", http_status, "Gateway Timeout", {}, None)
        return reply("step_env", {"action": [0] * 4, "num_steps": 2}, "step")

    directory = tmp_path / "attempt"
    record = collector.collect_attempt(transport, ROW, directory=directory, seed=20310000,
                                       api_key="test", endpoint="", model="Qwen", max_turns=2, model_call=model)
    assert record["success"]
    assert delays == [10, 20]
    assert requests[1] == requests[2] == requests[3]
    assert len(record["turns"]) == len(record["model_calls"]) == 2
    assert [name for name, _ in transport.calls].count("step_env") == 1
    assert [entry["status"] for entry in record["model_calls"][1]["attempts"]] == ["error", "error", "success"]
    assert "error" not in record["model_calls"][1]
    assert len(samples_from_episode(directory / "episode.json", {ROW["env_id"]: ROW}, HOLDOUT)) == 2


def test_api_retry_exhaustion_is_bounded_and_persisted(tmp_path, monkeypatch):
    delays, requests = [], []
    monkeypatch.setattr(collector.time, "sleep", delays.append)
    transport = Transport()
    path = tmp_path / "attempt/episode.json"

    def model(payload, **kwargs):
        requests.append(payload)
        saved = json.loads(path.read_text())
        assert len(saved["model_calls"][0]["attempts"]) == len(requests)
        raise URLError(TimeoutError("slow model"))

    record = collector.collect_attempt(transport, ROW, directory=path.parent, seed=20310000,
                                       api_key="test", endpoint="", model="Qwen", model_call=model)
    assert len(requests) == 5
    assert delays == [10, 20, 40, 60]
    assert record["stop_reason"] == "infrastructure_error"
    assert record["model_calls"][0]["retry_exhausted"] is True
    assert record["turns"] == []
    assert [name for name, _ in transport.calls] == ["create_env", "reset_env", "observe_env", "close_env"]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_api_errors_are_not_retried(tmp_path, monkeypatch, status):
    delays = []
    monkeypatch.setattr(collector.time, "sleep", delays.append)

    def model(payload, **kwargs):
        raise HTTPError("https://example.invalid", status, "Permanent error", {}, None)

    record = collector.collect_attempt(Transport(), ROW, directory=tmp_path / "attempt", seed=20310000,
                                       api_key="test", endpoint="", model="Qwen", model_call=model)
    assert len(record["model_calls"][0]["attempts"]) == 1
    assert not delays
    assert record["stop_reason"] == "infrastructure_error"


def test_mcp_timeout_is_never_retried(tmp_path, monkeypatch):
    class TimeoutTransport(Transport):
        def call_tool(self, name, arguments, timeout_s):
            result = super().call_tool(name, arguments, timeout_s)
            if name == "step_env":
                raise TimeoutError("MCP result unknown")
            return result

    transport = TimeoutTransport()
    delays = []
    monkeypatch.setattr(collector.time, "sleep", delays.append)
    record = collector.collect_attempt(transport, ROW, directory=tmp_path / "attempt", seed=20310000,
                                       api_key="test", endpoint="", model="Qwen",
                                       model_call=lambda *args, **kwargs: reply("step_env", {"action": [0] * 4, "num_steps": 1}, "step"))
    assert record["stop_reason"] == "infrastructure_error"
    assert [name for name, _ in transport.calls].count("step_env") == 1
    assert len(record["model_calls"][0]["attempts"]) == 1
    assert not delays


def test_interrupt_during_backoff_stops_without_action(tmp_path, monkeypatch):
    def sleep(delay):
        raise KeyboardInterrupt

    def model(*args, **kwargs):
        raise TimeoutError("slow model")

    monkeypatch.setattr(collector.time, "sleep", sleep)
    transport = Transport()
    record = collector.collect_attempt(transport, ROW, directory=tmp_path / "attempt", seed=20310000,
                                       api_key="test", endpoint="", model="Qwen", model_call=model)
    assert record["stop_reason"] == "interrupted"
    assert len(record["model_calls"][0]["attempts"]) == 1
    assert "step_env" not in [name for name, _ in transport.calls]


@pytest.mark.parametrize("infrastructure_error", [False, True])
def test_diagnostic_batch_keeps_five_failed_episodes_and_images(tmp_path, monkeypatch, infrastructure_error):
    rows = [{**ROW, "task_id": f"reach-{index}", "env_id": f"openeta/metaworld_reach-{index}-v0"} for index in range(5)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": rows}))
    output = tmp_path / "diagnostics"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-secret")
    monkeypatch.setattr(collector, "SseSimulatorMcpTransport", lambda url: Transport())

    def model(*args, **kwargs):
        if infrastructure_error:
            raise TimeoutError("test API timeout")
        return reply("task_complete", {}, "finish")

    monkeypatch.setattr(collector, "_call_model", model)
    monkeypatch.setattr("sys.argv", ["collector", "--provider", "deepseek", "--manifest", str(manifest),
                                   "--bench", "metaworld", "--output-root", str(output),
                                   "--holdout", str(HOLDOUT_PATH), "--attempt-limit", "5",
                                   "--max-attempts-per-task", "1", "--model-max-attempts", "1",
                                   "--retain-failures", "--continue-on-infrastructure-error"])
    collector.main()
    paths = list(output.glob("metaworld/*/attempt-*/episode.json"))
    assert len(paths) == 5
    ledger = [json.loads(line) for line in (output / "metaworld.attempts.jsonl").read_text().splitlines()]
    assert len(ledger) == 5
    for path in paths:
        record = json.loads(path.read_text())
        assert record["success"] is False
        assert record["images_retained"] is True
        assert record["run_config"]["retain_failures"] is True
        assert record["run_config"]["continue_on_infrastructure_error"] is True
        assert record["stop_reason"] == ("infrastructure_error" if infrastructure_error else "model_finish")
        assert record["initial_frames"]
        assert all((path.parent / name).is_file() for name in record["initial_frames"])
        assert samples_from_episode(path, {row["env_id"]: row for row in rows}, HOLDOUT) == []

    first_pass = {path: path.read_bytes() for path in paths}
    arguments = list(collector.sys.argv)
    arguments[arguments.index("--attempt-limit") + 1] = "10"
    arguments[arguments.index("--max-attempts-per-task") + 1] = "3"
    monkeypatch.setattr("sys.argv", arguments)
    collector.main()
    assert len(list(output.glob("metaworld/*/attempt-*/episode.json"))) == 15
    assert all(path.read_bytes() == content for path, content in first_pass.items())
    for row in rows:
        attempts = sorted((output / "metaworld" / row["task_id"]).glob("attempt-*/episode.json"))
        assert [path.parent.name for path in attempts] == ["attempt-0000", "attempt-0001", "attempt-0002"]
        assert len({json.loads(path.read_text())["seed"] for path in attempts}) == 3
    collector.main()
    assert len((output / "metaworld.attempts.jsonl").read_text().splitlines()) == 15


def test_debug5_manifest_respects_holdout():
    manifest = HOLDOUT_PATH.parent / "deepseek_xml_debug5.v1.json"
    rows = json.loads(manifest.read_text())["tasks"]
    collector._validate_manifest(rows, HOLDOUT, require_full=False)
    assert len(rows) == 20
    for bench in collector.ACTION_DIMS:
        assert len({row["env_id"] for row in rows if row["bench"] == bench}) == 5
