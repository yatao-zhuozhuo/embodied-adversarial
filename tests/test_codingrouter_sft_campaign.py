import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

from scripts import run_gemini_sft_comparison as campaign
from agent.runtime.two_tool_xml import serialize_decision


@pytest.mark.parametrize("available", [True, False])
@pytest.mark.parametrize("attempts", [1, 3])
def test_single_gpt_campaign_preflight_gate_and_twenty_tasks(tmp_path, monkeypatch, available, attempts):
    monkeypatch.setattr(campaign, "ROOT", tmp_path)
    monkeypatch.setenv("CODINGROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(campaign.sys, "argv", ["campaign", "--provider", "codingrouter", "--models",
                                             "gpt-5.6-luna", "--stamp", "test", "--attempts-per-task", str(attempts)])
    source = tmp_path / "SFT_data/deepseek_input_v1_nothink_20261006_065047/metaworld/task/attempt-0000/episode.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({"model_calls": [{"input_package": {
        "messages": [{"role": "user", "content": "real input fixture"}]}}]}))
    commands = []

    def request(payload, **kwargs):
        assert payload["model"] == "gpt-5.6-luna"
        assert "thinking" not in payload and "reasoning_effort" not in payload
        assert kwargs["endpoint"] == "https://api.codingrouter.ai/v1/chat/completions"
        if not available:
            raise HTTPError(kwargs["endpoint"], 401, "Unauthorized", {}, None)
        return {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": serialize_decision("observe_env", {}, "Inspect current view.")}}]}

    class Process:
        pid = 12345

        def __init__(self, command, **kwargs):
            commands.append(command)
            assert kwargs["env"]["CODINGROUTER_API_KEY"] == "test-secret"

        def wait(self):
            return 0

    monkeypatch.setattr(campaign.collector, "_call_model", request)
    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    if available:
        campaign.main()
        assert len(commands) == 4
        assert {command[command.index("--bench") + 1] for command in commands} == set(campaign.BENCHES)
        for command in commands:
            assert command[command.index("--task-limit") + 1] == "5"
            assert command[command.index("--max-attempts-per-task") + 1] == str(attempts)
            assert command[command.index("--target-per-task") + 1] == str(attempts)
            assert command[command.index("--attempt-limit") + 1] == str(5 * attempts)
            assert "--retain-failures" in command and "--unified-input" in command
            assert "test-secret" not in str(command)
    else:
        with pytest.raises(SystemExit, match="Preflight failed"):
            campaign.main()
        assert not commands
        status = json.loads((tmp_path / "SFT_data/reports/codingrouter_comparison_test/campaign.status.json").read_text())
        assert status["episodes_launched"] == 0


def test_continuation_waits_for_original_process_without_signalling(tmp_path, monkeypatch):
    (tmp_path / "metaworld.current.json").write_text(json.dumps({
        "pid": 123, "model": "gemini-3.6-flash", "command": ["bash", "collect"]}))
    commands = iter([b"bash\0collect\0", b"unrelated-process\0"])
    delays = []
    monkeypatch.setattr(Path, "read_bytes", lambda path: next(commands))
    monkeypatch.setattr(campaign.time, "sleep", delays.append)
    campaign.wait_for_previous_lane(tmp_path, "metaworld", "gemini-3.6-flash")
    assert delays == [15]
    with pytest.raises(ValueError, match="does not match"):
        campaign.wait_for_previous_lane(tmp_path, "metaworld", "gemini-3.1-pro-preview")
