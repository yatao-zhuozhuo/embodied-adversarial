import json
from types import SimpleNamespace

import pytest

from scripts import watch_two_tool_sft as watchdog


def setup_state(tmp_path, monkeypatch, **changes):
    state = {"successful_episodes": 0, "completed_attempts": 1, "pending_tasks": 25,
             "quota_complete": False, "manifest": str(tmp_path / "manifest.json")}
    state.update(changes)
    monkeypatch.setattr(watchdog, "progress", lambda *args: state.copy())
    monkeypatch.setattr(watchdog, "port_in_use", lambda port: False)
    launched = []

    def launch(command, **kwargs):
        launched.append((command, kwargs))
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr(watchdog.subprocess, "Popen", launch)
    return launched


def test_only_missing_collector_is_restarted(tmp_path, monkeypatch):
    launched = setup_state(tmp_path, monkeypatch)
    processes = [{"pid": 10, "kind": "collector"}]
    state = watchdog.check_bench(tmp_path, "metaworld", processes, {}, tmp_path)
    assert state["status"] == "running"
    assert not launched
    state = watchdog.check_bench(tmp_path, "metaworld", [], {"INF_API_KEY": "private-test"}, tmp_path)
    assert state["status"] == "restart_launched"
    command, options = launched[0]
    assert watchdog.option(command, "--max-tokens") == "8192"
    assert watchdog.option(command, "--model-max-attempts") == "5"
    assert "--enable-thinking" in command
    assert options["start_new_session"] is True
    assert options["env"]["OPENETA_TWO_TOOL_MCP_PORT"] == "8766"
    assert "private-test" not in json.dumps(state)


@pytest.mark.parametrize("quota,status", [(True, "quota_complete"), (False, "attempt_budget_exhausted")])
def test_finished_work_is_not_restarted(tmp_path, monkeypatch, quota, status):
    launched = setup_state(tmp_path, monkeypatch, pending_tasks=0, quota_complete=quota)
    state = watchdog.check_bench(tmp_path, "metaworld", [], {}, tmp_path)
    assert state["status"] == status
    assert not launched


def test_bootstrap_port_conflict_and_dry_run_never_duplicate(tmp_path, monkeypatch):
    launched = setup_state(tmp_path, monkeypatch)
    state = watchdog.check_bench(tmp_path, "metaworld", [{"pid": 1, "kind": "launcher"}], {}, tmp_path)
    assert state["status"] == "starting_or_stopping"
    state = watchdog.check_bench(tmp_path, "metaworld", [], {}, tmp_path, dry_run=True)
    assert state["status"] == "would_restart"
    monkeypatch.setattr(watchdog, "port_in_use", lambda port: True)
    state = watchdog.check_bench(tmp_path, "metaworld", [], {}, tmp_path)
    assert state["status"] == "port_conflict"
    assert not launched


def test_private_environment_is_required(tmp_path):
    path = tmp_path / "environment.json"
    path.write_text(json.dumps({"INF_API_KEY": "private-test"}))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        watchdog.load_environment(path)
    path.chmod(0o600)
    assert watchdog.load_environment(path)["INF_API_KEY"] == "private-test"


def test_progress_respects_attempt_budget_and_trusted_success(tmp_path):
    manifest = tmp_path / "SFT_data/manifests/two_tool_metaworld_25.v1.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"tasks": [{"bench": "metaworld", "task_id": "task", "env_id": "env"}]}))
    root = tmp_path / "SFT_data/qwen_xml_v1"
    root.mkdir()
    (root / "metaworld.attempts.jsonl").write_text((json.dumps({"env_id": "env", "stop_reason": "turn_limit"}) + "\n") * 20)
    assert watchdog.progress(tmp_path, "metaworld")["pending_tasks"] == 0
    assert watchdog.progress(tmp_path, "metaworld")["quota_complete"] is False
    for attempt in range(3):
        path = root / "metaworld/task" / f"attempt-{attempt:04d}/episode.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"schema_version": watchdog.SCHEMA, "action_source": "qwen_api",
                                   "env_id": "env", "success": True, "stop_reason": "official_success",
                                   "turns": [{"tool": "step_env", "official_success": True, "receipt": {"info": {"success": True}}}]}))
    assert watchdog.progress(tmp_path, "metaworld")["quota_complete"] is True
