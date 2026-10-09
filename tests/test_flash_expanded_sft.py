import json
from pathlib import Path
import threading

from scripts import run_flash_expanded_sft as campaign
from scripts.collect_two_tool_sft import _write_json


def test_single_attempt_global_success_cap_and_resume(tmp_path, monkeypatch):
    rows = [{"bench": bench, "task_id": f"new-{index}", "env_id": f"openeta/{bench}_new-{index}",
             "task": f"New task {index}"} for bench in campaign.ACTION_DIMS for index in range(3)]
    _write_json(tmp_path / "manifest.json", {"tasks": rows, "counts": {bench: 3 for bench in campaign.ACTION_DIMS}})
    monkeypatch.setenv("CODINGROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(campaign.sys, "argv", ["campaign", "--output-root", str(tmp_path), "--target", "3"])
    monkeypatch.setattr(campaign.signal, "signal", lambda *args: None)
    monkeypatch.setattr(campaign, "samples_from_episode", lambda *args: ["verified"])
    commands = []

    class Process:
        pid = 999999999

        def __init__(self, command, **kwargs):
            self.command = command
            commands.append(command)

        def wait(self):
            identity = self.command[self.command.index("--env-id") + 1]
            row = next(row for row in rows if row["env_id"] == identity)
            _write_json(tmp_path / row["bench"] / row["task_id"] / "attempt-0000/episode.json",
                        {**row, "success": True, "stop_reason": "official_success"})
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    campaign.main()
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert state["status"] == "target_reached"
    assert len(state["successes"]) == len(commands) == 3
    assert len(set(state["successes"])) == 3
    for command in commands:
        assert command[command.index("--max-attempts-per-task") + 1] == "1"
        assert command[command.index("--attempt-limit") + 1] == "1"
    campaign.main()
    assert len(commands) == 3


def test_expanded_maniskill_profiles_are_strict():
    from tests.test_two_tool_input import profile
    from agent.runtime.two_tool_input import task_description

    for task in ["AssemblingKits-v1", "TurnFaucet-v1"]:
        selected = profile("maniskill", task)
        assert selected["wrist"] == ["hand_camera"]
        assert selected["camera_dimensions"]["hand_camera"] == [128, 128]
        assert not task_description(task, "ManiSkill task: " + task).startswith("ManiSkill task:")


def test_failed_tasks_retry_after_round_barrier_and_success_is_not_repeated(tmp_path, monkeypatch):
    rows = [{"bench": "metaworld", "task_id": f"new-{index}", "env_id": f"openeta/metaworld_new-{index}",
             "task": f"New task {index}"} for index in range(3)]
    _write_json(tmp_path / "manifest.json", {"tasks": rows, "counts": {"metaworld": 3}})
    monkeypatch.setenv("CODINGROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(campaign.sys, "argv", ["campaign", "--output-root", str(tmp_path), "--target", "100"])
    monkeypatch.setattr(campaign.signal, "signal", lambda *args: None)
    monkeypatch.setattr(campaign, "samples_from_episode", lambda *args: ["verified"])
    launches = []

    class Process:
        pid = 999999999

        def __init__(self, command, **kwargs):
            self.identity = command[command.index("--env-id") + 1]
            self.round = int(command[command.index("--max-attempts-per-task") + 1]) - 1
            launches.append((self.identity, self.round))

        def wait(self):
            row = next(row for row in rows if row["env_id"] == self.identity)
            success = row["task_id"] == "new-0" and self.round == 1
            _write_json(tmp_path / "metaworld" / row["task_id"] / f"attempt-{self.round:04d}/episode.json",
                        {**row, "success": success, "stop_reason": "official_success" if success else "turn_limit"})
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    campaign.main()
    assert [round_index for identity, round_index in launches] == [0, 0, 0, 1, 1, 1, 2, 2]
    assert (rows[0]["env_id"], 2) not in launches
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert state["success_count"] == 1
    assert len(state["attempt_history"]) == 8
    campaign.main()
    assert len(launches) == 8


def test_eight_concurrent_tasks_with_distinct_ports(tmp_path, monkeypatch):
    rows = [{"bench": "metaworld", "task_id": f"new-{index}", "env_id": f"openeta/metaworld_new-{index}",
             "task": f"New task {index}"} for index in range(8)]
    _write_json(tmp_path / "manifest.json", {"tasks": rows, "counts": {"metaworld": 8}})
    monkeypatch.setenv("CODINGROUTER_API_KEY", "test-secret")
    monkeypatch.setattr(campaign.sys, "argv", ["campaign", "--output-root", str(tmp_path),
                                             "--target", "8", "--concurrency", "8"])
    monkeypatch.setattr(campaign.signal, "signal", lambda *args: None)
    monkeypatch.setattr(campaign, "samples_from_episode", lambda *args: ["verified"])
    barrier = threading.Barrier(8, timeout=10)
    ports = []

    class Process:
        pid = 999999999

        def __init__(self, command, **kwargs):
            self.identity = command[command.index("--env-id") + 1]
            ports.append(kwargs["env"]["OPENETA_TWO_TOOL_MCP_PORT"])

        def wait(self):
            barrier.wait()
            row = next(row for row in rows if row["env_id"] == self.identity)
            _write_json(tmp_path / "metaworld" / row["task_id"] / "attempt-0000/episode.json",
                        {**row, "success": True, "stop_reason": "official_success"})
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(campaign.subprocess, "Popen", Process)
    campaign.main()
    state = json.loads((tmp_path / "campaign.json").read_text())
    assert state["success_count"] == 8
    assert state["concurrency"] == 8
    assert len(set(ports)) == 8
    assert state["status"] == "target_reached"
