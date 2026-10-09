import base64
import copy
import io
import json

import pytest
from PIL import Image

from agent.runtime import two_tool_input as inputs
from scripts import collect_two_tool_sft as collector
from scripts.export_two_tool_sft import samples_from_episode
from tests.test_two_tool_sft import Transport, ROW, HOLDOUT, reply


def profile(bench="metaworld", task="reach-v3"):
    dimension = collector.ACTION_DIMS[bench]
    return inputs.make_profile(bench, task, {"input_control_contract": {
        "slots": [str(index) for index in range(dimension)], "mode": "pd_ee_delta_pose",
        "tcp": {"frame": "world"}}}, {"action_dim": dimension, "action_low": [-1]*dimension, "action_high": [1]*dimension})


def observation(tmp_path, selected, turn=0):
    frames, cameras = [], []
    for index, name in enumerate(selected["primary"] + selected["wrist"]):
        path = tmp_path / f"{turn:04d}-{name}.png"
        Image.new("RGB", tuple(selected["camera_dimensions"][name]), (index * 50, turn * 30, 30)).save(path)
        frames.append(path.name)
        cameras.append({"frame_id": name})
    return inputs.snapshot({"cameras": cameras, "robot": {"end_effector_pose": {"xyz": [turn, 0, 0]}},
                            "reward": 123, "objects": [{"secret": True}]}, frames, tmp_path, selected, f"obs-{turn}", turn)


@pytest.mark.parametrize("bench,task,count", [("metaworld","reach-v3",2), ("libero","libero_spatial_task6",3),
                                            ("maniskill","PullCube-v1",2), ("maniskill","PickSingleYCB-v1",3),
                                            ("robocasa","CoffeeSetupMug",5)])
def test_profile_camera_order_and_bounded_text_history(tmp_path, bench, task, count):
    selected = profile(bench, task)
    previous, current = observation(tmp_path, selected, 0), observation(tmp_path, selected, 1)
    history = [{"turn": index, "model_message": {"content": "<think>hidden</think><decision>answer</decision>",
                                               "reasoning_content": "secret thinking"},
                "valid": True, "validation": {"valid": True}, "tool": "step_env",
                "arguments": {"action": [0], "num_steps": 5}, "executed_control_steps": 2,
                "reward": 123, "official_success": True} for index in range(1, 7)]
    package = inputs.build_input(task="Do the task", profile=selected, current=current, previous=previous,
                                 history=history, turn=7, max_turns=60, remaining_steps=398)
    assert len(package["images"]) == count
    assert package["history_decision_ids"] == [3,4,5,6]
    assert all(isinstance(message["content"], str) for message in package["messages"][:-1])
    assert "hidden" not in json.dumps(package["messages"])
    assert "secret" not in json.dumps(package["messages"])
    assert "official_success" not in json.dumps(package["messages"])
    assert '"reward"' not in json.dumps(package["messages"])
    assert package["dynamic"]["last_action"]["executed_steps"] == 2
    assert package["dynamic"]["robot_delta"]["tcp_displacement_m"] == [1,0,0]
    assert package["images"][0]["references"][0]["timing"] == "previous_decision_before"


def test_identical_evidence_keeps_both_time_references(tmp_path):
    selected = profile()
    current = observation(tmp_path, selected)
    package = inputs.build_input(task="Reach", profile=selected, current=current, previous=current,
                                 history=[], turn=2, max_turns=60, remaining_steps=400)
    assert len(package["images"]) == 1
    assert len(package["images"][0]["references"]) == 2
    assert package["dynamic"]["observation"]["is_new"] is False


def test_missing_camera_and_unknown_execution_fail_closed(tmp_path):
    with pytest.raises(ValueError, match="Required camera missing"):
        inputs.snapshot({"cameras": []}, [], tmp_path, profile(), "obs-0", None)
    feedback = inputs.execution_feedback({"turn": 1, "tool_request": {"name": "step_env"}, "valid": False})
    assert feedback["execution"] == "error"
    assert feedback["last_action"]["executed_steps"] is None
    assert feedback["result_known"] is False


def test_unified_collection_and_export_share_exact_conditioning(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (256,256), "red").save(image, format="PNG")

    class UnifiedTransport(Transport):
        def call_tool(self, name, arguments, timeout_s):
            result = super().call_tool(name, arguments, timeout_s)
            if name == "create_env":
                result["input_control_contract"] = {"slots": ["dx","dy","dz","gripper"]}
            observation = result.get("observation", result)
            for camera in observation.get("cameras", []):
                camera["rgb_base64"] = base64.b64encode(image.getvalue()).decode()
            return result

    requests = []

    def model(payload, **kwargs):
        requests.append(copy.deepcopy(payload))
        return reply("observe_env", {}, "observe") if len(requests) == 1 else reply("step_env", {"action": [0]*4, "num_steps": 1}, "step")

    directory = tmp_path / "metaworld/reach-v3/attempt-0000"
    record = collector.collect_attempt(UnifiedTransport(), ROW, directory=directory, seed=20310000,
                                       api_key="test", endpoint="", model="deepseek-flash", provider="deepseek",
                                       model_call=model, max_turns=2, unified_input=True, enable_thinking=False)
    assert record["success"], record.get("error")
    assert record["input_protocol"] == inputs.INPUT_PROTOCOL
    assert requests[0]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in requests[0]
    assert "reasoning_content" not in json.dumps(requests[1])
    assert record["model_calls"][1]["input_package"]["messages"] == requests[1]["messages"]
    samples = samples_from_episode(directory / "episode.json", {ROW["env_id"]: ROW}, HOLDOUT)
    assert len(samples) == 2
    assert sum(message.get("loss") is True for message in samples[1]["messages"]) == 1
    assert samples[1]["images"] == [image["path"] for image in record["model_calls"][1]["input_package"]["images"]]
