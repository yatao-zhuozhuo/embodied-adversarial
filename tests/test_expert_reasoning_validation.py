import pytest

from scripts.annotate_expert_sft_reasoning import _reasoning_from_response


def response(content: str) -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


def test_task_object_opening_is_not_gripper_contradiction():
    text, _ = _reasoning_from_response(
        response("The hand pushes the handle toward opening the microwave door."),
        gripper_state="close")
    assert "microwave" in text


def test_explicit_wrong_gripper_state_is_rejected():
    with pytest.raises(ValueError, match="opening"):
        _reasoning_from_response(response("The robot opens the gripper to release the object."),
                                 gripper_state="close")
