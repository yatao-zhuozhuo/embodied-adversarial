import pytest

from agent.runtime.embodied_two_tool import parse_legacy_command as parse_command, project_observation


BOUNDS = {"action_low": [-1.0] * 7, "action_high": [1.0] * 7}


def test_only_exact_commands_are_accepted():
    assert parse_command('{"tool":"observe_env"}', **BOUNDS) == {"tool": "observe_env"}
    assert parse_command('{"response":"finish"}', **BOUNDS) == {"response": "finish"}
    command = parse_command('{"tool":"step_env","action":[0,0,0,0,0,0,1],"num_steps":10}', **BOUNDS)
    assert command["num_steps"] == 10
    for raw in (
        '{"tool":"observe_env","handle":"secret"}',
        '{"tool":"step_env","action":[0,0,0,0,0,0,true],"num_steps":1}',
        '{"tool":"step_env","action":[0,0,0,0,0,0,1.1],"num_steps":1}',
        '{"tool":"step_env","action":[0,0,0,0,0,0,1],"num_steps":true}',
        '{"tool":"step_env","action":[0,0,0,0,0,0,1],"num_steps":11}',
        '{"tool":"step_env","action":[0,0,0,0,0,0,NaN],"num_steps":1}',
    ):
        with pytest.raises(ValueError):
            parse_command(raw, **BOUNDS)


def test_visible_projection_drops_truth_fields():
    projected = project_observation({
        "objects": [{"name": "cube", "position": [1, 2, 3]}],
        "metadata": {"success": True},
        "robot": {
            "joint_positions": [0], "joint_velocities": [0],
            "end_effector_pose": {"xyz": [0, 0, 0], "quat_xyzw": [0, 0, 0, 1], "secret": 1},
            "gripper_state": {"open": True, "openness": 1, "secret": 2},
            "metadata": {"reward": 10},
        },
    })
    assert "objects" not in projected and "metadata" not in projected
    assert "secret" not in str(projected)
