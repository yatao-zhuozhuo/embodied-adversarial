"""Controller and observation contracts for the four replacement tasks."""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from agent.runtime.two_tool_input import make_profile
from sim.two_tool_contract import controller_contract
from sim.unified_env import UnifiedEnv


def test_stick_cartesian_controller_has_six_slots_without_gripper(monkeypatch):
    monkeypatch.setattr('sim.two_tool_contract.inspect.getfile', lambda _: __file__)
    arm = SimpleNamespace(config=SimpleNamespace(frame='root_translation', pos_lower=-0.1,
        pos_upper=0.1, rot_lower=-0.1, use_delta=True))
    raw = SimpleNamespace(agent=SimpleNamespace(controller=SimpleNamespace(controllers={'arm': arm})),
                          control_mode='pd_ee_delta_pose')
    native = controller_contract(SimpleNamespace(_unwrap=lambda: raw), 'maniskill')
    assert len(native['slots']) == 6
    assert native['gripper']['supported'] is False
    profile = make_profile('maniskill', 'DrawTriangle-v1', {'input_control_contract': native}, {'action_dim':6})
    assert profile['primary'] == ['base_camera'] and not profile['wrist']


def test_hand_joint_controller_preserves_native_joint_limits(monkeypatch):
    monkeypatch.setattr('sim.two_tool_contract.inspect.getfile', lambda _: __file__)
    joints = SimpleNamespace(config=SimpleNamespace(joint_names=[f'joint_{i}' for i in range(16)],
                              normalize_action=False, use_delta=False),
                             single_action_space=gym.spaces.Box(-0.5, 1.5, (16,), dtype=np.float32))
    raw = SimpleNamespace(agent=SimpleNamespace(controller=joints), control_mode='pd_joint_pos')
    native = controller_contract(SimpleNamespace(_unwrap=lambda: raw), 'maniskill')
    assert len(native['slots']) == 16
    assert native['components'][0]['action_high'] == [1.5]*16
    make_profile('maniskill', 'RotateSingleObjectInHandLevel3-v1', {'input_control_contract': native}, {'action_dim':16})
    with pytest.raises(ValueError, match='mode changed'):
        make_profile('maniskill', 'DrawTriangle-v1', {'input_control_contract': native}, {'action_dim':16})


def test_hand_observation_keeps_all_joints_and_exposes_goal_axis():
    env = object.__new__(UnifiedEnv)
    env._include_objects = False
    env._unwrap = lambda: SimpleNamespace(agent=None)
    qpos = np.arange(16, dtype=np.float32)
    obs = env._normalise_maniskill({'agent': {'qpos': qpos}, 'extra': {'rotate_dir': np.array([0,1,0])}})
    np.testing.assert_array_equal(obs['proprio']['joint_positions'], qpos)
    assert 'Y axis' in obs['task_description']
