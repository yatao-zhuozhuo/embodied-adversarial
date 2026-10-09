import numpy as np
import torch

from agent.training.embodied_grpo import (
    PolicyStep,
    clipped_grpo_objective,
    episode_reward,
    group_advantages,
    parse_action,
)


def _step(**overrides):
    values = {
        "prompt": "prompt",
        "image": np.zeros((2, 2, 3), dtype=np.uint8),
        "completion_ids": [1],
        "completion": '{"action":"GRASP"}',
        "action": "GRASP",
        "valid_action": True,
        "environment_reward": 0.1,
        "success": False,
        "is_grasped": False,
        "is_obj_placed": False,
        "terminated": False,
        "truncated": False,
    }
    values.update(overrides)
    return PolicyStep(**values)


def test_parse_action_is_bounded():
    assert parse_action('{"action":"grasp"}').action == "GRASP"
    parsed = parse_action('{"action":"RUN_PYTHON"}')
    assert parsed.action == "DONE"
    assert not parsed.valid


def test_reward_uses_trusted_receipts():
    plain = episode_reward([_step()], role="bob")
    grasp = episode_reward([_step(is_grasped=True)], role="bob")
    success = episode_reward([_step(is_obj_placed=True)], role="bob")
    assert plain < grasp < success


def test_group_advantages_are_centered():
    advantages = group_advantages([0.0, 0.5, 1.0])
    assert abs(sum(advantages)) < 1e-8
    assert advantages[0] < advantages[1] < advantages[2]


def test_clipped_loss_backpropagates():
    current = torch.tensor([-0.2, -0.4], requires_grad=True)
    old = torch.tensor([-0.3, -0.3])
    reference = torch.tensor([-0.25, -0.35])
    loss = clipped_grpo_objective(
        current,
        old,
        reference,
        1.0,
        clip_epsilon=0.2,
        beta=0.01,
    )
    loss.backward()
    assert current.grad is not None
    assert torch.isfinite(current.grad).all()
