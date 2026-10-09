import math

import pytest

from scripts.compress_expert_replays import group_actions, selected_episodes
from scripts.maniskill_demo_actions import canonicalize_pd_ee_delta_pose


def test_maniskill_rotation_clips_norm_not_components():
    raw = [2.0, -2.0, 0.25, 0.0, 1.0, 2.0, -2.0]
    converted = canonicalize_pd_ee_delta_pose(raw)
    assert converted[:3] == [1.0, -1.0, 0.25]
    assert converted[3:6] == pytest.approx([0.0, 1 / math.sqrt(5), 2 / math.sqrt(5)])
    assert converted[6] == -1.0


def test_group_actions_repeats_at_most_ten_steps():
    actions = [[0.01, -1.0]] * 23
    assert [segment["num_steps"] for segment in group_actions(actions, 0.0)] == [10, 10, 3]


def test_group_actions_preserves_large_gripper_change():
    actions = [[0.0, -1.0], [0.02, -1.0], [0.03, 1.0]]
    assert [segment["num_steps"] for segment in group_actions(actions, 0.05)] == [2, 1]


def test_compression_requires_all_original_successes(tmp_path):
    with pytest.raises(ValueError, match="shortfall"):
        selected_episodes(tmp_path, allow_partial=False)
