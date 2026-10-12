"""MetaWorld must be seeded before task selection and during construction."""
from types import ModuleType
from unittest.mock import patch

import gymnasium as gym
import numpy as np

from sim.env_registry import _MetaWorldSeededReset, _make_metaworld_direct


class RawIgnoringResetSeed(gym.Env):
    action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    observation_space = gym.spaces.Box(0, 50, shape=(1,), dtype=np.float64)

    def seed(self, seed):
        self.np_random = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        return np.asarray([self.selected_task], dtype=float), {}


class SelectBeforeReset(gym.Wrapper):
    def reset(self, *, seed=None, options=None):
        self.unwrapped.selected_task = self.np_random.integers(50)
        return self.env.reset(seed=seed, options=options)


def test_seed_is_applied_before_ignoring_reset_selects_a_task():
    env = _MetaWorldSeededReset(SelectBeforeReset(RawIgnoringResetSeed()))
    first, _ = env.reset(seed=1234)
    env.reset(seed=5678)
    repeated, _ = env.reset(seed=1234)
    np.testing.assert_array_equal(first, repeated)
    assert first[0] == np.random.default_rng(1234).integers(50)


def test_creation_seed_is_forwarded_to_mt1_task_pool():
    module = ModuleType('metaworld')
    module.register_mw_envs = lambda: None
    raw = SelectBeforeReset(RawIgnoringResetSeed())
    with patch.dict('sys.modules', {'metaworld': module}), patch('sim.env_registry.gym.make', return_value=raw) as make:
        env = _make_metaworld_direct('assembly-v3', seed=314159)
    assert make.call_args.kwargs['seed'] == 314159
    assert env._openeta_creation_seed == 314159
    assert env._openeta_seed_protocol == 'metaworld-create-and-preselect-seed.v1'
