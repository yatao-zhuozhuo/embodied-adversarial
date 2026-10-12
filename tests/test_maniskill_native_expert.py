import unittest

import numpy as np

from scripts.generate_maniskill_native_expert import Finished, RecordingEnv


class NativeEnv:
    action_space = type("Space", (), {"shape": (7,)})()

    def __init__(self):
        self.unwrapped = self
        self.steps = 0

    def reset(self, seed=None):
        self.seed = seed
        return {}, {"success": False}

    def get_state_dict(self):
        return {"seed": self.seed}

    def step(self, action):
        self.steps += 1
        return {}, 0, False, False, {"success": self.steps == 7}


class NativeExpertSafetyTest(unittest.TestCase):
    def setUp(self):
        self.raw = NativeEnv()
        self.env = RecordingEnv(self.raw, {"task_id": "PickCube-v1", "seed": 123}, 4)

    def test_wrong_seed_rejected(self):
        with self.assertRaises(ValueError):
            self.env.reset(seed=124)

    def test_reset_is_not_an_expert_decision(self):
        self.env.reset(seed=123)
        self.assertEqual(self.raw.steps, 5)
        self.assertEqual(self.env.actions, [])
        with self.assertRaises(ValueError):
            self.env.reset(seed=123)

    def test_first_environment_success_stops_expert(self):
        self.env.reset(seed=123)
        self.env.step(np.zeros(7))
        with self.assertRaises(Finished):
            self.env.step(np.zeros(7))
        self.assertTrue(self.env.success)
        self.assertEqual(len(self.env.actions), 2)

    def test_invalid_action_never_executes(self):
        self.env.reset(seed=123)
        for action in [np.zeros(8), np.full(7, np.nan), np.full(7, 1.1)]:
            with self.assertRaises(ValueError):
                self.env.step(action)
        self.assertEqual(self.raw.steps, 5)


if __name__ == "__main__":
    unittest.main()
