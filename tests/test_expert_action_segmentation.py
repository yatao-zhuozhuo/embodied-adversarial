import unittest

import numpy as np

from scripts.expert_action_segmentation import budgeted_segments


class BudgetedSegmentationTest(unittest.TestCase):
    def test_controls_and_command_integral_are_preserved(self):
        generator = np.random.default_rng(42)
        actions = generator.uniform(-0.5, 0.5, (60, 7))
        actions[:, -1] = -1
        segments = budgeted_segments(actions.tolist(), 15)
        self.assertEqual(len(segments), 15)
        self.assertEqual(sum(segment["num_steps"] for segment in segments), 60)
        self.assertTrue(all(1 <= segment["num_steps"] <= 10 for segment in segments))
        actual = sum(np.asarray(segment["action"]) * segment["num_steps"] for segment in segments)
        np.testing.assert_allclose(actual, actions.sum(axis=0), atol=1e-12)

    def test_gripper_transition_neighborhood_executes_exact_actions(self):
        actions = np.zeros((40, 7))
        actions[:, -1] = -1
        actions[20:, -1] = 1
        actions[:, 0] = np.arange(40) / 40
        segments = budgeted_segments(actions.tolist(), 18, protection_radius=3)
        offset = 0
        for segment in segments:
            if any(index in range(17, 24) for index in range(offset, offset + segment["num_steps"])):
                self.assertEqual(segment["num_steps"], 1)
                np.testing.assert_array_equal(segment["action"], actions[offset])
            self.assertTrue(np.all(actions[offset:offset + segment["num_steps"], -1] == segment["action"][-1]))
            offset += segment["num_steps"]

    def test_impossible_budget_rejected(self):
        with self.assertRaises(ValueError):
            budgeted_segments([[0, 0, 0, 0, 0, 0, -1]] * 31, 3)


if __name__ == "__main__":
    unittest.main()
