import unittest

from adapter.dummy_sim import DummySimulatorAdapter
from adapter.protocol import EnvAction


class EmbodiedSnapshotTest(unittest.TestCase):
    def test_dummy_snapshot_restores_initial_state(self) -> None:
        env = DummySimulatorAdapter()
        initial = env.reset(task="alice task", seed=3)
        snapshot = env.capture_snapshot()
        env.step(EnvAction(action_type="MOVE"))
        self.assertNotEqual(env.observe().metadata["step_idx"], initial.metadata["step_idx"])

        env.restore_snapshot(snapshot)
        restored = env.observe()
        self.assertEqual(restored.to_dict(), initial.to_dict())

    def test_unknown_snapshot_is_rejected(self) -> None:
        env = DummySimulatorAdapter()
        env.reset()
        snapshot = env.capture_snapshot()
        other = DummySimulatorAdapter()
        with self.assertRaises(FileNotFoundError):
            other.restore_snapshot(snapshot)
