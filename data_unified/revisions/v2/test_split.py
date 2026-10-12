"""Regression checks for task aliases and initial-state leakage."""
import unittest
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from data_unified.build import canonical, fingerprint, partition_cached
from data_unified.check import HERE, check


class SplitTests(unittest.TestCase):
    def records(self, n=50):
        return [{'reference_demo_index': i, 'state_fingerprint': f'state-{i}'} for i in range(n)]

    def test_robocasa_alias_cannot_be_new_task(self):
        self.assertEqual(canonical('openeta/robocasa_pretrain_OpenDrawer-v0'),
                         canonical('openeta/robocasa_target_OpenDrawer-v0'))

    def test_metaworld_suite_alias_cannot_be_new_task(self):
        self.assertEqual(canonical('openeta/metaworld_50_drawer-open-v3-v0'),
                         canonical('openeta/metaworld_45_ood_drawer-open-v3-v0'))

    def test_protected_demo_equivalence_group_never_enters_test(self):
        records = self.records()
        records.append({'reference_demo_index': 100, 'state_fingerprint': 'state-0'})
        counts = {'sft_train': 15, 'rl_train': 15, 'eval_seen': 20}
        split = partition_cached(records, {100}, 'fixed', counts)
        self.assertNotIn('state-0', {r['state_fingerprint'] for r in split['eval_seen']})
        sets = [{r['state_fingerprint'] for r in rows} for rows in split.values()]
        self.assertEqual(sum(map(len, sets)), len(set.union(*sets)))
        self.assertEqual(split, partition_cached(records, {100}, 'fixed', counts))

    def test_repeated_demos_do_not_fill_independent_state_quota(self):
        records = self.records(49)
        records.append({'reference_demo_index': 100, 'state_fingerprint': 'state-0'})
        with self.assertRaises(ValueError):
            partition_cached(records, set(), 'fixed', {'sft_train': 15, 'rl_train': 15, 'eval_seen': 20})

    def test_excess_historical_exposure_fails_instead_of_leaking(self):
        with self.assertRaises(ValueError):
            partition_cached(self.records(100), set(range(31)), 'fixed',
                             {'sft_train': 15, 'rl_train': 15, 'eval_seen': 20})

    def test_validation_does_not_reassign_frozen_training_or_test_states(self):
        counts = {'sft_train': 15, 'rl_train': 15, 'eval_seen': 20}
        for size in (50, 100):
            old = partition_cached(self.records(size), {0, 1}, 'fixed', counts)
            new = partition_cached(self.records(size), {0, 1}, 'fixed', {**counts, 'validation': 1})
            for phase in counts:
                self.assertEqual(old[phase], new[phase])
            if size == 50:
                self.assertNotIn('validation', new)  # caller creates native seed candidate
            else:
                self.assertEqual(len(new['validation']), 1)
                self.assertNotIn(new['validation'][0]['state_fingerprint'],
                                 {r['state_fingerprint'] for rows in old.values() for r in rows})

    def test_simulation_time_does_not_make_new_mujoco_initial_state(self):
        self.assertEqual(fingerprint({'state': [0, 0.1, 0.2]}, mujoco=True),
                         fingerprint({'state': [2, 0.1, 0.2]}, mujoco=True))
        self.assertNotEqual(fingerprint({'state': [0, 0.1, 0.2]}, mujoco=True),
                            fingerprint({'state': [0, 0.2, 0.2]}, mujoco=True))

    def test_checker_rejects_controller_drift_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'partition'
            shutil.copytree(HERE, root, ignore=shutil.ignore_patterns('__pycache__', 'revisions'))
            path = root / 'manifests/rl_train.json'
            data = json.loads(path.read_text())
            data['tasks'][0]['expected_action_dim'] = 99
            path.write_text(json.dumps(data))
            checksum_path = root / 'checksums.json'
            checksums = json.loads(checksum_path.read_text())
            checksums['manifests/rl_train.json'] = hashlib.sha256(path.read_bytes()).hexdigest()
            checksum_path.write_text(json.dumps(checksums))
            with self.assertRaisesRegex(ValueError, 'metadata differs'):
                check(root)


if __name__ == '__main__':
    unittest.main()
