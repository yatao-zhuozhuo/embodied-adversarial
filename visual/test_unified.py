"""Task browsing, preview attribution and media access checks."""
import json
import tempfile
import unittest
from pathlib import Path

from starlette.testclient import TestClient
from visual.server import Catalog, create_app


class UnifiedViewerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data_unified'
        self.data.mkdir()
        self.task = {'task_key': 'libero/test', 'task_id': 'test', 'bench': 'libero',
                     'role': 'shared', 'env_id': 'openeta/libero_test-v0'}
        self.case = {'case_id': 'libero/test/validation/000', 'task_key': 'libero/test',
                     'phase': 'validation', 'seed': 1, 'runtime_verified': False, 'initial_state': {'kind': 'native_seed'}}
        (self.data / 'task_catalog.json').write_text(json.dumps({'split_revision': 2, 'tasks': [self.task]}))
        self.write_cases()
        (self.data / 'source_refs.jsonl').write_text('')
        self.catalog = Catalog(*(self.root / name for name in ('baseline', 'sft', 'training')))
        self.client = TestClient(create_app(self.catalog, legacy=False, unified_root=self.data))
        self.key = self.client.get('/api/unified/tasks').json()['items'][0]['id']

    def write_cases(self):
        (self.data / 'initial_states.jsonl').write_text(json.dumps(self.case) + '\n')

    def test_native_validation_is_not_mislabeled_as_restored(self):
        row = self.client.get('/api/unified/tasks').json()['items'][0]
        self.assertEqual(row['case_counts'], {'validation': 1})
        detail = self.client.get('/api/unified/tasks/' + self.key).json()
        self.assertEqual(detail['preview']['kind'], 'missing')
        self.assertFalse(detail['selected_case']['runtime_verified'])

    def test_history_preview_explicitly_does_not_claim_selected_case(self):
        frame = self.root / 'sft/campaign/libero/test/attempt-0000/frames/0000-view.png'
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b'fixture')
        detail = self.client.get('/api/unified/tasks/' + self.key).json()
        self.assertEqual(detail['preview']['kind'], 'missing')
        self.assertEqual(detail['preview']['frames'], [])
        preview = detail['example_preview']
        self.assertEqual(preview['kind'], 'historical_example')
        self.assertIn('不代表所选初态', preview['label'])
        self.assertEqual(self.client.get('/' + preview['frames'][0]['url']).content, b'fixture')

    def test_switching_seed_never_promotes_same_history_image_to_case_preview(self):
        frame = self.root / 'sft/campaign/libero/test/attempt-0000/frames/0000-view.png'
        frame.parent.mkdir(parents=True)
        frame.write_bytes(b'fixture')
        second = {**self.case, 'case_id': 'libero/test/validation/001', 'seed': 2}
        first = {**self.case, 'seed': 1}
        (self.data / 'initial_states.jsonl').write_text(json.dumps(first) + '\n' + json.dumps(second) + '\n')
        a = self.client.get('/api/unified/tasks/' + self.key).json()
        b = self.client.get('/api/unified/tasks/' + self.key, params={'case': second['case_id']}).json()
        self.assertNotEqual(a['selected_case']['seed'], b['selected_case']['seed'])
        self.assertEqual(a['preview']['frames'], [])
        self.assertEqual(b['preview']['frames'], [])
        self.assertEqual(a['example_preview'], b['example_preview'])

    def test_reference_preview_uses_selected_demo_without_exposing_actions(self):
        self.case['initial_state'] = {'kind': 'reference_demo', 'source_ref_id': 'ref-7', 'reference_demo_index': 7}
        self.write_cases()
        (self.data / 'source_refs.jsonl').write_text(json.dumps({'source_ref_id': 'ref-7',
            'dataset_uri': '/not-installed/demo.hdf5', 'reference_demo_index': 7, 'kind': 'libero_demo_state'}) + '\n')
        result = self.client.get('/api/unified/tasks/' + self.key).json()
        self.assertEqual(result['preview']['demo_index'], 7)
        self.assertEqual(len(result['preview']['frames']), 2)
        self.assertNotIn('actions', result)

    def test_real_capture_is_bound_to_case_and_replaces_history(self):
        import hashlib
        first = {**self.case, 'seed': 1}
        second = {**self.case, 'case_id': 'libero/test/validation/001', 'seed': 2}
        (self.data / 'initial_states.jsonl').write_text(json.dumps(first)+'\n'+json.dumps(second)+'\n')
        captures = {}
        for i, case in enumerate([first, second]):
            frame = self.data / f'runtime_checks/libero/test/{i}.png'
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(f'captured-{i}'.encode())
            captures[case['case_id']] = {'capture_success': True,
                'seed': case['seed'], 'initial_state_selector': case['initial_state'],
                'state_fingerprint': f'physical-state-{i}', 'same_instance_reproducible': True,
                'frames': [{'path': str(frame.relative_to(self.data)), 'camera': 'actual-view',
                            'sha256': hashlib.sha256(frame.read_bytes()).hexdigest()}]}
        (self.data / 'runtime_checks/index.json').write_text(json.dumps({'cases': captures, 'tasks': []}))
        a = self.client.get('/api/unified/tasks/'+self.key).json()
        b = self.client.get('/api/unified/tasks/'+self.key,params={'case':second['case_id']}).json()
        self.assertEqual(a['preview']['kind'], 'actual_reset')
        self.assertEqual(b['preview']['case_id'], second['case_id'])
        self.assertNotEqual(a['preview']['frames'][0]['url'],b['preview']['frames'][0]['url'])
        self.assertEqual(self.client.get('/'+a['preview']['frames'][0]['url']).content,b'captured-0')
        self.assertEqual(self.client.get('/'+b['preview']['frames'][0]['url']).content,b'captured-1')
        self.assertIsNone(a['example_preview'])

    def test_old_capture_cannot_follow_reassigned_seed(self):
        (self.data / 'runtime_checks').mkdir()
        (self.data / 'runtime_checks/index.json').write_text(json.dumps({'cases': {
            self.case['case_id']: {'seed': 99, 'initial_state_selector': self.case['initial_state'],
                                  'capture_success': True, 'frames': []}}, 'tasks': []}))
        detail = self.client.get('/api/unified/tasks/'+self.key).json()
        self.assertEqual(detail['preview']['kind'], 'missing')
        self.assertEqual(detail['runtime_check'], {})

    def test_case_and_media_paths_cannot_be_injected(self):
        self.assertEqual(self.client.get('/api/unified/tasks/' + self.key + '?case=other/task').status_code, 404)
        self.assertEqual(self.client.get('/api/unified/media/not-registered').status_code, 404)
        frame = self.root / 'sft/campaign/libero/test/attempt-0000/frames/0000-view.png'
        frame.parent.mkdir(parents=True)
        outside = self.root / 'private.png'
        outside.write_bytes(b'private')
        frame.symlink_to(outside)
        result = self.client.get('/api/unified/tasks/' + self.key).json()
        self.assertEqual(result['preview']['frames'], [])


if __name__ == '__main__':
    unittest.main()
