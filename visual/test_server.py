"""Artifact adapter and media-boundary regression checks (no simulator required)."""
import json
import tempfile
import threading
from unittest.mock import patch
import unittest
from pathlib import Path

from starlette.testclient import TestClient
from visual.server import Catalog, create_app


class ViewerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.catalog = Catalog(*(self.root / name for name in ('baseline', 'sft', 'training')))
        self.client = TestClient(create_app(self.catalog, legacy=False))

    def write(self, relative, record):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record))
        return path

    def test_baseline_deduplicates_and_preserves_verdict(self):
        data = {'episode': {'task': 'drawer', 'metadata': {'benchmark': 'metaworld'}},
                'success': False, 'native_success_observed': True,
                'turns': [{'content': 'action', 'native_reward': 0.5,
                           'details': {'vision_attachments': [{'path': 'images/000-0.png'}]}}],
                'final_observation': {'vision_image_paths': ['final-images/000-0.png']}}
        self.write('baseline/run/model/task/attempt-1/progress.json', data)
        self.write('baseline/run/model/task/attempt-1/result.json', data)
        self.write('baseline/run/model/task/final.json', data)
        rows, errors = self.catalog.scan('baseline')
        self.assertEqual(len(rows), 1)
        self.assertEqual(errors, [])
        record = self.catalog.records[rows[0]['id']]
        self.assertIs(record['success'], False)
        self.assertTrue(record['metadata']['native_success_observed'])
        self.assertNotEqual(record['steps'][0]['before_frames'], record['steps'][0]['frames'])

    def test_sft_reasoning_refresh_and_partial_write(self):
        data = {'bench': 'libero', 'success': True, 'turns': [{'model_message': {'reasoning': 'think', 'content': 'act'}, 'frames': ['frames/0001-wrist.png'], 'reward': 0}]}
        path = self.write('sft/qwen_live_v1/libero/task/attempt-0001/episode.json', data)
        rows, _ = self.catalog.scan('sft')
        record = self.catalog.records[rows[0]['id']]
        self.assertEqual(record['steps'][0]['text'], 'think\n\nact')
        self.assertEqual(record['steps'][0]['reward'], 0)
        data['success'] = False
        path.write_text(json.dumps(data))
        rows, _ = self.catalog.scan('sft')
        self.assertFalse(rows[0]['success'])
        path.write_text('{')
        rows, errors = self.catalog.scan('sft')
        self.assertFalse(rows)
        self.assertEqual(len(errors), 1)

    def test_unified_model_inputs_are_separate_from_host_frames(self):
        root = self.root / 'sft/deepseek_input_v1_nothink_test/libero/task/attempt-0000'
        root.mkdir(parents=True)
        image = root / 'model.png'
        image.write_bytes(b'image-fixture')
        package = {'images': [{'path': str(image), 'image_id': 'image-0', 'camera_id': 'wrist'}],
                   'dynamic': {'execution': 'executed'}}
        self.write('sft/deepseek_input_v1_nothink_test/libero/task/attempt-0000/episode.json',
                   {'bench': 'libero', 'turns': [{'model_call_index': 0, 'frames': ['host.png'], 'reward': 1}],
                    'model_calls': [{'input_package': package}]})
        rows, errors = self.catalog.scan('sft')
        self.assertFalse(errors)
        step = self.catalog.records[rows[0]['id']]['steps'][0]
        self.assertEqual(step['model_input'], package)
        self.assertEqual(len(step['input_frames']), 1)
        self.assertNotEqual(step['input_frames'], step['frames'])

    def test_xml_audit_visible_and_live_episode_takes_priority(self):
        data = {'bench': 'libero', 'model_protocol': 'openeta.two_tool_xml.v1',
                'success': False, 'images_retained': False, 'turns': []}
        self.write('sft/qwen_xml_v1/audit/libero/task/attempt-0001.json', data)
        rows, errors = self.catalog.scan('sft')
        self.assertFalse(errors)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['protocol'], 'openeta.two_tool_xml.v1')
        self.assertFalse(rows[0]['images_retained'])
        self.assertIn('attempt-0001', rows[0]['label'])
        self.write('sft/qwen_xml_v1/libero/task/attempt-0001/episode.json', data)
        rows, _ = self.catalog.scan('sft')
        self.assertEqual(len(rows), 1)
        self.assertNotIn('audit', rows[0]['group'])

    def test_background_listing_returns_while_scan_is_blocked(self):
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        def slow_scan(source):
            entered.set()
            release.wait(5)
            done.set()
            return [], []
        with patch.object(self.catalog, 'scan', side_effect=slow_scan) as scan:
            try:
                response = self.client.get('/api/episodes?source=baseline&background=1')
                self.assertTrue(response.json()['loading'])
                self.assertTrue(entered.wait(2))
                poll = self.client.get('/api/episodes?source=baseline&background=1&poll=1')
                self.assertTrue(poll.json()['loading'])
                self.client.get('/api/episodes?source=baseline&background=1')
                self.assertEqual(scan.call_count, 1)
            finally:
                release.set()
                done.wait(2)

    def test_expert_annotation_is_attributed_and_refreshes(self):
        import hashlib
        xml = '<decision><kind>tool_call</kind><name>step_env</name><reasoning /><parameters><num_steps>1</num_steps></parameters></decision>'
        episode = {'bench': 'libero', 'task_id': 'task', 'success': True,
                   'model_protocol': 'openeta.two_tool_xml.v1', 'turns': [
                       {'turn': 1, 'model_message': {'content': xml}, 'official_success': True}]}
        self.write('sft/expert_replay_v1/episodes/libero/task/seed-1/episode.json', episode)
        sample = {'metadata': {'source_turn': 1}, 'messages': [
            {'role': 'user', 'content': 'observation'}, {'role': 'assistant', 'content': xml}], 'images': []}
        samples = self.root / 'sft/expert_replay_v1/episodes/libero/task/seed-1/samples.jsonl'
        samples.write_text(json.dumps(sample) + '\n')
        rows, errors = self.catalog.scan('expert')
        self.assertFalse(errors)
        self.assertEqual(rows[0]['annotated_turns'], 0)
        annotation = {'reasoning': 'Approach the object.', 'reasoning_model': 'Gemini',
                      'source_sample_sha256': hashlib.sha256(json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        self.write('sft/expert_replay_v1/reasoning_preview_gemini_flash/episodes/libero/task/seed-1/annotations/turn-0001.json', annotation)
        rows, errors = self.catalog.scan('expert')
        self.assertEqual(rows[0]['annotated_turns'], 1)
        record = self.catalog.records[rows[0]['id']]
        self.assertEqual(record['steps'][0]['text'], xml)
        self.assertIn('Approach the object.', record['steps'][0]['annotated_xml'])
        annotation['source_sample_sha256'] = 'wrong'
        self.write('sft/expert_replay_v1/reasoning_preview_gemini_flash/episodes/libero/task/seed-1/annotations/turn-0001.json', annotation)
        rows, _ = self.catalog.scan('expert')
        self.assertEqual(rows[0]['annotated_turns'], 0)

    def test_compressed_expert_uses_matching_full_annotations(self):
        import hashlib
        xml = '<decision><kind>tool_call</kind><name>step_env</name><reasoning /><parameters><num_steps>1</num_steps></parameters></decision>'
        episode = {'bench': 'libero', 'task_id': 'task', 'success': True,
                   'turns': [{'turn': 1, 'model_message': {'content': xml}}]}
        suffix = 'episodes/libero/task/seed-1/'
        sample = {'metadata': {'source_turn': 1}, 'messages': [{'role': 'assistant', 'content': xml}], 'images': []}
        for dataset in ['expert_replay_v1', 'expert_replay_compressed_v1']:
            path = self.write('sft/' + dataset + '/' + suffix + 'episode.json', episode)
            path.with_name('samples.jsonl').write_text(json.dumps(sample) + '\n')
        annotation = {'reasoning': 'Move toward the target.', 'reasoning_model': 'gemini-3.6-flash',
                      'source_sample_sha256': hashlib.sha256(json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()).hexdigest()}
        annotation_path = self.write('sft/expert_replay_compressed_reasoning_v1/' + suffix + 'annotations/turn-0001.json', annotation)
        rows, errors = self.catalog.scan('expert')
        self.assertFalse(errors)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['group'], 'expert_replay_compressed_v1')
        self.assertEqual(rows[0]['annotated_turns'], 1)
        record = self.catalog.records[rows[0]['id']]
        self.assertEqual(record['steps'][0]['text'], xml)
        self.assertIn(annotation['reasoning'], record['steps'][0]['annotated_xml'])
        annotation['source_sample_sha256'] = 'wrong'
        annotation_path.write_text(json.dumps(annotation))
        rows, errors = self.catalog.scan('expert')
        self.assertEqual(rows[0]['annotated_turns'], 0)

    def test_slow_baseline_does_not_block_expert_scan(self):
        entered, release, expert_done = threading.Event(), threading.Event(), threading.Event()
        self.write('baseline/run/result.json', {'episode': {}})
        original = self.catalog.normalize
        def slow_normalize(source, path, data):
            if source == 'baseline':
                entered.set()
                release.wait(5)
            return original(source, path, data)
        with patch.object(self.catalog, 'normalize', side_effect=slow_normalize):
            worker = threading.Thread(target=lambda: self.catalog.scan('baseline'))
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                expert = threading.Thread(target=lambda: (self.catalog.scan('expert'), expert_done.set()))
                expert.start()
                self.assertTrue(expert_done.wait(2), 'Expert scan blocked by baseline')
            finally:
                release.set()
                worker.join(2)
                expert.join(2)

    def test_training_alignment_and_media_boundary(self):
        self.write('training/round-0000/alice_rollout/rollouts/alice/one.json',
                   {'role': 'alice', 'image_hashes': ['initial', 'after'],
                    'steps': [{'completion': 'move', 'environment_reward': 0.2}],
                    'final_info': {'reward': -0.5, 'success': False}})
        rows, _ = self.catalog.scan('training')
        record = self.catalog.records[rows[0]['id']]
        self.assertEqual(record['reward'], -0.5)
        self.assertEqual(record['steps'][0]['reward'], 0.2)
        initial = record['initial_frames'][0]
        self.assertEqual(initial, record['steps'][0]['before_frames'][0])
        self.assertNotEqual(initial['url'], record['steps'][0]['frames'][0]['url'])
        outside = self.root / 'secret.png'
        outside.write_bytes(b'secret')
        inside = self.catalog.roots['training'] / 'link.png'
        inside.symlink_to(outside)
        self.assertIsNone(self.catalog.frame(inside, self.catalog.roots['training'], 'x'))
        self.assertEqual(self.client.get('/api/media/unknown').status_code, 404)
        self.assertEqual(self.client.get('/api/episodes?source=bad').status_code, 400)
        self.assertEqual(self.client.get('/api/episode/' + record['id']).status_code, 200)

    def test_xml_sft_protocol_and_raw_output_are_visible(self):
        data = {'bench': 'metaworld', 'model_protocol': 'openeta.two_tool_xml.v1',
                'success': False, 'output_source': 'original_model_xml',
                'turns': [{'model_message': {'reasoning_content': 'thinking', 'content': '<decision/>'},
                           'validation': {'valid': False, 'repaired': False}}]}
        self.write('sft/qwen_xml_v1/metaworld/task/attempt-0000/episode.json', data)
        rows, errors = self.catalog.scan('sft')
        self.assertFalse(errors)
        self.assertEqual(len(rows), 1)
        record = self.catalog.records[rows[0]['id']]
        self.assertEqual(record['group'], 'qwen_xml_v1')
        self.assertEqual(record['steps'][0]['text'], 'thinking\n\n<decision/>')
        self.assertFalse(record['steps'][0]['raw']['validation']['valid'])

    def test_deepseek_sft_is_not_labeled_as_qwen(self):
        data = {'bench': 'metaworld', 'model': 'deepseek-flash', 'provider': 'deepseek',
                'model_protocol': 'openeta.two_tool_xml.v1', 'success': False,
                'turns': [{'model_message': {'reasoning_content': 'live reasoning', 'content': '<decision/>'}}]}
        self.write('sft/deepseek_xml_v1/metaworld/task/attempt-0000/episode.json', data)
        rows, errors = self.catalog.scan('sft')
        self.assertFalse(errors)
        self.assertEqual(len(rows), 1)
        record = self.catalog.records[rows[0]['id']]
        self.assertEqual(record['group'], 'deepseek_xml_v1')
        self.assertEqual(record['model'], 'deepseek-flash')
        self.assertEqual(record['metadata']['provider'], 'deepseek')


    def test_debug_batch_failed_episode_keeps_visible_frames(self):
        directory = 'sft/deepseek_xml_debug5_20261006_000000/metaworld/task/attempt-0000'
        self.write(directory + '/episode.json', {
            'bench': 'metaworld', 'model': 'deepseek-flash', 'success': False,
            'stop_reason': 'turn_limit', 'images_retained': True,
            'initial_frames': ['frames/0000-view.png'],
            'turns': [{'frames': ['frames/0001-view.png'], 'model_message': {'content': '<decision/>'}}]})
        frames = self.root / directory / 'frames'
        frames.mkdir()
        (frames / '0000-view.png').write_bytes(b'initial-image')
        (frames / '0001-view.png').write_bytes(b'after-image')
        rows, errors = self.catalog.scan('sft')
        self.assertFalse(errors)
        self.assertEqual(len(rows), 1)
        record = self.catalog.records[rows[0]['id']]
        self.assertFalse(record['success'])
        self.assertEqual(record['status'], 'turn_limit')
        self.assertFalse(record['initial_frames'][0]['missing'])
        self.assertFalse(record['steps'][0]['frames'][0]['missing'])


if __name__ == '__main__':
    unittest.main()
