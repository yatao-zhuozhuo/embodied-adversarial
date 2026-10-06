import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('watchdog', Path(__file__).with_name('watchdog.py'))
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class RecoveryTest(unittest.TestCase):
    def test_preserves_task_results_and_archives_only_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            formal = root / 'formal'
            monitor = root / 'monitor'
            manifests = root / 'baseline/manifests'
            manifests.mkdir(parents=True)
            rows = [{'episode_id': str(i)} for i in range(3)]
            (manifests / 'all_40.json').write_text(json.dumps({'episodes': rows}))
            for i, status in enumerate(['success', 'turn_limit', 'infrastructure_error']):
                directory = formal / 'model' / str(i)
                directory.mkdir(parents=True)
                (directory / 'final.json').write_text(json.dumps({'status': status, 'success': status == 'success'}))
            with patch.multiple(w, ROOT=root, FORMAL=formal, MONITOR=monitor), patch.object(w, 'processes', return_value=[]), patch.object(w, 'ready', return_value=True), patch.object(w, 'listening', return_value=True), patch.object(w, 'launch', return_value=123) as launch:
                report = w.check_model('model', 0, 8100, 8200)
                self.assertEqual(report['state'], 'restarted')
                self.assertTrue((formal / 'model/0/final.json').exists())
                self.assertTrue((formal / 'model/1/final.json').exists())
                self.assertFalse((formal / 'model/2/final.json').exists())
                self.assertEqual(len(list(monitor.rglob('final.json'))), 1)
                launch.assert_called_once()
                # A fully finished campaign must never be relaunched.
                directory = formal / 'model/2'
                directory.mkdir()
                (directory / 'final.json').write_text(json.dumps({'status': 'turn_limit'}))
                launch.reset_mock()
                self.assertEqual(w.check_model('model', 0, 8100, 8200)['state'], 'complete')
                launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
