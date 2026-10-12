"""Read unified task manifests and serve clearly attributed cached previews."""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections import Counter, defaultdict
from pathlib import Path

from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

PHASES = {'sft_train': 'SFT 训练', 'rl_train': 'RL 训练', 'validation': '验证',
          'eval_seen': '同任务测试', 'eval_generalization': '泛化测试'}


class UnifiedTasks:
    def __init__(self, root, sft, repo, baseline=None):
        self.root, self.sft, self.repo = Path(root), Path(sft), Path(repo)
        self.cache_root = self.repo / 'visual/.runtime/unified-previews'
        self.lock = threading.RLock()
        self.tasks, self.cases, self.refs, self.media = {}, {}, {}, {}
        self.signature = None
        self.baseline = Path(baseline) if baseline else None
        self.runtime_cases = {}
        self.runtime_tasks = {}

    def load(self):
        with self.lock:
            paths = [self.root / name for name in ('task_catalog.json', 'initial_states.jsonl', 'source_refs.jsonl')]
            signature = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
            runtime_path = self.root / 'runtime_checks/index.json'
            signature += ((runtime_path.stat().st_mtime_ns, runtime_path.stat().st_size) if runtime_path.is_file() else None,)
            if signature == self.signature:
                return
            catalog = json.loads(paths[0].read_text())
            cases = defaultdict(list)
            for line in paths[1].read_text().splitlines():
                row = json.loads(line)
                cases[row['task_key']].append(row)
            self.refs = {r['source_ref_id']: r for r in map(json.loads, paths[2].read_text().splitlines())}
            self.tasks = {hashlib.sha256(t['task_key'].encode()).hexdigest()[:24]: t for t in catalog['tasks']}
            self.cases, self.catalog, self.signature = cases, catalog, signature
            runtime = json.loads(runtime_path.read_text()) if runtime_path.is_file() else {}
            # A reused display id must never attach an old seed/demo screenshot.
            selectors = {c['case_id']: c for rows in cases.values() for c in rows}
            self.runtime_cases = {key: value for key, value in runtime.get('cases', {}).items()
                                  if key in selectors and value.get('seed') == selectors[key]['seed']
                                  and value.get('initial_state_selector') == selectors[key]['initial_state']}
            self.runtime_tasks = {t['task_key']:t for t in runtime.get('tasks', [])}

    def listing(self, request):
        try:
            self.load()
            with self.lock:
                rows = []
                for key, task in self.tasks.items():
                    cases = self.cases[task['task_key']]
                    rows.append({**task, 'id': key, 'case_counts': dict(Counter(c['phase'] for c in cases)),
                                 'initial_state_check': self.runtime_tasks.get(task['task_key']),
                                 'captured_cases': sum(bool(self.runtime_cases.get(c['case_id'], {}).get('capture_success')) for c in cases),
                                 'verified_cases': sum(bool(self.runtime_cases.get(c['case_id'], {}).get('same_instance_reproducible')) for c in cases)})
                return JSONResponse({'items': rows, 'split_revision': self.catalog.get('split_revision', 1),
                    'active_benches': self.catalog.get('active_benches', list(dict.fromkeys(t['bench'] for t in self.tasks.values()))),
                    'phase_labels': PHASES, 'root': str(self.root),
                    'case_counts': dict(Counter(c['phase'] for rows in self.cases.values() for c in rows))})
        except (OSError, ValueError, KeyError) as exc:
            return JSONResponse({'error': f'无法读取统一题单：{exc}'}, status_code=503)

    def history_preview(self, task):
        """Use saved same-task frames as examples, never as selected-case evidence."""
        for campaign in sorted(self.sft.iterdir()) if self.sft.is_dir() else []:
            for base in (campaign / 'episodes' / task['bench'] / task['task_id'],
                         campaign / task['bench'] / task['task_id']):
                if not base.is_dir():
                    continue
                for episode in sorted(base.iterdir()):
                    frame_dir = episode / 'frames'
                    if not frame_dir.is_dir():
                        continue
                    paths = sorted(frame_dir.glob('*.png'))
                    if not paths:
                        continue
                    prefix = paths[0].stem.split('-')[0]
                    frames = []
                    for p in [p for p in paths if p.stem.split('-')[0] == prefix][:3]:
                        if not p.resolve().is_relative_to(self.sft.resolve()):
                            continue
                        token = hashlib.sha256(str(p).encode()).hexdigest()
                        self.media[token] = {'path': p, 'root': self.sft.resolve()}
                        frames.append({'camera': p.stem, 'url': f'api/unified/media/{token}'})
                    if frames:
                        return {'kind': 'historical_example', 'label': '历史同题示例 · 不代表所选初态',
                                'source': str(episode), 'frames': frames}
        if self.baseline:
            v2 = self.baseline.parent / 'v2'
            manifest = v2 / 'manifests/all.json'
            if manifest.is_file():
                def identity(env_id):
                    return env_id.replace('robocasa_pretrain_', 'robocasa_').replace('robocasa_target_', 'robocasa_')
                entries = json.loads(manifest.read_text()).get('episodes', [])
                for ep in entries:
                    if identity(ep['env_id']) != identity(task['env_id']):
                        continue
                    for campaign in sorted((v2 / 'campaigns').glob('*')):
                        for phase in ('formal', 'canary'):
                            for images in sorted((campaign / phase).glob(f"*/{ep['episode_id']}/attempt-*/images")):
                                frames = []
                                for p in sorted(images.glob('*.png'))[:3]:
                                    root = v2.resolve()
                                    if not p.resolve().is_relative_to(root):
                                        continue
                                    token = hashlib.sha256(str(p).encode()).hexdigest()
                                    self.media[token] = {'path': p, 'root': root}
                                    frames.append({'camera': p.stem, 'url': f'api/unified/media/{token}'})
                                if frames:
                                    return {'kind': 'historical_example',
                                            'label': '历史 baseline 同题示例 · 不代表所选初态',
                                            'source': str(images.parent), 'example_env_id': ep['env_id'],
                                            'example_seed': ep['seed'], 'frames': frames}
        return {'kind': 'missing', 'label': '此初态尚无可用图像；未启动仿真器', 'frames': []}

    def preview(self, case, task):
        observed = self.runtime_cases.get(case['case_id'])
        if observed and observed.get('capture_success'):
            frames = []
            for frame in observed['frames']:
                p = (self.root / frame['path']).resolve()
                roots = [(self.root / name).resolve() for name in ('runtime_checks', 'diversity_candidates')]
                media_root = next((root for root in roots if p.is_relative_to(root)), None)
                if media_root is None or p.suffix.lower() != '.png':
                    continue
                token = hashlib.sha256((str(p) + frame['sha256']).encode()).hexdigest()
                self.media[token] = {'path': p, 'root': media_root}
                frames.append({'camera': frame['camera'], 'url': f'api/unified/media/{token}'})
            if frames:
                return {'kind': 'actual_reset', 'case_id': case['case_id'],
                        'label': '所选初态的实际重置 / 状态恢复截图',
                        'source': str(media_root), 'frames': frames,
                        'state_fingerprint': observed['state_fingerprint']}
        reference = self.refs.get(case['initial_state'].get('source_ref_id'))
        if reference and reference['kind'] in ('libero_demo_state', 'robocasa_demo_state'):
            cameras = ('agentview', 'wrist') if task['bench'] == 'libero' else ('agentview_left', 'agentview_right', 'wrist')
            frames = []
            for camera in cameras:
                token = hashlib.sha256(f"{reference['dataset_uri']}:{reference['reference_demo_index']}:{camera}".encode()).hexdigest()
                self.media[token] = {'reference': reference, 'camera': camera}
                frames.append({'camera': camera, 'url': f'api/unified/media/{token}'})
            return {'kind': 'cached_demo', 'label': '所选缓存示范的首帧 · 尚未核验状态恢复',
                    'source': reference['dataset_uri'], 'demo_index': reference['reference_demo_index'], 'frames': frames}
        return {'kind': 'missing', 'case_id': case['case_id'],
                'label': '所选初态尚未生成真实截图，不能比较 seed 的场景差异', 'frames': []}

    def detail(self, request):
        try:
            self.load()
            with self.lock:
                task = self.tasks.get(request.path_params['key'])
                if not task:
                    return JSONResponse({'error': 'Unknown task'}, status_code=404)
                cases = self.cases[task['task_key']]
                wanted = request.query_params.get('case')
                case = next((c for c in cases if c['case_id'] == wanted), None) if wanted else cases[0]
                if case is None:
                    return JSONResponse({'error': 'Case does not belong to this task'}, status_code=404)
                preview = self.preview(case, task)
                example = self.history_preview(task) if preview['kind'] == 'missing' else None
                case_options = [{**c, 'runtime_capture_available': bool(self.runtime_cases.get(c['case_id'], {}).get('capture_success'))}
                                for c in cases]
                return JSONResponse({'task': task, 'cases': case_options, 'selected_case': case,
                                     'runtime_check': {k:v for k,v in self.runtime_cases.get(case['case_id'], {}).items()
                                                       if k not in ('physical_state', 'traceback')},
                                     'initial_state_check': self.runtime_tasks.get(task['task_key']),
                                     'preview': preview,
                                     'example_preview': example if example and example['frames'] else None})
        except (OSError, ValueError, KeyError) as exc:
            return JSONResponse({'error': str(exc)}, status_code=503)

    def image(self, request):
        entry = self.media.get(request.path_params['key'])
        if not entry:
            return JSONResponse({'error': 'Unknown preview'}, status_code=404)
        if 'path' in entry:
            p = entry['path'].resolve()
            if p.is_relative_to(entry['root']) and p.is_file():
                return FileResponse(p)
            return JSONResponse({'error': 'Preview unavailable'}, status_code=404)
        output = self.cache_root / (request.path_params['key'] + '.png')
        try:
            with self.lock:
                if not output.is_file():
                    self.cache_root.mkdir(parents=True, exist_ok=True)
                    ref = entry['reference']
                    python = self.repo / 'sim/venvs/libero/bin/python' if ref['kind'] == 'libero_demo_state' else self.repo / '.venv/bin/python'
                    subprocess.run([str(python), str(Path(__file__).with_name('unified_media.py')),
                        '--dataset', ref['dataset_uri'], '--index', str(ref['reference_demo_index']),
                        '--camera', entry['camera'], '--kind', ref['kind'], '--output', str(output)],
                        cwd=self.repo, capture_output=True, check=True, timeout=15)
            return FileResponse(output, media_type='image/png')
        except (OSError, subprocess.SubprocessError):
            return JSONResponse({'error': '缓存示范图像不可用'}, status_code=404)

    def routes(self):
        return [Route('/api/unified/tasks', self.listing), Route('/api/unified/tasks/{key}', self.detail),
                Route('/api/unified/media/{key}', self.image)]
