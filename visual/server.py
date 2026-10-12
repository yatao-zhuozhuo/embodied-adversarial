#!/usr/bin/env python3
"""Unified, local trajectory viewer. Reads artifacts without starting simulators."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
from pathlib import Path

from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

REPO = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent


def clean(value):
    """Keep evidence, omit repeated request histories and binary image payloads."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if k not in {'exchange', 'model_calls'} and not k.endswith('base64')}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, str) and value.startswith('data:image/'):
        return '[image payload omitted]'
    return value


def branch(path):
    try:
        return subprocess.check_output(['git', '-C', str(path), 'branch', '--show-current'],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def baseline_default():
    try:
        output = subprocess.check_output(['git', '-C', str(REPO), 'worktree', 'list', '--porcelain'], text=True)
        for block in output.split('\n\n'):
            if 'branch refs/heads/feat/baseline' in block:
                return Path(block.splitlines()[0].removeprefix('worktree ')) / 'baseline/results'
    except (OSError, subprocess.CalledProcessError):
        pass
    return REPO / '.worktrees/baseline/baseline/results'


class Catalog:
    def __init__(self, baseline, sft, training):
        self.roots = {'baseline': Path(baseline).resolve(), 'sft': Path(sft).resolve(),
                      'training': Path(training).resolve(), 'expert': Path(sft).resolve(),
                      'expert100': Path(sft).resolve()}
        self.baseline_roots = [self.roots['baseline']]
        v2 = self.roots['baseline'].parent / 'v2/campaigns'
        if self.roots['baseline'].name == 'results' and v2.is_dir():
            self.baseline_roots.append(v2.resolve())
        self.cache = {}
        self.records = {}
        self.media = {}
        self.lock = threading.RLock()
        self.errors = []
        self.jobs_lock = threading.Lock()
        self.jobs = {}
        self.scan_locks = {source: threading.Lock() for source in self.roots}

    def background_scan(self, source):
        """Return immediately so proxy requests never wait for large artifacts."""
        with self.jobs_lock:
            job = self.jobs.get(source)
            if job and job['loading']:
                return dict(job)
            self.jobs[source] = {'loading': True, 'processed': 0, 'total': 0,
                                 'items': (job or {}).get('items', []), 'errors': []}
        def run():
            try:
                rows, errors = self.scan(source)
                with self.jobs_lock:
                    self.jobs[source].update(loading=False, items=rows, errors=errors)
            except Exception as exc:
                with self.jobs_lock:
                    self.jobs[source].update(loading=False, error=str(exc))
        threading.Thread(target=run, daemon=True).start()
        with self.jobs_lock:
            return dict(self.jobs[source])

    def scan_status(self, source):
        with self.jobs_lock:
            return dict(self.jobs.get(source, {'loading': False, 'items': [], 'errors': []}))

    def frame(self, path, root, camera):
        path = path.resolve()
        if not path.is_relative_to(root) or path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            return None
        key = hashlib.sha256(str(path).encode()).hexdigest()
        self.media[key] = (path, root)
        return {'camera': camera, 'url': 'api/media/' + key, 'missing': not path.is_file()}

    def normalize(self, source, path, data):
        root = self.roots[source]
        if source == 'baseline':
            root = next(r for r in self.baseline_roots if path.resolve().is_relative_to(r))
        rel = path.relative_to(root)
        identity_rel = str(root) + '/' + str(rel.parent) if source == 'baseline' else str(rel)
        key = hashlib.sha256((source + '/' + identity_rel).encode()).hexdigest()[:24]
        steps = []
        def frames(paths):
            result = []
            for i, name in enumerate(paths):
                p = Path(name)
                p = p if p.is_absolute() else path.parent / p
                camera = p.stem.split('-', 1)[-1] if source == 'sft' else f'camera-{i}'
                f = self.frame(p, root, camera)
                if f:
                    result.append(f)
            return result
        if source in {'expert', 'expert100'}:
            return self.normalize_expert(path, data, source)
        if source == 'baseline':
            ep = data['episode']
            raw_steps = data.get('turns', [])
            def before(s):
                observation = s.get('observation_before') or {}
                if 'vision_image_paths' in observation:
                    return frames(observation['vision_image_paths'])
                return frames([v['path'] for v in s.get('details', {}).get('vision_attachments', [])
                               if v.get('path') and v.get('attached', True)])
            initial = before(raw_steps[0]) if raw_steps else []
            final = frames(data.get('final_observation', {}).get('vision_image_paths', []))
            for i, s in enumerate(raw_steps):
                steps.append({'frames': frames(s['observation_after']['vision_image_paths']) if 'vision_image_paths' in (s.get('observation_after') or {}) else (before(raw_steps[i + 1]) if i + 1 < len(raw_steps) else final),
                              'before_frames': before(s), 'reward': s.get('native_reward'),
                              'success': s.get('native_success', {}).get('success'),
                              'text': s.get('content', ''), 'raw': clean(s)})
            bench = ep.get('metadata', {}).get('benchmark', 'unknown')
            task = ep.get('task', ep.get('episode_id'))
            model = data.get('model', '')
            group = ('v2/' if root != self.roots['baseline'] else '') + str(rel.parents[3]) if len(rel.parents) > 3 else str(rel.parent)
            label = f"{ep.get('episode_id', '')} / attempt-{data.get('attempt', '?')}"
            success, status = data.get('success'), data.get('status', 'unknown')
            reward = None
            metadata = {k: clean(v) for k, v in data.items() if k != 'turns'}
        elif source == 'sft':
            initial = frames(data.get('initial_frames', []))
            previous = initial
            for s in data.get('turns', []):
                current = frames(s.get('frames', []))
                message = s.get('model_message') or {}
                call_index = s.get('model_call_index', -1)
                calls = data.get('model_calls', [])
                package = calls[call_index].get('input_package') if 0 <= call_index < len(calls) else None
                input_frames = []
                if package:
                    for evidence in package.get('images', []):
                        frame = self.frame(Path(evidence['path']), root, evidence['image_id'] + ' / ' + evidence['camera_id'])
                        if frame:
                            input_frames.append(frame)
                text = '\n\n'.join(str(message[k]) for k in ('reasoning_content', 'reasoning', 'content') if message.get(k))
                steps.append({'frames': current, 'before_frames': previous, 'reward': s.get('reward'),
                              'success': s.get('official_success'), 'text': text, 'raw': clean(s),
                              'input_frames': input_frames, 'model_input': clean(package)})
                previous = current
            bench, task, model = data.get('bench', 'unknown'), data.get('task'), data.get('model', '')
            group = rel.parts[0] + (' / audit' if 'audit' in rel.parts else '')
            label = str(rel.with_suffix('')) if 'audit' in rel.parts else str(rel.parent)
            success, status, reward = data.get('success'), data.get('stop_reason', 'unknown'), None
            metadata = {k: clean(v) for k, v in data.items() if k not in {'turns', 'model_calls'}}
        else:
            info = data.get('final_info') or {}
            hashes = data.get('image_hashes', [])
            def at(i):
                if i >= len(hashes):
                    return []
                f = self.frame(path.parent.parent / 'frames' / (hashes[i] + '.png'), root, 'policy-view')
                return [f] if f else []
            initial = at(0)
            for i, s in enumerate(data.get('steps', [])):
                steps.append({'frames': at(i + 1), 'before_frames': at(i),
                              'reward': s.get('environment_reward'), 'success': s.get('goal_success'),
                              'text': s.get('completion', ''), 'raw': clean(s)})
            snapshot = data.get('snapshot') or {}
            bench = snapshot.get('metadata', {}).get('backend', 'unknown')
            task, model = data.get('instruction', ''), info.get('adapter_name', data.get('role', ''))
            group = '/'.join(rel.parts[:-3]) or root.name
            label = f"{data.get('role', '?')} / {data.get('request_id', path.stem)}"
            success, status, reward = info.get('success'), info.get('termination_reason', 'unknown'), info.get('reward')
            metadata = {k: clean(v) for k, v in data.items() if k not in {'steps', 'image_hashes'}}
        return {'id': key, 'source': source, 'bench': bench, 'task': task, 'model': model,
                'group': group, 'label': label, 'success': success, 'status': status,
                'reward': reward, 'turns': len(steps), 'modified': path.stat().st_mtime,
                'protocol': data.get('model_protocol') or data.get('protocol_version') or data.get('protocol', 'unknown'),
                'images_retained': data.get('images_retained'),
                'path': str(path), 'initial_frames': initial, 'steps': steps, 'metadata': metadata}

    def expert_annotation_dirs(self, path):
        root = self.roots['expert']
        rel = path.relative_to(root)
        suffix = '/'.join(rel.parts[-4:-1])
        if rel.parts[0].startswith('expert_replay_unified_'):
            return [root / name / 'episodes' / suffix for name in (
                'expert_replay_unified_v6_compressed_reasoning_purpose_v2_20261011',
                'expert_replay_unified_v6_compressed_reasoning_20261011')]
        if rel.parts[0] == 'expert_replay_compressed_v1':
            return [root / 'expert_replay_compressed_reasoning_v1' / 'episodes' / suffix]
        return list((root / 'expert_replay_v1').glob('reasoning*/episodes/' + suffix))

    def normalize_expert(self, path, data, source='expert'):
        """Show replay evidence and separately attributed, action-preserving annotations."""
        import xml.etree.ElementTree as ET
        root = self.roots['expert']
        rel = path.relative_to(root)
        key = hashlib.sha256((source + '/' + str(rel)).encode()).hexdigest()[:24]
        frame_groups = {}
        for frame_path in sorted((path.parent / 'frames').glob('*.png')):
            number = frame_path.stem.split('-', 1)[0]
            if number.isdigit():
                frame_groups.setdefault(int(number), []).append(frame_path)
        def frames_at(index):
            frames = []
            for p in frame_groups.get(index, []):
                f = self.frame(p, root, p.stem.split('-', 1)[-1])
                if f:
                    frames.append(f)
            return frames
        samples = {}
        sample_file = path.parent / 'samples.jsonl'
        if sample_file.exists():
            with sample_file.open() as stream:
                for line in stream:
                    if line.strip():
                        sample = json.loads(line)
                        turn = sample.get('metadata', {}).get('source_turn')
                        if turn is not None:
                            samples[turn] = sample
        preview_dirs = self.expert_annotation_dirs(path)
        steps, annotated_count, annotation_errors = [], 0, []
        for index, event in enumerate(data.get('turns', [])):
            turn = event.get('turn', index + 1)
            sample = samples.get(turn)
            original = (event.get('model_message') or {}).get('content', '')
            annotation = None
            for directory in preview_dirs:
                candidate = directory / 'annotations' / f'turn-{turn:04d}.json'
                if candidate.is_file():
                    try:
                        value = json.loads(candidate.read_text())
                        if not sample or value.get('source_sample_sha256') != hashlib.sha256(
                                json.dumps(sample, ensure_ascii=False, sort_keys=True).encode()).hexdigest():
                            annotation_errors.append(f'turn {turn}: annotation source mismatch')
                            continue
                        annotation = value
                        break
                    except (OSError, ValueError):
                        annotation_errors.append(f'turn {turn}: annotation unreadable')
            annotated_xml = None
            if annotation:
                try:
                    node = ET.fromstring(original)
                    node.find('reasoning').text = annotation['reasoning']
                    annotated_xml = ET.tostring(node, encoding='unicode')
                    annotated_count += 1
                except (ET.ParseError, KeyError, AttributeError):
                    annotation_errors.append(f'turn {turn}: cannot apply reasoning')
            input_frames, model_input = [], None
            if sample:
                for name in sample.get('images', []):
                    image_path = Path(name)
                    f = self.frame(image_path, root, image_path.stem.split('-', 1)[-1])
                    if f:
                        input_frames.append(f)
                messages = sample.get('messages', [])
                model_input = {'messages': messages[:-1], 'metadata': sample.get('metadata')}
            steps.append({'frames': frames_at(index + 1), 'before_frames': frames_at(index),
                          'input_frames': input_frames, 'model_input': clean(model_input),
                          'reward': event.get('reward'), 'success': event.get('official_success'),
                          'text': original, 'annotated_xml': annotated_xml,
                          'reasoning_annotation': clean(annotation), 'raw': clean(event)})
        return {'id': key, 'source': source, 'bench': data.get('bench', 'unknown'),
                'task': data.get('task') or data.get('task_id'), 'model': 'artificial',
                'group': rel.parts[0], 'label': '/'.join(rel.parts[-4:-1]),
                'success': data.get('success'), 'status': data.get('stop_reason', 'unknown'),
                'reward': None, 'turns': len(steps), 'modified': path.stat().st_mtime,
                'protocol': data.get('model_protocol'), 'path': str(path),
                'initial_frames': frames_at(0), 'steps': steps,
                'annotated_turns': annotated_count,
                'output_label': '专家固定动作 XML（优先显示补写 reasoning 版本）' if annotated_count else '专家构造 XML（artificial）',
                'metadata': {**clean({k: v for k, v in data.items() if k != 'turns'}),
                             'reasoning_annotation_errors': annotation_errors,
                             'reasoning_note': '模型事后补写；动作来自专家回放，不是模型自主决策。'}}

    def scan(self, source):
        with self.scan_locks[source]:
            root = self.roots[source]
            if source == 'baseline':
                # One record per attempt. A completed result supersedes progress.
                paths = set(root.rglob('result.json'))
                paths.update(p for p in root.rglob('progress.json') if not p.with_name('result.json').exists())
                for extra in self.baseline_roots[1:]:
                    for campaign in extra.iterdir():
                        for section in ('formal', 'canary'):
                            directory = campaign / section
                            paths.update(directory.glob('**/attempt-*/result.json'))
                            paths.update(p for p in directory.glob('**/attempt-*/progress.json')
                                         if not p.with_name('result.json').exists())
            elif source == 'expert100':
                dataset = root / 'expert_replay_unified_v6_compressed_20261011'
                manifest = json.loads((dataset / 'snapshot/sft_train.json').read_text())
                paths = []
                for case in manifest['cases']:
                    outcome = dataset / 'results' / (case['case_id'] + '.json')
                    if not outcome.is_file():
                        continue
                    result = json.loads(outcome.read_text())
                    if result.get('success') is not True or not 0 < result.get('compressed_turns', 101) <= 100:
                        continue
                    path = dataset / 'episodes' / case['bench'] / case['task_id'] / f"seed-{case['seed']}" / 'episode.json'
                    data = json.loads(path.read_text())
                    if data.get('success') is True and data.get('unified_case') == case and len(data.get('turns', [])) <= 100:
                        paths.append(path)
            elif source == 'expert':
                selected = {}
                for dataset in ('expert_replay_v1', 'expert_replay_compressed_v1'):
                    for path in root.glob(dataset + '/episodes/*/*/seed-*/episode.json'):
                        selected[tuple(path.parts[-4:])] = path
                paths = selected.values()
            elif source == 'sft':
                paths = list(root.glob('qwen_live_v1/*/*/attempt-*/episode.json'))
                paths += list(root.glob('qwen_xml_v1/*/*/attempt-*/episode.json'))
                paths += list(root.glob('deepseek_xml_v1/*/*/attempt-*/episode.json'))
                paths += list(root.glob('deepseek_xml_debug5_*/*/*/attempt-*/episode.json'))
                paths += list(root.glob('deepseek_input_v1_nothink_*/*/*/attempt-*/episode.json'))
                paths += list(root.glob('gemini_input_v1_*/*/*/attempt-*/episode.json'))
                paths += list(root.glob('codingrouter_input_v1_*/*/*/attempt-*/episode.json'))
                paths += list(root.glob('gemini_flash_expanded_*/*/*/attempt-*/episode.json'))
                paths += [p for p in root.glob('qwen_xml_v1/audit/*/*/attempt-*.json')
                          if not (root / 'qwen_xml_v1' / p.parent.parent.name / p.parent.name / p.stem / 'episode.json').exists()]
            else:
                paths = root.glob('**/rollouts/*/*.json')
            records, errors, active = [], [], set()
            paths = sorted(paths)
            with self.jobs_lock:
                if source in self.jobs:
                    self.jobs[source].update(total=len(paths))
            for index, p in enumerate(paths):
                with self.jobs_lock:
                    if source in self.jobs:
                        self.jobs[source].update(processed=index)
                if not any(p.resolve().is_relative_to(r) for r in (self.baseline_roots if source == 'baseline' else [root])):
                    continue
                active.add(p)
                try:
                    stat = p.stat()
                    signature = (stat.st_mtime_ns, stat.st_size)
                    if source in {'expert', 'expert100'}:
                        companions = [p.parent / 'samples.jsonl']
                        for directory in self.expert_annotation_dirs(p):
                            companions.extend(directory.glob('annotations/turn-*.json'))
                        signature += tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size)
                                           for f in sorted(companions) if f.is_file())
                    cached = self.cache.get(p)
                    if cached and cached[0] == signature:
                        record = cached[1]
                    else:
                        data = json.loads(p.read_text(encoding='utf-8'))
                        record = self.normalize(source, p, data)
                        self.cache[p] = signature, record
                    records.append(record)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append({'path': str(p), 'error': str(exc)})
            with self.lock:
                self.records = {k: v for k, v in self.records.items() if v['source'] != source}
                self.records.update({r['id']: r for r in records})
            for p in list(self.cache):
                if self.cache[p][1]['source'] == source and p not in active:
                    del self.cache[p]
            with self.jobs_lock:
                if source in self.jobs:
                    self.jobs[source].update(processed=len(paths))
            records.sort(key=lambda r: r['modified'], reverse=True)
            return [{k: v for k, v in r.items() if k not in {'steps', 'initial_frames', 'metadata'}} for r in records], errors


def create_app(catalog, legacy=True, unified_root=None):
    try:
        from visual.unified import UnifiedTasks
    except ModuleNotFoundError:
        from unified import UnifiedTasks
    unified = UnifiedTasks(unified_root or REPO / 'data_unified', catalog.roots['sft'], REPO, catalog.roots['baseline'])
    def listing(request):
        source = request.query_params.get('source', 'baseline')
        if source not in catalog.roots:
            return JSONResponse({'error': 'Unknown source'}, status_code=400)
        if request.query_params.get('background') == '1':
            result = (catalog.scan_status(source) if request.query_params.get('poll') == '1'
                      else catalog.background_scan(source))
            return JSONResponse({**result, 'root': str(catalog.roots[source]),
                                 'exists': catalog.roots[source].is_dir()})
        rows, errors = catalog.scan(source)
        return JSONResponse({'items': rows, 'errors': errors, 'root': str(catalog.roots[source]),
                             'exists': catalog.roots[source].is_dir()})

    def episode(request):
        with catalog.lock:
            record = catalog.records.get(request.path_params['key'])
        return JSONResponse(record if record else {'error': 'Refresh the listing first'}, status_code=200 if record else 404)

    def media(request):
        entry = catalog.media.get(request.path_params['key'])
        if entry:
            path, root = entry
            if path.resolve().is_relative_to(root) and path.is_file():
                return FileResponse(path)
        return JSONResponse({'error': 'Frame missing'}, status_code=404)

    routes = [Route('/api/episodes', listing), Route('/api/episode/{key}', episode),
              Route('/api/media/{key}', media),
              Route('/api/context', lambda _: JSONResponse({'branch': branch(REPO),
                    'baseline_branch': branch(catalog.roots['baseline']),
                    'roots': {k: str(v) for k, v in catalog.roots.items()}}))]
    routes.extend(unified.routes())
    if legacy:
        directory = REPO / 'SFT_data/fronted'
        sys.path.insert(0, str(directory))
        spec = importlib.util.spec_from_file_location('openeta_legacy_viewer', directory / 'server.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.ROOT = catalog.roots['sft']
        module.DB_PATH = module.ROOT / 'viewer/index.sqlite'
        module.THUMB_DIR = module.ROOT / 'viewer/thumbs'
        if not module.DB_PATH.exists():
            module.indexer.build_index(module.ROOT, module.DB_PATH)
        routes.append(Mount('/sft-history', module.app))
    routes.append(Mount('/', StaticFiles(directory=HERE / 'static', html=True)))
    return Starlette(routes=routes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8780)
    parser.add_argument('--baseline-root', type=Path, default=baseline_default())
    parser.add_argument('--sft-root', type=Path, default=REPO / 'SFT_data')
    parser.add_argument('--training-root', type=Path, default=REPO / 'runs/formal_experiment2')
    parser.add_argument('--unified-root', type=Path, default=REPO / 'data_unified')
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(Catalog(args.baseline_root, args.sft_root, args.training_root), unified_root=args.unified_root),
                host=args.host, port=args.port)


if __name__ == '__main__':
    main()
