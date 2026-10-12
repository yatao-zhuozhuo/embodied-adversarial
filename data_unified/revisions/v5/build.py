#!/usr/bin/env python3
"""Build reference-only task/state splits. Never starts a simulator or training."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BENCHES = ('libero', 'metaworld', 'maniskill', 'robocasa')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(env_id):
    for bench in BENCHES:
        prefix = f'openeta/{bench}_'
        if env_id.startswith(prefix) and env_id.endswith('-v0'):
            key = env_id[len(prefix):-3]
            if bench == 'metaworld':
                key = re.sub(r'^(50|45_ind|45_ood)_', '', key)
            if bench == 'robocasa':
                key = re.sub(r'^(pretrain|target)_', '', key)
            return f'{bench}/{key}'
    raise ValueError(f'Unknown environment identity: {env_id}')


def fingerprint(arrays, *, mujoco=False):
    """Screen equal states at 1e-8 rounding; not a simulator validity receipt."""
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.asarray(value, dtype='<f8')
        if not np.all(np.isfinite(a)):
            raise ValueError('Non-finite initial state')
        if mujoco:
            a = a.reshape(-1)[1:]  # simulation time is not task configuration
        a = np.round(a, 8)
        a[a == 0] = 0  # canonicalise negative zero
        h.update(key.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def historical_sources():
    """Conservatively protect all existing expert source demo IDs, even unused."""
    rows = []
    for campaign in sorted((ROOT / 'SFT_data').iterdir()):
        if not campaign.is_dir():
            continue
        for source_dir in sorted(campaign.glob('sources*')):
            for p in sorted(source_dir.glob('*/*/seed-*.json')):
                d = read(p)
                if d.get('env_id'):
                    rows.append({'task_key': canonical(d['env_id']),
                                 'path': str(p.relative_to(ROOT)),
                                 'sha256': file_digest(p),
                                 'seed': d.get('seed'),
                                 'reference_demo_index': d.get('reference_demo_index'),
                                 'source_dataset': d.get('source_dataset')})
    return rows


def historical_training():
    """Actual mixed SFT samples plus the previously audited RL task pools."""
    evidence = defaultdict(lambda: {'sft_samples': 0, 'rl_pool': False,
                                     'baseline_v2': False})
    sources = []
    p = ROOT / 'SFT-train/mixed-data/train.jsonl'
    if p.is_file():
        sources.append({'path': str(p.relative_to(ROOT)), 'sha256': file_digest(p)})
        with p.open() as f:
            for line in f:
                d = json.loads(line).get('metadata', {})
                if d.get('env_id'):
                    evidence[canonical(d['env_id'])]['sft_samples'] += 1
    p = ROOT / 'docs/generated/current-task-membership-20261010.csv'
    if p.is_file():
        sources.append({'path': str(p.relative_to(ROOT)), 'sha256': file_digest(p)})
        with p.open(newline='') as f:
            for d in csv.DictReader(f):
                k = f"{d['benchmark']}/{d['task']}"
                evidence[k]['rl_pool'] = d['rl_curriculum_pool'] == '1'
                evidence[k]['baseline_v2'] = d['baseline_v2'] == '1'
    # Include successful collected episodes as exposure evidence. This does not
    # assert they were used in a checkpoint. Only inspect known episode layouts.
    for campaign in sorted((ROOT / 'SFT_data').iterdir()):
        if not campaign.is_dir():
            continue
        patterns = ('episodes/*/*/seed-*/episode.json',
                    '*/*/attempt-*/episode.json')
        for pattern in patterns:
            for p in sorted(campaign.glob(pattern)):
                d = read(p)
                if d.get('success') is True and d.get('env_id'):
                    item = evidence[canonical(d['env_id'])]
                    item.setdefault('successful_collected_episodes', []).append(str(p.relative_to(ROOT)))
    return dict(evidence), sources


def cache_inventory(task):
    bench, slug = task['bench'], task['task_id']
    records = []
    if bench == 'libero':
        p = Path(task['catalog']['demonstration_path'])
        if p.is_file():
            with h5py.File(p, 'r') as f:
                keys = sorted(f['data'], key=lambda x: int(x.split('_')[-1]))
                for key in keys:
                    g = f['data'][key]
                    state = np.asarray(g.attrs['init_state'])
                    xml = g.attrs['model_file']
                    if isinstance(xml, bytes):
                        xml = xml.decode()
                    records.append({'kind': 'libero_demo_state', 'dataset_uri': str(p),
                        'reference_demo_index': int(key.split('_')[-1]),
                        'state_locator': f'data/{key}@init_state',
                        'state_sha256': digest(state.astype('<f8').tobytes()),
                        'state_fingerprint': fingerprint({'state': state}, mujoco=True),
                        'model_xml_sha256': digest(xml.encode()),
                        'action_dim': int(g['actions'].shape[-1])})
    elif bench == 'robocasa':
        meta = read(HERE / 'catalogs/robocasa_demo_registry.json').get(slug, {})
        if meta.get('installed'):
            p = Path(meta['path'])
            info = read(p / 'meta/info.json')
            episodes = [json.loads(x) for x in (p / 'meta/episodes.jsonl').read_text().splitlines()]
            # Official filter is 100_demos: only use indices 0..99.
            for ep in episodes[:min(100, info['total_episodes'])]:
                i = ep['episode_index']
                extra = p / 'extras' / f'episode_{i:06d}'
                if not all((extra / x).is_file() for x in ('states.npz', 'model.xml.gz', 'ep_meta.json')):
                    continue
                with np.load(extra / 'states.npz', allow_pickle=False) as data:
                    state = np.asarray(data['states'][0])
                with gzip.open(extra / 'model.xml.gz', 'rb') as f:
                    xml = f.read()
                records.append({'kind': 'robocasa_demo_state', 'dataset_uri': str(p),
                    'reference_demo_index': i, 'state_locator': f'extras/episode_{i:06d}/states.npz:states[0]',
                    'state_sha256': digest(state.astype('<f8').tobytes()),
                    'state_fingerprint': fingerprint({'state': state}, mujoco=True),
                    'model_xml_sha256': digest(xml),
                    'episode_meta_sha256': file_digest(extra / 'ep_meta.json'),
                    'language': ep['tasks'][0], 'action_dim': 12})
    elif bench == 'maniskill':
        p = ROOT / 'SFT_data/expert_replay_v1/official_maniskill_demos' / slug / 'rl/trajectory.none.pd_ee_delta_pose.physx_cuda.h5'
        if p.is_file() and p.with_suffix('.json').is_file():
            meta = read(p.with_suffix('.json'))
            with h5py.File(p, 'r') as f:
                for ep in meta['episodes']:
                    if ep.get('success') is not True or ep.get('control_mode') != 'pd_ee_delta_pose':
                        continue
                    key = f"traj_{ep['episode_id']}"
                    if key not in f:
                        continue
                    arrays = {}
                    def visit(name, obj):
                        if isinstance(obj, h5py.Dataset):
                            arrays[name] = obj[0]
                    f[key]['env_states'].visititems(visit)
                    records.append({'kind': 'maniskill_demo_state', 'dataset_uri': str(p.relative_to(ROOT)),
                        'reference_demo_index': ep['episode_id'], 'episode_seed': ep['episode_seed'],
                        'state_locator': f'{key}/env_states/*[0]',
                        'state_fingerprint': fingerprint(arrays),
                        'metadata_sha256': file_digest(p.with_suffix('.json')),
                        'required_sim_backend': 'physx_cuda',
                        'action_dim': int(f[key]['actions'].shape[-1])})
                    if len(records) >= 100:
                        break
    for record in records:
        if record['action_dim'] != task['expected_action_dim']:
            raise ValueError(f'Cache action dimension mismatch: {slug}')
        record['runtime_verified'] = False
    return records


def partition_cached(records, protected, seed, counts):
    by_fingerprint = {}
    # Protect equivalence groups if *any* member appeared in old source plans.
    protected_fp = {r['state_fingerprint'] for r in records
                    if r['reference_demo_index'] in protected}
    for r in records:
        by_fingerprint.setdefault(r['state_fingerprint'], r)
    rows = list(by_fingerprint.values())
    rng = random.Random(seed)
    rng.shuffle(rows)
    train_count = counts['sft_train'] + counts['rl_train']
    prior = [r for r in rows if r['state_fingerprint'] in protected_fp]
    fresh = [r for r in rows if r['state_fingerprint'] not in protected_fp]
    if len(prior) > train_count or len(rows) < train_count + counts['eval_seen']:
        raise ValueError('Not enough independent cache states after historical protection')
    train = prior + fresh[:train_count-len(prior)]
    test = fresh[train_count-len(prior):train_count-len(prior)+counts['eval_seen']]
    rng.shuffle(train)
    result = {'sft_train': train[:counts['sft_train']],
              'rl_train': train[counts['sft_train']:], 'eval_seen': test}
    # Keep the existing training/test allocation stable. A full 50-state
    # cache cannot supply a 51st independent state: use a new native seed
    # candidate for validation instead of taking a state from test.
    validation_count = counts.get('validation', 0)
    unused = fresh[train_count-len(prior)+counts['eval_seen']:]
    if validation_count and len(unused) >= validation_count:
        result['validation'] = unused[:validation_count]
    return result


def build(*, refresh_history=False):
    config = read(HERE / 'selection.v1.json')
    if config.get('split_revision') == 5:
        if refresh_history:
            raise ValueError('Revision 5 extends frozen v4 history; refresh requires a new revision')
        import sys
        sys.path.insert(0, str(ROOT))
        from data_unified.expand import build as expand_build
        expand_build()
        return
    if config.get('split_revision') == 4:
        if refresh_history:
            raise ValueError('Revision 4 reduces frozen v3 pools; historical evidence cannot be refreshed in place')
        import sys
        sys.path.insert(0, str(ROOT))
        from data_unified.reduce import build as reduce_build
        reduce_build()
        return
    history_path = HERE / 'catalogs/historical_evidence.json'
    if history_path.is_file() and not refresh_history:
        snapshot = read(history_path)
        sources, training, history_files = snapshot['expert_sources'], snapshot['training'], snapshot['files']
    else:
        sources = historical_sources()
        training, history_files = historical_training()
        write(history_path, {'expert_sources': sources, 'training': training, 'files': history_files})
    protected = defaultdict(set)
    for s in sources:
        if s['reference_demo_index'] is not None:
            protected[s['task_key']].add(s['reference_demo_index'])
    old_holdout = read(ROOT / 'SFT_data/configs/two_tool_test_holdout.v1.json')
    tasks, cases, refs = [], [], []
    counts = config['states_per_task']
    for bench in BENCHES:
        catalog = read(HERE / f'catalogs/{bench}.json')['tasks']
        eligible = {}
        for row in catalog:
            if bench == 'metaworld' and row['suite'] != 'metaworld_50':
                continue
            if bench == 'robocasa' and not row['task_slug'].startswith('pretrain_'):
                continue
            eligible[canonical(row['id'])] = row
        for role in ('shared', 'generalization'):
            chosen = config['tasks'][bench][role]
            expected = config['shared_tasks_per_bench'] if role == 'shared' else config['generalization_tasks_per_bench']
            if len(chosen) != expected:
                raise ValueError(f'Wrong {role} task count for {bench}: expected {expected}')
            for selected in chosen:
                slug = selected['task_id']
                key = f'{bench}/{slug}'
                row = eligible[key]
                exposure = training.get(key, {})
                if role == 'generalization' and (exposure.get('sft_samples') or exposure.get('rl_pool')
                        or exposure.get('successful_collected_episodes') or protected[key]):
                    raise ValueError(f'Generalization task has local training/expert exposure: {key}')
                old_descriptions = old_holdout['tasks'][bench]
                old_keys = old_holdout.get('canonical_excluded_task_keys', {}).get(bench, [])
                was_reserved = slug in old_keys or row['task_description'] in old_descriptions
                task = {'task_key': key, 'bench': bench, 'task_id': slug, 'env_id': row['id'],
                    'role': role, 'task': row['task_description'],
                    'family': selected['family'], 'selection_reason': selected['reason'],
                    'expected_action_dim': selected.get('expected_action_dim', 6 if bench == 'maniskill' and slug == 'PushT-v1' else {'libero': 7, 'metaworld': 4, 'maniskill': 7, 'robocasa': 12}[bench]),
                    'robot': row.get('default_robot'), 'horizon': selected.get('horizon', row['max_episode_steps']),
                    'historical_exposure': exposure, 'legacy_holdout': was_reserved,
                    'catalog': row, 'instruction_source': selected.get('instruction_source', 'local_registry')}
                task['task'] = selected.get('instruction', task['task'])
                task['generalization_scope'] = selected.get('generalization_scope', 'unseen_task' if role == 'generalization' else None)
                if bench == 'maniskill':
                    robots = row.get('supported_robots') or []
                    task['robot'] = selected.get('robot') or row.get('native_default_robot') or ('panda_stick' if slug == 'PushT-v1' else 'panda' if 'panda' in robots else 'panda_wristcam')
                    if robots and task['robot'] not in robots:
                        raise ValueError(f'Unsupported single-arm task {key}')
                    if not robots and task['robot'] != row.get('native_default_robot'):
                        preflight = selected.get('preflight_result')
                        if not preflight or not (HERE / preflight).is_file():
                            raise ValueError(f'Task requires a live audited robot specification: {key}')
                        observed = read(HERE / preflight)
                        if observed['task_key'] != key or not observed['records'] or not all(
                                r.get('capture_success') and r.get('actual_robot') == task['robot'] and
                                r.get('actual_action_dim') == task['expected_action_dim'] and
                                r.get('same_instance_reproducible') and not r.get('initial_success')
                                for r in observed['records']):
                            raise ValueError(f'Native robot preflight failed: {key}')
                    if slug == 'PickClutterYCB-v1':
                        raise ValueError('Installed PickClutterYCB evaluator always returns false')
                inventory_path = HERE / 'inventory' / bench / f'{slug}.json'
                if inventory_path.is_file():
                    inventory = read(inventory_path)
                    if inventory['task_key'] != key:
                        raise ValueError(f'Inventory task mismatch: {key}')
                    records = inventory['records']
                else:
                    records = cache_inventory(task) if role == 'shared' else []
                    inventory = {'task_key': key, 'records': records}
                    write(inventory_path, inventory)
                task['cache_states'] = len(records)
                task['cache_unique_states'] = len({x['state_fingerprint'] for x in records})
                task['sft_source'] = ('cached_official_demo' if records else
                                      'official_scripted_policy' if bench == 'metaworld' else
                                      'new_teacher_collection_required')
                if role == 'generalization':
                    task['sft_source'] = 'forbidden_generalization_holdout'
                if bench == 'metaworld':
                    task['expert_policy'] = selected['expert_policy']
                if bench == 'robocasa' and records:
                    languages = sorted({r['language'] for r in records})
                    task['task'] = languages[0] if len(languages) == 1 else task['task']
                    task['instruction_source'] = 'cache_episode_language' if len(languages) == 1 else 'per_case_cache_episode_language'
                    task['language_variant_count'] = len(languages)
                task['environment_config'] = {
                    'robot': task['robot'], 'control_mode': selected.get('control_mode', 'pd_ee_delta_pose' if bench == 'maniskill' else 'benchmark_native'),
                    'obs_mode': selected.get('obs_mode', 'rgbd'),
                    'pass_robot_uids': selected.get('pass_robot_uids', True),
                    'image_width': 256, 'image_height': 256,
                    'model_scope': 'native_task_distribution' if not records else 'frozen_cached_episode_scene',
                    'robocasa_split': 'pretrain' if bench == 'robocasa' else None,
                    'sim_backend': 'physx_cuda' if bench == 'maniskill' else 'mujoco'}
                task.pop('catalog')
                tasks.append(task)
                phase_rows = partition_cached(records, protected[key], key + str(config['selection_seed']), counts) if records else {}
                phases = ('sft_train', 'rl_train', 'validation', 'eval_seen') if role == 'shared' else ('eval_generalization',)
                for phase in phases:
                    n = counts[phase]
                    for i in range(n):
                        # Stable per-task, phase-specific 31-bit seeds; collision
                        # checks below fail rather than silently accepting reuse.
                        seed = int(digest(f"{config['selection_seed']}:{key}:{phase}:{i}".encode())[:8], 16) % (2**31)
                        case_id = f'{key}/{phase}/{i:03d}'
                        ref = phase_rows.get(phase, [None] * n)[i]
                        state = ({'kind': 'reference_demo', 'reference_demo_index': ref['reference_demo_index'],
                                  'source_ref_id': case_id, 'state_fingerprint': ref['state_fingerprint']}
                                 if ref else {'kind': 'native_seed', 'state_fingerprint': None})
                        if ref:
                            refs.append({'source_ref_id': case_id, 'task_key': key, 'phase': phase, **ref})
                        cases.append({'case_id': case_id, 'task_key': key, 'bench': bench,
                            'task_id': slug, 'env_id': row['id'], 'phase': phase, 'seed': seed,
                            'seed_role': 'pre_restore_seed' if ref else 'environment_seed',
                            'cached_episode_seed': ref.get('episode_seed') if ref else None,
                            'task': ref.get('language', task['task']) if ref else task['task'],
                            'initial_state': state, 'runtime_verified': False})
                print(f"{key}: {role}, cache={len(records)}, unique={task['cache_unique_states']}", flush=True)
    assert len({x['task_key'] for x in tasks}) == len(tasks)
    assert len({x['seed'] for x in cases}) == len(cases)
    fingerprint_phases = defaultdict(set)
    for c in cases:
        fp = c['initial_state']['state_fingerprint']
        if fp:
            fingerprint_phases[c['task_key'], fp].add(c['phase'])
    assert all(len(p) == 1 for p in fingerprint_phases.values())
    revision = {'schema_version': 'openeta.unified_data.v1', 'selection_seed': config['selection_seed'],
        'validation_split': config['validation_split'], 'split_revision': config['split_revision'],
        'status': 'partitioned_pending_runtime_verification',
        'task_selection': 'selection.v1.json', 'task_count': len(tasks),
        'shared_task_count': sum(t['role'] == 'shared' for t in tasks),
        'generalization_task_count': sum(t['role'] == 'generalization' for t in tasks),
        'states_per_task': counts, 'tasks': tasks}
    write(HERE / 'task_catalog.json', revision)
    for name, rows in [('initial_states.jsonl', cases), ('source_refs.jsonl', refs)]:
        (HERE / name).write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))
    for phase in counts:
        selected = [c for c in cases if c['phase'] == phase]
        phase_tasks = [t for t in tasks if t['role'] == ('generalization' if phase == 'eval_generalization' else 'shared')]
        write(HERE / 'manifests' / f'{phase}.json', {'schema_version': 'openeta.unified_phase.v1',
            'phase': phase, 'runtime_ready': False, 'tasks': phase_tasks, 'cases': selected,
            'execution_requirement': 'Use initial_state selector, not seed alone. Verify state and action contract before execution.'})
    report = {'task_counts': {b: dict(Counter(t['role'] for t in tasks if t['bench'] == b)) for b in BENCHES},
        'case_counts': dict(Counter(c['phase'] for c in cases)),
        'cached_case_count': len(refs), 'native_seed_case_count': len(cases)-len(refs),
        'cache_only_task_count': sum(bool(t['cache_states']) for t in tasks if t['role'] == 'shared'),
        'missing_sft_cache': [t['task_key'] for t in tasks if t['role'] == 'shared' and not t['cache_states']],
        'legacy_holdout_now_shared': [t['task_key'] for t in tasks if t['role'] == 'shared' and t['legacy_holdout']],
        'known_historical_demo_indices_protected_from_test': {k: sorted(v) for k,v in protected.items() if v},
        'historical_training_evidence_files': history_files,
        'historical_audit_scope': 'mixed-data train.jsonl, audited RL pools, known successful episode layouts, expert source plans; not every checkpoint or external dataset',
        'historical_expert_sources': sources,
        'runtime_verified_case_count': 0,
        'excluded_tasks': [{'task_key': 'maniskill/PickClutterYCB-v1',
            'reason': 'Installed evaluate() returns success=False for every state.',
            'evidence': 'sim/venvs/maniskill312/lib/python3.12/site-packages/mani_skill/envs/tasks/tabletop/pick_clutter_ycb.py:179'}],
        'pending': ['Native seed cases, including validation with no spare cache state, need actual state capture, reproducibility and cross-phase deduplication.',
                    'Cached states need reset/playback and official success checks; source success is not replay success.',
                    'ManiSkill robot/controller/backend and object/model scope must match cached demonstrations.',
                    'LIBERO direct reset ignores seed; explicit cached-state restoration is required.',
                    'MetaWorld constructor seed needs to be unified with reset seed.',
                    'Extend the existing ManiSkill camera subprofile allowlist for the newly selected tasks.',
                    'Generalization tasks with new robots or joint control need per-task native action/observation contracts and separate transfer reporting.',
                    'Use fresh training under this split; old checkpoints need additional state exposure audit.']}
    write(HERE / 'audit.json', report)
    paths = [HERE / 'build.py', HERE / 'check.py', HERE / 'discover.py', HERE / 'selection.v1.json', HERE / 'task_catalog.json', HERE / 'initial_states.jsonl',
             HERE / 'source_refs.jsonl', HERE / 'audit.json']
    paths += sorted((HERE / 'manifests').glob('*.json')) + sorted((HERE / 'catalogs').glob('*.json'))
    paths += sorted((HERE / 'inventory').glob('*/*.json'))
    for bench in config['tasks'].values():
        for role in bench.values():
            for selected in role:
                if selected.get('preflight_result'):
                    paths.append(HERE / selected['preflight_result'])
    if (HERE / 'replacement-record.v3.json').is_file():
        paths.append(HERE / 'replacement-record.v3.json')
    write(HERE / 'checksums.json', {str(p.relative_to(HERE)): file_digest(p) for p in paths})
    print(json.dumps({k: report[k] for k in ('task_counts', 'case_counts', 'cached_case_count', 'native_seed_case_count')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-history', action='store_true',
                        help='Rescan local training evidence instead of using the frozen snapshot')
    args = parser.parse_args()
    build(refresh_history=args.refresh_history)
