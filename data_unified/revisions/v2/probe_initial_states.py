#!/usr/bin/env python3
"""Run real initial-state checks in isolated per-benchmark interpreters.

Artifacts are an overlay: frozen task/state manifests are never rewritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
PROTOCOL = 'openeta.initial_state_probe.v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def numpy(value):
    import numpy as np
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def flatten(value, prefix=''):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(child, f'{prefix}/{key}'))
        return result
    try:
        a = numpy(value).astype('float64')
        if not a.size:
            return {}
        if not __import__('numpy').all(__import__('numpy').isfinite(a)):
            raise ValueError(f'Non-finite state field: {prefix}')
        return {prefix: a.tolist()}
    except (TypeError, ValueError):
        if isinstance(value, str):
            return {prefix: value}
        raise


def fingerprint(fields):
    import numpy as np
    quantized = {k: np.round(np.asarray(v, dtype=float), 6).tolist() if not isinstance(v, str) else v
                 for k, v in fields.items()}
    return hashlib.sha256(json.dumps(quantized, sort_keys=True).encode()).hexdigest()


def max_difference(a, b):
    import numpy as np
    if set(a) != set(b):
        return None
    difference = 0.0
    for key in a:
        if isinstance(a[key], str) or isinstance(b[key], str):
            if a[key] != b[key]:
                return None
            continue
        # Saved single-env trajectories omit the leading singleton batch axis.
        x, y = np.asarray(a[key]).squeeze(), np.asarray(b[key]).squeeze()
        if x.shape != y.shape:
            return None
        difference = max(difference, float(np.max(np.abs(x-y))))
    return difference


def create(task, case):
    import gymnasium as gym
    if task['bench'] == 'metaworld':
        from sim.env_registry import _make_metaworld_direct
        return _make_metaworld_direct(task['task_id'], image_width=256, image_height=256, seed=case['seed'])
    import mani_skill.envs  # noqa: F401
    config = task['environment_config']
    robot_kwargs = {'robot_uids': config['robot']} if config.get('pass_robot_uids', True) else {}
    return gym.make(task['task_id'], **robot_kwargs,
        obs_mode=config['obs_mode'], control_mode=config['control_mode'], num_envs=1,
        sim_backend=config['sim_backend'], max_episode_steps=task['horizon'],
        sensor_configs={'width': 256, 'height': 256}, render_mode='rgb_array')


def reset(env, task, case, reference):
    if reference:
        import h5py
        from mani_skill.trajectory import utils
        p = Path(reference['dataset_uri'])
        if not p.is_absolute():
            p = ROOT / p
        obs, info = env.reset(seed=int(reference['episode_seed']))
        with h5py.File(p, 'r') as f:
            state = utils.dict_to_list_of_dicts(f[f"traj_{reference['reference_demo_index']}/env_states"])[0]
        env.unwrapped.set_state_dict(state)
        obs, info = env.unwrapped.get_obs(), env.unwrapped.get_info()
        expected = flatten(state)
        actual = flatten(env.unwrapped.get_state_dict())
        # BaseEnv restores physics; controller history is not restored by its
        # set_state_dict implementation. Audit physics separately and disclose.
        expected = {k:v for k,v in expected.items() if not k.startswith('/controller')}
        actual = {k:v for k,v in actual.items() if not k.startswith('/controller')}
        restoration = max_difference(expected, actual)
        return obs, info, restoration
    obs, info = env.reset(seed=case['seed'])
    return obs, info, None


def capture(env, task, obs, info, directory):
    import numpy as np
    from PIL import Image
    raw = env.unwrapped
    frames = []
    if task['bench'] == 'metaworld':
        physical = flatten({'qpos': raw.data.qpos.copy(), 'qvel': raw.data.qvel.copy(),
                            'mocap_pos': raw.data.mocap_pos.copy(), 'mocap_quat': raw.data.mocap_quat.copy()})
        relevant = flatten({'task_random_parameters': raw._last_rand_vec,
                            'object_position': raw._get_pos_objects(), 'goal_position': raw._target_pos})
        verdict = raw.evaluate_state(raw._get_obs(), np.zeros(4))
        if isinstance(verdict, tuple):
            verdict = verdict[-1]
        initial_success = bool(verdict.get('success', False))
        frame = env.render()
        images = {'native_camera_2': frame}
    else:
        physical = flatten(raw.get_state_dict())
        relevant = {}
        for name, actor in raw.scene.actors.items():
            if name in ('table', 'ground', 'floor') or name.startswith('dot_'):
                continue
            relevant.update(flatten(actor.pose.raw_pose, 'actor/' + name))
        for name, articulation in raw.scene.articulations.items():
            if articulation is raw.agent.robot:
                continue
            relevant.update(flatten({'pose': articulation.pose.raw_pose,
                                    'qpos': articulation.get_qpos()}, 'articulation/' + name))
        for name in ('goal_pos', 'goal_rot', 'target_angle', 'target_angle_diff', '_model_ids', 'model_id'):
            value = getattr(raw, name, None)
            if value is not None:
                try:
                    relevant.update(flatten(value, 'task/' + name))
                except (TypeError, ValueError):
                    relevant['task/' + name] = str(value)
        physical.update(flatten(obs.get('sensor_param', {}), 'camera'))
        initial_success = bool(numpy(info.get('success', [False])).reshape(-1)[0])
        images = {name: numpy(data['rgb'])[0] for name,data in obs.get('sensor_data', {}).items() if 'rgb' in data}
    if not relevant:
        raise ValueError('No task-relevant object/goal state captured')
    if not images:
        raise ValueError('No real RGB frame captured')
    directory.mkdir(parents=True, exist_ok=True)
    for camera, frame in images.items():
        frame = numpy(frame)
        if frame.ndim != 3 or frame.shape[-1] < 3:
            raise ValueError('Invalid camera frame shape')
        path = directory / (camera + '.png')
        Image.fromarray(frame[..., :3].astype(np.uint8)).save(path)
        frames.append({'camera': camera, 'path': str(path.relative_to(HERE)),
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return {'physical_state': physical, 'task_state': relevant,
            'state_fingerprint': fingerprint(physical), 'task_state_fingerprint': fingerprint(relevant),
            'initial_success': initial_success, 'frames': frames,
            'actual_robot': getattr(raw.agent, 'uid', None) if task['bench'] == 'maniskill' else 'sawyer',
            'actual_action_dim': int(np.prod(env.action_space.shape))}


def one_case(task, case, reference, check_fresh):
    import numpy as np
    directory = HERE / 'runtime_checks' / task['bench'] / task['task_id'] / case['phase'] / case['case_id'].split('/')[-1]
    env = None
    try:
        env = create(task, case)
        obs, info, restoration = reset(env, task, case, reference)
        first = capture(env, task, obs, info, directory)
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        obs, info, _ = reset(env, task, case, reference)
        repeat = capture(env, task, obs, info, directory / 'repeat')
        difference = max_difference(first['physical_state'], repeat['physical_state'])
        task_difference = max_difference(first['task_state'], repeat['task_state'])
        same_instance = difference is not None and difference <= 1e-5 and task_difference is not None and task_difference <= 1e-5
        fresh_difference = None
        if check_fresh:
            env.close()
            env = create(task, case)
            obs, info, _ = reset(env, task, case, reference)
            fresh = capture(env, task, obs, info, directory / 'fresh')
            fresh_difference = max_difference(first['physical_state'], fresh['physical_state'])
            fresh_task_difference = max_difference(first['task_state'], fresh['task_state'])
            fresh_pass = fresh_difference is not None and fresh_difference <= 1e-5 and fresh_task_difference is not None and fresh_task_difference <= 1e-5
        else:
            fresh_pass = None
        return {'schema_version': PROTOCOL, 'case_id': case['case_id'], 'task_key': task['task_key'],
                'seed': case['seed'], 'initial_state_selector': case['initial_state'],
                'reset_protocol': 'metaworld-create-and-preselect-seed.v1' if task['bench'] == 'metaworld' else 'maniskill-native-or-saved-state.v1',
                'creation_seed': case['seed'] if task['bench'] == 'metaworld' else None,
                'environment_config': task['environment_config'],
                'capture_success': True, **first, 'same_instance_reproducible': same_instance,
                'repeat_max_difference': difference, 'repeat_task_max_difference': task_difference,
                'fresh_instance_checked': check_fresh, 'fresh_instance_reproducible': fresh_pass,
                'fresh_max_difference': fresh_difference, 'restoration_max_difference': restoration,
                'restoration_passed': (restoration is not None and restoration <= 1e-5) if reference else None,
                'controller_history_restored': False if reference else None,
                'action_dim_matches': first['actual_action_dim'] == task['expected_action_dim']}
    finally:
        if env is not None:
            env.close()


def worker(task_key, limit, candidate_spec=None):
    if candidate_spec:
        specification = read(candidate_spec)
        task, candidates = specification['task'], specification['cases']
        result_path = HERE / 'runtime_checks/replacement_candidates' / f"{task['task_id']}.json"
    else:
        tasks = {t['task_key']: t for t in read(HERE / 'task_catalog.json')['tasks']}
        task = tasks[task_key]
        candidates = [c for c in map(json.loads, (HERE / 'initial_states.jsonl').read_text().splitlines()) if c['task_key'] == task_key]
        result_path = HERE / 'runtime_checks' / task['bench'] / task['task_id'] / 'results.json'
    refs = {r['source_ref_id']: r for r in map(json.loads, (HERE / 'source_refs.jsonl').read_text().splitlines())}
    chosen, phase_counts = [], Counter()
    for case in candidates:
        if not limit or phase_counts[case['phase']] < limit:
            chosen.append(case)
            phase_counts[case['phase']] += 1
    records = []
    for index, case in enumerate(chosen):
        try:
            reference = refs.get(case['initial_state'].get('source_ref_id'))
            record = one_case(task, case, reference, index == 0)
            records.append(record)
            print(json.dumps({'case': case['case_id'], 'repeat': record['same_instance_reproducible'],
                              'fresh': record['fresh_instance_reproducible']}, ensure_ascii=False), flush=True)
        except Exception as exc:
            records.append({'case_id': case['case_id'], 'task_key': task_key,
                            'capture_success': False, 'error': f'{type(exc).__name__}: {exc}',
                            'traceback': traceback.format_exc()})
            print(records[-1]['error'], flush=True)
            if index == 0:
                break  # unavailable task/assets; do not repeat the same failing construction
        write(result_path,
              {'task_key': task_key, 'candidate_count': len(candidates), 'requested_sample_count': len(chosen), 'records': records})
    write(result_path,
          {'task_key': task_key, 'candidate_count': len(candidates), 'requested_sample_count': len(chosen), 'records': records})


def _summarize_unlocked():
    records, tasks = [], []
    for path in sorted((HERE / 'runtime_checks').glob('*/*/results.json')):
        d = read(path)
        good = [r for r in d['records'] if r.get('capture_success')]
        groups = defaultdict(set)
        for r in good:
            phase = r['case_id'].split('/')[-2]
            groups[r['task_state_fingerprint']].add(phase)
        duplicates = {fp: sorted(phases) for fp,phases in groups.items() if len(phases) > 1}
        passed = (len(good) == d['requested_sample_count'] and
                  all(r['same_instance_reproducible'] and r['action_dim_matches'] and not r['initial_success']
                      and r.get('restoration_passed') is not False for r in good) and
                  all(r['fresh_instance_reproducible'] for r in good if r['fresh_instance_checked']) and not duplicates)
        tasks.append({'task_key': d['task_key'], 'status': 'sample_passed' if passed else 'needs_attention',
            'candidate_count': d['candidate_count'], 'sample_count': len(d['records']), 'captured_count': len(good),
            'unique_task_states': len(groups), 'cross_phase_duplicates': duplicates,
            'errors': [{'case_id':r['case_id'], 'error':r['error']} for r in d['records'] if not r.get('capture_success')],
            'source': str(path.relative_to(HERE))})
        records.extend(d['records'])
    report = {'schema_version': PROTOCOL, 'scope': 'Real reset probes; sample coverage is explicit. Unchecked cases remain unverified.',
              'numeric_tolerance': 1e-5, 'tasks': tasks,
              'case_count': len(records), 'captured_case_count': sum(r.get('capture_success', False) for r in records),
              'task_status_counts': dict(Counter(t['status'] for t in tasks)),
              'cases': {r['case_id']: r for r in records}}
    write(HERE / 'runtime_checks/index.json', report)
    return report


def summarize():
    import fcntl
    directory = HERE / 'runtime_checks'
    directory.mkdir(exist_ok=True)
    with (directory / '.index.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _summarize_unlocked()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benches', nargs='+', choices=['metaworld', 'maniskill'], default=['metaworld', 'maniskill'])
    parser.add_argument('--limit-per-phase', type=int, default=2, help='0 checks every candidate; default samples two per phase')
    parser.add_argument('--worker-task')
    parser.add_argument('--candidate-spec', type=Path, help='Check a proposed task/case specification before changing the frozen selection')
    parser.add_argument('--task-timeout', type=int, default=600)
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    if args.summarize:
        report = summarize()
        print(json.dumps({k:v for k,v in report.items() if k != 'cases'}, ensure_ascii=False, indent=2))
        return
    if args.worker_task:
        worker(args.worker_task, args.limit_per_phase, args.candidate_spec)
        return
    python = {'metaworld': ROOT / 'sim/venvs/metaworld/bin/python', 'maniskill': ROOT / 'sim/venvs/maniskill312/bin/python'}
    for task in read(HERE / 'task_catalog.json')['tasks']:
        if task['bench'] not in args.benches:
            continue
        directory = HERE / 'runtime_checks' / task['bench'] / task['task_id']
        directory.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env['MUJOCO_GL'] = 'egl'
        env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
        env['OMP_NUM_THREADS'] = '1'
        command = [str(python[task['bench']]), str(Path(__file__).resolve()), '--worker-task', task['task_key'],
                   '--limit-per-phase', str(args.limit_per_phase)]
        with (directory / 'probe.log').open('w') as log:
            try:
                completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.task_timeout)
                if completed.returncode:
                    raise RuntimeError(f'Isolated worker exited with code {completed.returncode}')
            except Exception as exc:
                write(directory / 'results.json', {'task_key': task['task_key'], 'candidate_count': None,
                    'requested_sample_count': 1, 'records': [{'case_id': task['task_key'] + '/worker_error',
                      'task_key':task['task_key'], 'capture_success':False, 'error':str(exc)}]})
        summarize()
        result = read(directory / 'results.json')
        print(f"{task['task_key']}: captured {sum(r.get('capture_success', False) for r in result['records'])}/{len(result['records'])}", flush=True)


if __name__ == '__main__':
    main()
