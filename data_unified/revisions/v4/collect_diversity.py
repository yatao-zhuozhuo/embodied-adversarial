"""Capture task geometry for revision-3 native candidate pools, without rollouts."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from data_unified.probe_initial_states import create, reset, capture

HERE = ROOT / 'data_unified'


def worker(key):
    old = HERE / 'revisions/v3'
    task = next(t for t in json.loads((old/'task_catalog.json').read_text())['tasks'] if t['task_key'] == key)
    cases = [c for c in map(json.loads, (old/'initial_states.jsonl').read_text().splitlines())
             if c['task_key'] == key and c['initial_state']['kind'] == 'native_seed']
    # Seven measured candidates per phase are sufficient for four selections;
    # retain the earlier two reproducibility probes and sample the remaining pool.
    from collections import defaultdict
    import numpy as np
    grouped = defaultdict(list)
    for c in cases:
        grouped[c['phase']].append(c)
    cases = []
    for rows in grouped.values():
        indices = sorted(set([0, min(1,len(rows)-1)] + np.linspace(2,len(rows)-1,min(5,max(0,len(rows)-2)),dtype=int).tolist()))
        cases.extend(rows[i] for i in indices)
    if task['role'] == 'shared':
        c = dict(next(c for c in cases if c['phase'] == 'validation'))
        import hashlib
        c.update(case_id=key+'/validation/001', seed=int(hashlib.sha256(f"20261010:{key}:validation:1".encode()).hexdigest()[:8],16) % 2**31)
        cases.append(c)
    rows, env = [], None
    try:
        for c in cases:
            try:
                # MetaWorld's constructor seed defines its task pool. ManiSkill
                # resets reuse the native scene; selected states remain candidates
                # until independently recreated by the reproducibility probe.
                if env is None or task['bench'] == 'metaworld':
                    if env is not None:
                        env.close()
                    env = create(task, c)
                obs, info, _ = reset(env, task, c, None)
                d = HERE/'diversity_candidates'/key/c['phase']/c['case_id'].split('/')[-1]
                value = capture(env, task, obs, info, d)
                value.pop('physical_state')
                rows.append({'case_id':c['case_id'], 'task_key':key, 'seed':c['seed'],
                             'initial_state_selector':c['initial_state'], 'capture_success':True,
                             'same_instance_reproducible':False, 'diversity_capture_only':True, **value})
            except Exception as exc:
                rows.append({'case_id':c['case_id'], 'seed':c['seed'], 'capture_success':False,
                             'error':f'{type(exc).__name__}: {exc}'})
        p = HERE/'diversity_candidates'/key/'features.json'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({'task_key':key,'records':rows}, ensure_ascii=False, indent=2)+'\n')
        print(key, sum(r['capture_success'] for r in rows), '/',len(rows), flush=True)
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-task')
    parser.add_argument('--bench', choices=['metaworld','maniskill'])
    args = parser.parse_args()
    if args.worker_task:
        worker(args.worker_task)
        return
    tasks = json.loads((HERE/'revisions/v3/task_catalog.json').read_text())['tasks']
    for task in tasks:
        if task['bench'] != args.bench or task['cache_states']:
            continue
        p = HERE/'diversity_candidates'/task['task_key']/'features.json'
        if p.is_file():
            continue
        python = ROOT/('sim/venvs/metaworld/bin/python' if args.bench == 'metaworld' else 'sim/venvs/maniskill312/bin/python')
        env = {**os.environ, 'MUJOCO_GL':'egl'}
        result = subprocess.run([str(python),str(Path(__file__)), '--worker-task',task['task_key']], env=env)
        if result.returncode:
            print('FAILED',task['task_key'],result.returncode,flush=True)


if __name__ == '__main__':
    main()
