"""Build native-task GRPO prompts without teacher answers or trajectory reuse."""
import argparse
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def canonical_task(task):
    """Compare native tasks across benchmark splits and manifest versions."""
    env = task['env_id'].removeprefix('openeta/').removesuffix('-v0')
    if env.startswith('robocasa_'):
        return 'robocasa:' + re.sub(r'^(pretrain|target)_', '', env[len('robocasa_'):])
    if env.startswith('metaworld_'):
        return 'metaworld:' + re.sub(r'^(50|45_ind|45_ood)_', '', env[len('metaworld_'):])
    if env.startswith('libero_'):
        return 'libero:' + re.sub(r'[^a-z0-9]+', '_', task['task'].lower()).strip('_')
    return env


def build_rows(manifest, holdout, *, seed=20320000, max_turns=60, max_control_steps=400, tasks=None, baseline=None):
    source = manifest['tasks']
    if not source or len({(r['bench'], r['task_id']) for r in source}) != len(source):
        raise ValueError('Expected nonempty distinct tasks')
    baseline_rows = (baseline or {}).get('tasks', (baseline or {}).get('episodes', []))
    excluded_envs = {r['env_id'] for r in baseline_rows}
    excluded_keys = {canonical_task(r) for r in baseline_rows}
    rows = []
    for i, task in enumerate(source):
        if tasks and task['task_id'] not in tasks:
            continue
        if task['env_id'] in excluded_envs or canonical_task(task) in excluded_keys:
            raise ValueError('Baseline task cannot train')
        excluded = holdout['canonical_excluded_task_keys'].get(task['bench'], [])
        if task['task_id'] in excluded or task['task'] in holdout['tasks'].get(task['bench'], []):
            raise ValueError('Holdout task cannot train')
        task_seed = seed + i * 100
        if task_seed in holdout['reserved_seeds']:
            raise ValueError('Reserved evaluation seed cannot train')
        key = f"{task['bench']}:{task['task_id']}:{task_seed}"
        config = {**task, 'seed': task_seed, 'initial_key': key,
                  'max_turns': max_turns, 'max_control_steps': max_control_steps}
        # Unique placeholder preserves Swift prompt grouping until the live initial hook replaces it.
        rows.append({'messages': [{'role': 'user', 'content': f'Native task bootstrap: {key}'}],
                     'env_request': config})
    if tasks and len(rows) != len(set(tasks)):
        raise ValueError('Unknown requested task')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20320000)
    p.add_argument('--max-turns', type=int, default=60)
    p.add_argument('--max-control-steps', type=int, default=400)
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--manifest', type=Path, default=ROOT/'SFT_data/configs/deepseek_xml_debug5.v1.json')
    p.add_argument('--baseline', type=Path, default=ROOT/'docs/baseline-40-tasks.json')
    a=p.parse_args()
    manifest=json.loads(a.manifest.read_text())
    baseline=json.loads(a.baseline.read_text())
    holdout=json.loads((ROOT/'SFT_data/configs/two_tool_test_holdout.v1.json').read_text())
    rows=build_rows(manifest,holdout,seed=a.seed,max_turns=a.max_turns,
                    max_control_steps=a.max_control_steps,tasks=a.tasks,baseline=baseline)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps({'dataset':str(a.output),'tasks':len(rows)}))

if __name__=='__main__': main()
