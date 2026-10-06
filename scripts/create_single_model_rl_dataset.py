"""Build native-task GRPO prompts without teacher answers or trajectory reuse."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_rows(manifest, holdout, *, seed=20320000, max_turns=60, max_control_steps=400, tasks=None):
    source = manifest['tasks']
    if len(source) != 20 or len({(r['bench'], r['task_id']) for r in source}) != 20:
        raise ValueError('Expected exactly twenty distinct tasks')
    rows = []
    for i, task in enumerate(source):
        if tasks and task['task_id'] not in tasks:
            continue
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
    a=p.parse_args()
    manifest=json.loads((ROOT/'SFT_data/configs/deepseek_xml_debug5.v1.json').read_text())
    holdout=json.loads((ROOT/'SFT_data/configs/two_tool_test_holdout.v1.json').read_text())
    rows=build_rows(manifest,holdout,seed=a.seed,max_turns=a.max_turns,
                    max_control_steps=a.max_control_steps,tasks=a.tasks)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps({'dataset':str(a.output),'tasks':len(rows)}))

if __name__=='__main__': main()
