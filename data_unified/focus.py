"""Project frozen v5 bindings onto the three active benchmarks for revision 6."""

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ACTIVE = ['libero', 'metaworld', 'maniskill']


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build():
    previous = HERE / 'revisions/v5'
    config = read(previous / 'selection.v1.json')
    config.update(split_revision=6, active_benches=ACTIVE,
                  shared_tasks_per_bench={bench: config['shared_tasks_per_bench'][bench] for bench in ACTIVE},
                  tasks={bench: config['tasks'][bench] for bench in ACTIVE},
                  selection_policy='v6 uses three benchmarks; every retained v5 task, case, seed and source binding is unchanged.')
    write(HERE / 'selection.v1.json', config)
    catalog = read(previous / 'task_catalog.json')
    tasks = [task for task in catalog['tasks'] if task['bench'] in ACTIVE]
    catalog.update(split_revision=6, active_benches=ACTIVE, tasks=tasks, task_count=len(tasks),
                   shared_task_count=sum(task['role'] == 'shared' for task in tasks),
                   generalization_task_count=sum(task['role'] == 'generalization' for task in tasks),
                   shared_tasks_per_bench=config['shared_tasks_per_bench'])
    write(HERE / 'task_catalog.json', catalog)
    cases = [json.loads(line) for line in (previous / 'initial_states.jsonl').read_text().splitlines()]
    cases = [case for case in cases if case['bench'] in ACTIVE]
    refs = [json.loads(line) for line in (previous / 'source_refs.jsonl').read_text().splitlines()]
    refs = [ref for ref in refs if ref['task_key'].split('/')[0] in ACTIVE]
    for name, rows in [('initial_states.jsonl', cases), ('source_refs.jsonl', refs)]:
        (HERE / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    for phase in config['states_per_task']:
        manifest = read(previous / 'manifests' / f'{phase}.json')
        manifest.update(split_revision=6, cases=[case for case in cases if case['phase'] == phase],
                        tasks=[task for task in tasks if task['role'] == ('generalization' if phase == 'eval_generalization' else 'shared')])
        write(HERE / 'manifests' / f'{phase}.json', manifest)
    audit = read(previous / 'audit.json')
    audit.update(active_benches=ACTIVE, inactive_benches=['robocasa'],
                 task_counts={bench: dict(Counter(task['role'] for task in tasks if task['bench'] == bench)) for bench in ACTIVE},
                 case_counts=dict(Counter(case['phase'] for case in cases)),
                 cached_case_count=len(refs), native_seed_case_count=len(cases) - len(refs),
                 cache_only_task_count=sum(bool(task['cache_states']) for task in tasks if task['role'] == 'shared'),
                 missing_sft_cache=[task['task_key'] for task in tasks if task['role'] == 'shared' and not task['cache_states']],
                 legacy_holdout_now_shared=[task['task_key'] for task in tasks if task['role'] == 'shared' and task['legacy_holdout']])
    write(HERE / 'audit.json', audit)
    write(HERE / 'projection-record.v6.json', dict(split_revision=6, previous_revision='revisions/v5',
          active_benches=ACTIVE, preserved_case_count=len(cases), preserved_source_ref_count=len(refs),
          policy='Filter only; no task reselection, seed changes, phase changes or source rebinding.'))
    paths = set(read(previous / 'checksums.json')) | {'focus.py', 'projection-record.v6.json'}
    write(HERE / 'checksums.json', {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in sorted(paths)})
    print(json.dumps(audit['case_counts']))


if __name__ == '__main__':
    build()
