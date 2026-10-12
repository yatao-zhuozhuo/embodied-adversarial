#!/usr/bin/env python3
"""Check the frozen partition and references without simulator dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def check(root=HERE):
    root = Path(root)
    checksums = json.loads((root / 'checksums.json').read_text())
    for relative, expected in checksums.items():
        require(hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected,
                f'Checksum changed: {relative}')
    catalog = json.loads((root / 'task_catalog.json').read_text())
    selection = json.loads((root / 'selection.v1.json').read_text())
    require(catalog['states_per_task'] == selection['states_per_task'], 'Catalog counts differ from selection configuration')
    require(catalog['validation_split'] is True, 'Missing validation split')
    require(catalog['states_per_task']['validation'] == (2 if catalog.get('split_revision', 1) >= 4 else 1),
            'Wrong validation state count for split revision')
    tasks = {t['task_key']: t for t in catalog['tasks']}
    require(len(tasks) == 80, 'Expected 60 shared and 20 generalization tasks')
    for bench in ('libero', 'metaworld', 'maniskill', 'robocasa'):
        counts = Counter(t['role'] for t in tasks.values() if t['bench'] == bench)
        require(counts == {'shared': 15, 'generalization': 5}, f'Wrong task counts: {bench}')
    cases = read_jsonl(root / 'initial_states.jsonl')
    refs = read_jsonl(root / 'source_refs.jsonl')
    require(len({c['case_id'] for c in cases}) == len(cases), 'Duplicate case IDs')
    require(len({c['seed'] for c in cases}) == len(cases), 'Seed collision')
    require(len({r['source_ref_id'] for r in refs}) == len(refs), 'Duplicate source references')
    references = {r['source_ref_id']: r for r in refs}
    fingerprints = defaultdict(set)
    selectors = defaultdict(set)
    counts = Counter((c['task_key'], c['phase']) for c in cases)
    audit = json.loads((root / 'audit.json').read_text())
    historical = audit['known_historical_demo_indices_protected_from_test']
    for c in cases:
        require(c['task_key'] in tasks, f"Unknown task: {c['task_key']}")
        task = tasks[c['task_key']]
        require(c['env_id'] == task['env_id'], 'Environment differs across phases')
        require(c['phase'] in catalog['states_per_task'], 'Unexpected phase')
        require((task['role'] == 'generalization') == (c['phase'] == 'eval_generalization'),
                'Generalization task leaked into shared training/evaluation')
        state = c['initial_state']
        if state['kind'] == 'reference_demo':
            require(state['source_ref_id'] in references, 'Missing source reference')
            ref = references[state['source_ref_id']]
            require(ref['task_key'] == c['task_key'] and ref['phase'] == c['phase'], 'Source phase/task mismatch')
            require(ref['state_fingerprint'] == state['state_fingerprint'], 'State fingerprint mismatch')
            require(ref['reference_demo_index'] == state['reference_demo_index'], 'Demo selector mismatch')
            require(ref['action_dim'] == task['expected_action_dim'], 'Cached controller dimension mismatch')
            fingerprints[c['task_key'], ref['state_fingerprint']].add(c['phase'])
            selectors[c['task_key'], ref['dataset_uri'], ref['reference_demo_index']].add(c['phase'])
            if c['phase'] in ('validation', 'eval_seen'):
                require(ref['reference_demo_index'] not in historical.get(c['task_key'], []),
                        'Historical expert source appeared in validation/test')
                require(ref['reference_demo_index'] not in audit.get('v3_training_pool_protected_from_validation_test',{}).get(c['task_key'],[]),
                        'Previous training-pool state appeared in validation/test')
        else:
            require(state['kind'] == 'native_seed', 'Unknown state selector')
            require(state['state_fingerprint'] is None and not c['runtime_verified'],
                    'Uncaptured seed falsely marked as verified')
    require(all(len(phases) == 1 for phases in fingerprints.values()), 'State fingerprint crosses phases')
    require(all(len(phases) == 1 for phases in selectors.values()), 'Cached episode crosses phases')
    for t in tasks.values():
        phases = ('eval_generalization',) if t['role'] == 'generalization' else ('sft_train', 'rl_train', 'validation', 'eval_seen')
        for phase in phases:
            require(counts[t['task_key'], phase] == catalog['states_per_task'][phase], 'Wrong per-task state count')
        if t['role'] == 'generalization':
            e = t['historical_exposure']
            require(not (e.get('sft_samples') or e.get('rl_pool') or e.get('successful_collected_episodes')),
                    'Generalization task has known training exposure')
    for phase in catalog['states_per_task']:
        manifest = json.loads((root / 'manifests' / f'{phase}.json').read_text())
        require(manifest['cases'] == [c for c in cases if c['phase'] == phase], 'Manifest differs from master cases')
        expected = [t for t in tasks.values()
                    if t['role'] == ('generalization' if phase == 'eval_generalization' else 'shared')]
        require(manifest['tasks'] == expected, 'Manifest task metadata differs from master catalog')
        require(manifest['runtime_ready'] is False, 'Unverified manifest marked executable')
    require(set(references) == {c['initial_state']['source_ref_id'] for c in cases
                               if c['initial_state']['kind'] == 'reference_demo'}, 'Unreferenced cache records')
    return {'tasks': len(tasks), 'cases': len(cases), 'cached_cases': len(refs),
            'checksums_verified': len(checksums), 'partition_check': 'passed',
            'runtime_verification': 'pending'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    args = parser.parse_args()
    print(json.dumps(check(args.root), ensure_ascii=False))
