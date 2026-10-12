import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'data_unified'


def test_v6_preserves_all_retained_case_and_reference_bindings():
    active = {'libero', 'metaworld', 'maniskill'}
    for name in ['initial_states.jsonl', 'source_refs.jsonl']:
        previous = [json.loads(line) for line in (ROOT / 'revisions/v5' / name).read_text().splitlines()]
        expected = [row for row in previous if row['task_key'].split('/')[0] in active]
        current = [json.loads(line) for line in (ROOT / name).read_text().splitlines()]
        assert current == expected


def test_v6_task_counts_and_sft_scope():
    catalog = json.loads((ROOT / 'task_catalog.json').read_text())
    assert catalog['split_revision'] == 6
    assert catalog['shared_tasks_per_bench'] == {'libero': 25, 'metaworld': 25, 'maniskill': 15}
    manifest = json.loads((ROOT / 'manifests/sft_train.json').read_text())
    assert len(manifest['cases']) == 260
    assert len(manifest['tasks']) == 65
    assert all(case['bench'] != 'robocasa' for case in manifest['cases'])
