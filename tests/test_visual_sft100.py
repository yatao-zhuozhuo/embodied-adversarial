import json

from visual.server import Catalog


def test_short_sft_view_requires_success_and_matching_frozen_case(tmp_path):
    dataset = tmp_path / 'expert_replay_unified_v6_compressed_20261011'
    snapshot = dataset / 'snapshot'
    snapshot.mkdir(parents=True)
    cases = []
    for seed, success, turns in [(1, True, 1), (2, True, 101), (3, False, 1)]:
        case = dict(case_id=f'metaworld/task/sft_train/{seed}', bench='metaworld', task_id='task', seed=seed)
        cases.append(case)
        outcome = dataset / 'results' / (case['case_id'] + '.json')
        outcome.parent.mkdir(parents=True, exist_ok=True)
        outcome.write_text(json.dumps(dict(success=success, compressed_turns=turns)))
        episode = dataset / 'episodes/metaworld/task' / f'seed-{seed}' / 'episode.json'
        episode.parent.mkdir(parents=True)
        episode.write_text(json.dumps(dict(success=success, unified_case=case, bench='metaworld', turns=[{}] * turns)))
    (snapshot / 'sft_train.json').write_text(json.dumps(dict(cases=cases)))
    catalog = Catalog(tmp_path / 'baseline', tmp_path, tmp_path / 'training')
    rows, errors = catalog.scan('expert100')
    assert not errors
    assert len(rows) == 1
    assert rows[0]['source'] == 'expert100'
    assert rows[0]['turns'] == 1
