from scripts.annotate_unified_expert_sft import select_entries, publish_status
from scripts.collect_unified_expert_sft import read
from scripts.export_reasoning_pilot import select_diverse


def test_bench_workers_partition_frozen_plan():
    entries = [dict(case_id=f'{bench}/task/sft_train/000', turns=1) for bench in ['libero', 'metaworld', 'maniskill']]
    plan = dict(episodes=entries)
    assert select_entries(plan, None) == entries
    for entry in entries:
        assert select_entries(plan, entry['case_id'].split('/')[0]) == [entry]


def test_parallel_status_preserves_other_bench_progress(tmp_path):
    plan = dict(episode_count=3, turn_count=30)
    for bench in ['libero', 'maniskill']:
        publish_status(tmp_path, plan, dict(model='gemini-3.6-flash-low', completed_episodes=0, failures=[]), bench)
    status = read(tmp_path / 'status.json')
    assert set(status['benches']) == {'libero', 'maniskill'}
    assert status['planned_episodes'] == 3
    assert not status['complete']


def test_pilot_spreads_selection_across_tasks():
    entries = [dict(case_id=f'libero/{task}/sft_train/{seed}') for task in ['a', 'b', 'c'] for seed in range(4)]
    selected = select_diverse(entries, 5)
    assert len(selected) == 5
    assert len({entry['case_id'].split('/')[1] for entry in selected}) == 3
