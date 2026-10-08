import json
from pathlib import Path
from types import SimpleNamespace
from agent.training.curriculum import summarize_groups,gate_result,install_zero_reward_stop
from scripts.run_single_model_curriculum import prepare
ROOT=Path(__file__).resolve().parents[1]


def test_curriculum_keeps_easy_first_excludes_eval_and_budgets(tmp_path):
    plan=prepare(ROOT/'configs/single_model_rl_curriculum_mixed.json',tmp_path)
    assert plan['max_turns']==48 and plan['per_turn_tokens']==1536
    assert plan['episode_tokens']==48*1536 and plan['context_length']==32768
    assert plan['total_updates']==96
    assert [(s['rows'],s['start_step'],s['target_step']) for s in plan['stages']]==[(16,0,32),(16,32,64),(32,64,96)]
    assert sum(s['epochs'] for s in plan['stages'])==5
    first=json.loads(Path(plan['stages'][0]['dataset']).read_text().splitlines()[0])
    assert first['env_request']['task_id']=='reach-v3'
    baseline={r['env_id'] for r in json.loads((ROOT/'docs/baseline-40-tasks.json').read_text())['episodes']}
    keys=set()
    for stage in plan['stages']:
        for line in Path(stage['dataset']).read_text().splitlines():
            row=json.loads(line)['env_request']
            assert row['env_id'] not in baseline and row['max_turns']==48
            assert row['initial_key'] not in keys
            keys.add(row['initial_key'])


def test_gate_rejects_all_zero_and_filters_old_history():
    rows=[{'step':1,'reward':1.},{'step':32,'reward':1.},
          {'global_step/max_steps':'33/64','reward':0.}, {'step':34,'reward':.25},
          {'step':34,'reward':.25}]
    summary=summarize_groups(rows,32)
    assert summary=={'groups':2,'success_rate':.125,'informative_groups':1,'all_zero':False}
    gate={'min_success_rate':.1,'min_successful_tasks':2}
    assert gate_result(summary,2,gate)
    assert not gate_result(summary,1,gate)
    assert not gate_result(summarize_groups([{'step':1,'reward':0.}]),0,gate)


def test_zero_reward_callback_stops_without_duplicate_log_count(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENETA_CURRICULUM_ZERO_LIMIT','4')
    callbacks=[]
    install_zero_reward_stop(SimpleNamespace(add_callback=callbacks.append))
    callback=callbacks[0]
    state=SimpleNamespace(global_step=0,is_world_process_zero=True)
    control=SimpleNamespace(should_training_stop=False,should_save=False)
    args=SimpleNamespace(output_dir=str(tmp_path))
    for step in range(1,4):
        state.global_step=step
        callback.on_log(args,state,control,{'reward':0.})
        callback.on_log(args,state,control,{'reward':0.})
        assert not control.should_training_stop
    state.global_step=4
    callback.on_log(args,state,control,{'reward':0.})
    assert control.should_training_stop and control.should_save
    assert json.loads((tmp_path/'curriculum_zero_reward_stop.json').read_text())['step']==4
