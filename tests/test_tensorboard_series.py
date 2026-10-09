import json
from pathlib import Path
from agent.training.tensorboard_series import scalar_tags,history_points,active_state,ScalarSeries


def test_reward_aliases_include_zero_and_skip_nonfinite():
    tags=scalar_tags({'reward':0.,'reward_std':.2,'loss':float('nan'),'note':'x','step':8})
    assert tags=={'train/reward':0.,'train/reward_std':.2,'train/success_rate':0.,'train/success_percent':0.}
    assert scalar_tags({'rewards/NativeSuccessReward/mean':.03125})['train/success_percent']==3.125


def test_active_lineage_ignores_old_future_steps(tmp_path):
    run=tmp_path/'resume';run.mkdir()
    old=tmp_path/'checkpoint-6';old.mkdir()
    (old/'trainer_state.json').write_text(json.dumps({'global_step':6,'log_history':[]}))
    (run/'curriculum_plan.json').write_text(json.dumps({'initial_checkpoint':str(old)}))
    (tmp_path/'current_run.json').write_text(json.dumps({'run_root':str(run)}))
    p=tmp_path/'formal/trainer/01_easy/v-old/checkpoint-99';p.mkdir(parents=True)
    (p/'trainer_state.json').write_text(json.dumps({'global_step':99,'log_history':[]}))
    assert active_state(tmp_path)[1]['global_step']==6
    p=run/'formal/trainer/01_easy/v-new/checkpoint-7';p.mkdir(parents=True)
    for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt']:(p/name).touch()
    (p/'trainer_state.json').write_text(json.dumps({'global_step':7,'log_history':[]}))
    assert active_state(tmp_path)[1]['global_step']==7


def test_history_uses_global_steps_and_real_reward_only():
    state={'global_step':6,'log_history':[{'step':1,'reward':.03125},{'step':4,'reward':.71875},
        {'step':6,'reward':0.},{'step':7,'reward':1.}]}
    points=history_points(state)
    assert sorted(points)==['1','4','6']
    assert points['4']['train/success_percent']==71.875


def test_same_run_survives_collector_restart_and_rollbacks(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    series=ScalarSeries(tmp_path/'continuous')
    state={'global_step':2,'log_history':[{'step':1,'reward':.03125},{'step':2,'reward':0.}]}
    assert series.publish(state)>0
    assert series.publish(state)==0
    series.close()
    series=ScalarSeries(tmp_path/'continuous')
    state={'global_step':3,'log_history':[{'step':1,'reward':.03125},{'step':2,'reward':0.},{'step':3,'reward':.5}]}
    series.publish(state);series.close()
    events=EventAccumulator(str(tmp_path/'continuous'),size_guidance={'scalars':0}).Reload()
    assert [(x.step,x.value) for x in events.Scalars('train/reward')]==[(1,.03125),(2,0.),(3,.5)]
    series=ScalarSeries(tmp_path/'continuous')
    series.publish({'global_step':2,'log_history':state['log_history'][:2]});series.close()
    events=EventAccumulator(str(tmp_path/'continuous'),size_guidance={'scalars':0}).Reload()
    assert [x.step for x in events.Scalars('train/reward')]==[1,2]
