"""Batch accounting and native-task holdout guarantees for the 2-node run."""
from collections import Counter
import json
from pathlib import Path
import pytest
from scripts.create_single_model_rl_dataset import canonical_task, build_rows
from scripts.run_single_model_curriculum import prepare
from scripts.submit_single_model_rl_yi_jobs import request

ROOT=Path(__file__).resolve().parents[1]


def test_curriculum_has_48_disjoint_tasks_and_36_updates(tmp_path):
    cfg=json.loads((ROOT/'configs/single_model_rl_curriculum_mixed_yi_2n16.json').read_text())
    original=json.loads((ROOT/'configs/single_model_rl_curriculum_mixed_yi.json').read_text())
    for key in ('input_protocol','model_protocol','reward','model','max_turns','max_control_steps',
                'per_turn_tokens','episode_tokens','context_length','history_groups','learning_rate',
                'lora_rank','lora_alpha','beta','loss_scale','vllm_mode','initialization'):
        assert cfg[key]==original[key]
    plan=prepare(ROOT/'configs/single_model_rl_curriculum_mixed_yi_2n16.json',tmp_path)
    assert [(s['rows'],s['start_step'],s['target_step']) for s in plan['stages']]==[(16,0,12),(32,12,36)]
    tasks=[t for s in cfg['curriculum'] for t in s['tasks']]
    assert len({canonical_task(t) for t in tasks})==48
    assert {t['bench'] for t in tasks}=={'metaworld','libero','robocasa','maniskill'}
    baseline=json.loads((ROOT/cfg['baseline_exclusion']).read_text())
    assert len(baseline['tasks'])==69
    assert not {canonical_task(t) for t in tasks}&{canonical_task(t) for t in baseline['tasks']}
    assert all(s['epochs']==3 and s['seeds_per_task']==1 for s in cfg['curriculum'])


def test_native_alias_cannot_bypass_baseline():
    a={'env_id':'openeta/robocasa_target_CloseFridge-v0','task':'Close fridge'}
    b={'bench':'robocasa','task_id':'CloseFridge','env_id':'openeta/robocasa_pretrain_CloseFridge-v0','task':'different description'}
    with pytest.raises(ValueError,match='Baseline'):
        build_rows({'tasks':[b]}, {}, baseline={'tasks':[a]})
    assert canonical_task({'env_id':'openeta/metaworld_45_ind_reach-v3-v0'})==canonical_task({'env_id':'openeta/metaworld_50_reach-v3-v0'})


def test_actual_grpo_sampler_has_four_tasks_sixteen_rollouts_each():
    from trl.trainer.utils import RepeatSampler
    for count in (16,32):
        samples=list(RepeatSampler(range(count),mini_repeat_count=16,batch_size=4,repeat_count=1,shuffle=False))
        assert len(samples)==count*16
        for start in range(0,len(samples),64):
            batch=samples[start:start+64]
            assert sorted(Counter(batch).values())==[16]*4
            # Accelerate distributes consecutive per-device batches to global ranks.
            ranks=[batch[i*4:(i+1)*4] for i in range(16)]
            assert all(len(set(rows))==1 for rows in ranks)
            assert len({r[0] for r in ranks[:8]})==2
            assert len({r[0] for r in ranks[8:]})==2


def test_job_requests_two_nodes_eight_cards(tmp_path):
    for rank in range(2):
        r=request(f'test-{rank}',rank,tmp_path,tmp_path/'acceptance.json')
        task=r['spec']['taskSpecs']['task1']
        c=task['template']['spec']['containers'][0]
        assert task['replicas']==1 and c['resources']['limits']['nvidia.com/gpu']=='8'
        env={v['name']:v['value'] for v in c['env']}
        assert env['NNODES']=='2' and env['NODE_RANK']==str(rank)
        assert not any('TOKEN' in k or 'SECRET' in k for k in env)


def test_preflight_reuse_requires_all_nodes_and_exact_data(tmp_path,monkeypatch):
    from scripts.single_model_rl_preflight_history import validate_history
    monkeypatch.setattr('scripts.single_model_rl_preflight_history.subprocess.run',lambda *a,**k:None)
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    rows=[{'env_request':{'bench':b,'task_id':'task','seed':20480000}} for b in ['metaworld','libero','maniskill','robocasa']]
    for root in (old,new):
        (root/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    original={'config_sha256':'abc','stages':[{'name':'easy','dataset':str(old/'tasks.jsonl')}]}
    (old/'curriculum_plan.json').write_text(json.dumps(original))
    reports=[{'bench':r['env_request']['bench'],'task_id':'task','status':'passed'} for r in rows]
    for rank in range(2):
        p=old/f'nodes/{rank}/easy/preflight';p.mkdir(parents=True)
        (p/'summary.json').write_text(json.dumps(reports))
    plan={**original,'stages':[{'name':'easy','dataset':str(new/'tasks.jsonl')}],
          'batch':{'nnodes':2},'preflight_history':str(old),'preflight_source_revision':'HEAD'}
    assert len(validate_history(plan))==4
    bad=old/'nodes/1/easy/preflight/summary.json'
    reports[0]['status']='failed';bad.write_text(json.dumps(reports))
    with pytest.raises(ValueError,match='incomplete'):
        validate_history(plan)
    reports[0]['status']='passed';bad.write_text(json.dumps(reports))
    rows[0]['env_request']['seed']+=1
    (new/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='seeds changed'):
        validate_history(plan)
