import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent.training.single_model_rollout import official_success, audit_initial_state
from scripts.create_single_model_rl_dataset import build_rows
from agent.runtime.two_tool_xml import serialize_decision


@pytest.mark.parametrize('value',[True,1,1.0,[True]])
def test_success_receipt(value):
    assert official_success({'info':{'success':value}},'maniskill')


@pytest.mark.parametrize('payload',[{'reward':1}, {'info':{}}, {'info':{'success':'true'}},
                                    {'info':{'success':[True,False]}}, {'info':{'success':2}}])
def test_bad_success_is_error(payload):
    with pytest.raises(ValueError): official_success(payload,'maniskill')


def test_dense_reward_and_termination_are_not_success():
    assert not official_success({'info':{'success':False},'reward':100,'terminated':True},'metaworld')


def test_group_reset_identity(tmp_path):
    audit_initial_state(tmp_path,'same-task-seed',{'physics':'abc'})
    audit_initial_state(tmp_path,'same-task-seed',{'physics':'abc'})
    with pytest.raises(ValueError,match='Initial-state mismatch'):
        audit_initial_state(tmp_path,'same-task-seed',{'physics':'changed'})


def test_twenty_tasks_and_holdout():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'SFT_data/configs/deepseek_xml_debug5.v1.json').read_text())
    holdout=json.loads((root/'SFT_data/configs/two_tool_test_holdout.v1.json').read_text())
    rows=build_rows(manifest,holdout)
    assert len(rows)==20
    assert len({r['env_request']['initial_key'] for r in rows})==20
    assert all(len(r['messages'])==1 for r in rows)
    assert set(r['env_request']['seed'] for r in rows).isdisjoint(holdout['reserved_seeds'])
    with pytest.raises(ValueError): build_rows(manifest,holdout,tasks=['PickCube-v1'])


@pytest.fixture
def scheduler(tmp_path,monkeypatch):
    monkeypatch.setenv('OPENETA_INPUT_PROTOCOL','openeta.two_tool_input.v1')
    monkeypatch.setenv('OPENETA_SWIFT_ARTIFACT_ROOT',str(tmp_path))
    monkeypatch.setenv('OPENETA_ENABLE_THINKING','false')
    from plugins import single_model_swift_grpo as plugin
    class Environment:
        closed=0
        def __init__(self,config):
            self.config=config
            self.task='Reach the goal'
            self.contract={'action_dim':4,'action_low':[-1]*4,'action_high':[1]*4}
            from agent.runtime.two_tool_input import make_profile
            self.profile=make_profile('metaworld','reach-v3',{'input_control_contract':{'slots':['x']*4}},self.contract)
            self.observation={'cameras':[{'frame_id':'render','rgb':np.full((256,256,3),30,dtype=np.uint8)}],
                              'robot':{'end_effector_pose':{'xyz':[0.,0.,0.]}}}
        def fingerprint(self):return {'state_sha256':'same-physics'}
        def observe(self):return self.observation
        def step(self,action,num_steps):
            self.observation['cameras'][0]['rgb']+=1
            return self.observation,{'info':{'success':False},'executed_control_steps':num_steps,'reward':99},False
        def close(self):type(self).closed+=1
    monkeypatch.setattr(plugin,'NativeTaskEnvironment',Environment)
    obj=plugin.SingleModelScheduler(max_turns=8)
    obj.input_encoder=lambda d:[1,2,3]  # Only hook logic; actual tokenization has a separate integration check.
    obj._extract_logprobs_from_choice=lambda choice:[-.1]*len(choice.token_ids)
    request=SimpleNamespace(uuid='episode-1',data_dict={'env_request':{
        'task_id':'reach-v3','bench':'metaworld','initial_key':'reach:0','max_turns':8,'max_control_steps':40}},
        messages=[],images=[],chat_template_kwargs={})
    return obj,request,plugin,Environment


def choice(xml,reason='stop'):
    return SimpleNamespace(message=SimpleNamespace(content=xml),token_ids=[1,2,3],finish_reason=reason)


def test_multiturn_history_reward_and_finish(scheduler):
    obj,req,plugin,env=scheduler
    async def run():
        await obj.on_trajectory_start([req])
        for turn in range(1,8):
            c=choice(serialize_decision('step_env',{'action':[0.]*4,'num_steps':1}))
            ret=await obj.on_turn_end(req,c,turn)
            assert not ret['done']
            obj.step(req,c,turn)
        assert obj.input_records['episode-1'][-1]['packet']['history_decision_ids']==[4,5,6,7]
        text=json.dumps(req.messages)
        assert 'official_success' not in text and '"reward"' not in text
        ret=await obj.on_turn_end(req,choice(serialize_decision('task_complete',{})),8)
        assert ret['done'] and ret['rollout_infos']['reward']==0
        assert len(obj.input_records['episode-1'])==8 and env.closed==1
        assert plugin.NativeSuccessReward()(['xml'],rollout_infos=[ret['rollout_infos']])==[0.]
    asyncio.run(run())


def test_three_invalid_do_not_execute(scheduler):
    obj,req,_,env=scheduler
    async def run():
        await obj.on_trajectory_start([req])
        for turn in range(1,4):
            c=choice('not valid XML')
            ret=await obj.on_turn_end(req,c,turn)
            if turn<3:obj.step(req,c,turn)
        assert ret['rollout_infos']['reward']==0
        assert ret['rollout_infos']['control_steps']==0
        assert ret['rollout_infos']['termination_reason']=='invalid_xml_limit'
    asyncio.run(run())


def test_infrastructure_error_aborts_batch(scheduler,monkeypatch):
    obj,req,plugin,env=scheduler
    def broken(*args,**kwargs):raise RuntimeError('transport failed after send')
    monkeypatch.setattr(env,'step',broken)
    async def run():
        await obj.on_trajectory_start([req])
        with pytest.raises(RuntimeError,match='transport'):
            await obj.on_turn_end(req,choice(serialize_decision('step_env',{'action':[0.]*4,'num_steps':1})),1)
        assert env.closed==1
        artifact=json.loads(next((obj._artifact_root/'episodes').glob('*.json')).read_text())
        assert artifact['final_info']['reward'] is None
    asyncio.run(run())


def test_native_fingerprint_tracks_physics_and_target():
    from sim.native_state_fingerprint import native_fingerprint
    native=SimpleNamespace(model=SimpleNamespace(body_pos=np.zeros((2,3))),
        data=SimpleNamespace(qpos=np.zeros(4),qvel=np.zeros(4),time=0.),_target_pos=np.array([1.,2.,3.]))
    env=SimpleNamespace(_backend='metaworld',_unwrap=lambda:native)
    initial=native_fingerprint(env)
    assert initial==native_fingerprint(env)
    native.data.qpos[0]=.01
    moved=native_fingerprint(env)
    assert moved!=initial
    native._target_pos[0]=2.
    assert native_fingerprint(env)!=moved
