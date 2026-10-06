import base64
import copy
import io
import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from agent.runtime.two_tool_input import (project_observation,build_input,render_messages,fit_budget,
    execution_record,robot_delta,control_contract)
from agent.runtime.two_tool_xml import parse_decision


def profile(names=('main','wrist')):
    return {'id':'test.v1','cameras':[{'id':n,'role':'wrist' if n=='wrist' else 'primary','size':[8,8]} for n in names], 'image_limit':5}


def obs(tmp_path,turn,colors=(10,20)):
    cameras=[]
    for name,value in zip(('main','wrist'), colors):
        f=io.BytesIO();Image.new('RGB',(8,8),(value,0,0)).save(f,format='PNG')
        cameras.append({'frame_id':name,'rgb_base64':base64.b64encode(f.getvalue()).decode()})
    return project_observation({'cameras':cameras,'robot':{'end_effector_pose':{'xyz':[turn,0,0]},'gripper_state':{'openness':turn/10},'secret':99},'reward':88,'objects':[{'goal':[1,2,3]}]},tmp_path,turn,profile())


def package(tmp_path,history=None):
    return build_input(task='Move the cube',role='controller',system='XML only',profile=profile(),
        contract={'version':'v1','action_dim':4,'slots':[0]*4,'tcp':{'frame':'world'}},
        current=obs(tmp_path,1,(30,40)),previous=obs(tmp_path,0),history=history or [],budget={'remaining_turns':99})


def test_window_labels_dedup_and_visibility(tmp_path):
    p=package(tmp_path)
    assert [e['references'][0]['timing'] for e in p['images']]==['historical','current','current']
    assert [e['references'][0]['camera_id'] for e in p['images']]==['main','main','wrist']
    assert p['current']['robot_delta']['tcp_displacement_m']==[1,0,0]
    messages=render_messages(p)
    text=''.join(x['text'] for x in messages[-1]['content'] if x['type']=='text')
    assert 'secret' not in text and 'reward' not in text and 'objects' not in text
    assert str(tmp_path) not in text
    assert [x['type'] for x in messages[-1]['content']]==['text','text','image_url','text','image_url','text','image_url']
    previous=obs(tmp_path,0)
    p=build_input(task='x',role='x',system='x',profile=profile(),contract={'version':'v1','action_dim':0,'slots':[],'tcp':{'frame':'world'}},current=obs(tmp_path,1),previous=previous,history=[],budget={})
    assert len(p['images'])==2 and len(p['images'][0]['references'])==2


def test_missing_or_wrong_size_camera_is_error(tmp_path):
    with pytest.raises(ValueError,match='missing'):project_observation({'cameras':[]},tmp_path,0,profile())
    f=io.BytesIO();Image.new('RGB',(9,8)).save(f,format='PNG')
    with pytest.raises(ValueError,match='expected native'):
        project_observation({'cameras':[{'frame_id':'main','rgb_base64':base64.b64encode(f.getvalue()).decode()}]},tmp_path,0,profile(('main',)))


def test_history_four_groups_no_hidden_thinking_or_latest_duplicate(tmp_path):
    history=[{'answer':'<think>hidden</think><decision/>','execution':{'decision_id':f'decision-{i}','status':'executed','terminated':False,'truncated':False}} for i in range(6)]
    p=package(tmp_path,history)
    assert p['history_decision_ids']==['decision-2','decision-3','decision-4','decision-5']
    messages=render_messages(p)
    assert 'hidden' not in json.dumps(messages)
    # The newest execution belongs only to current input, not historical receipts.
    assert all('previous_execution' not in m['content'] or json.loads(m['content'])['previous_execution']['decision_id'] != 'decision-5' for m in messages[:-1] if m['role']=='user')
    assert len([m for m in messages if m['role']=='assistant'])==4


def test_exact_budget_trims_whole_groups_then_history_images(tmp_path):
    history=[{'answer':'<decision/>','execution':{'decision_id':'d0','status':'executed','terminated':False,'truncated':False}}]
    p=package(tmp_path,history)
    def count(messages):
        return 100 + 20*sum(m['role']=='assistant' for m in messages)+10*sum(x['type']=='image_url' for x in messages[-1]['content'])
    fitted=fit_budget(p,count,125,5)
    assert not fitted['history'] and len(fitted['images'])==2
    assert fitted['prompt_tokens']==120 and len(fitted['trimming'])==2
    assert len(p['history'])==1
    with pytest.raises(ValueError,match='Minimum'):fit_budget(p,count,124,5)


def test_execution_unknown_and_rejection_are_distinct():
    valid={'valid':True,'decision':{'name':'step_env','parameters':{'action':[0]*4,'num_steps':10}}}
    r=execution_record(0,valid,receipt={'steps_executed':2,'terminated':True,'reward':100})
    assert r['steps_executed']==2 and r['terminated'] and 'reward' not in r
    r=execution_record(0,valid,error='transport interrupted',sent=True)
    assert r['status']=='error' and r['steps_executed'] is None and not r['result_known']
    r=execution_record(0,{'valid':False,'errors':['bad XML']})
    assert r['status']=='rejected' and r['steps_executed']==0


def test_push_t_has_no_gripper():
    created={'action_dim':6,'action_low':[-1]*6,'action_high':[1]*6,'input_capabilities':{'control_mode':'pd_ee_delta_pose','has_gripper':False,'translation_frame':'root_translation','rotation_frame':'root_aligned_body_rotation','position_upper':0.1}}
    c=control_contract(created,{'benchmark':'maniskill3','control_mode':'pd_ee_delta_pose','action_dim':6,'source_versions':{}})
    assert len(c['slots'])==6 and c['gripper']=={'supported':False}


def test_xml_whitespace_is_identical_in_validation_and_execution():
    c={'action_dim':4,'action_low':[-1]*4,'action_high':[1]*4}
    r=parse_decision('<decision><kind> tool_call\n</kind><name> observe_env\n</name><reasoning/><parameters/></decision>',c)
    assert r['valid'] and r['decision']['kind']=='tool_call'
    assert r['decision']['name']==r['execution_request']['name']=='observe_env'


def test_maniskill_tcp_is_not_robot_root_and_quaternion_is_xyzw():
    from sim.unified_env import UnifiedEnv
    env=object.__new__(UnifiedEnv)
    env._include_objects=False
    tcp=SimpleNamespace(p=np.array([[1.,2.,3.]]),q=np.array([[0.5,0.1,0.2,0.3]]))
    env._unwrap=lambda:SimpleNamespace(agent=SimpleNamespace(robot=SimpleNamespace(get_pose=lambda:None),tcp=SimpleNamespace(pose=tcp)))
    value=env._normalise_maniskill({'agent':{'qpos':np.zeros((1,9))}})
    assert np.allclose(value['proprio']['ee_pose'],[1,2,3,0.1,0.2,0.3,0.5])


def test_profile_drift_is_rejected():
    created={'action_dim':4,'action_low':[-1]*4,'action_high':[1]*4,'input_capabilities':{'control_mode':'mocap_delta','position_scale_m':0.02}}
    p={'benchmark':'metaworld','action_dim':4,'control_mode':'mocap_delta','source_versions':{},'expected_capabilities':{'control_mode':'mocap_delta','position_scale_m':0.01}}
    with pytest.raises(ValueError,match='changed'):control_contract(created,p)


def test_changed_image_after_snapshot_is_rejected(tmp_path):
    p=package(tmp_path)
    from pathlib import Path
    Path(p['images'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='Image changed'):render_messages(p)
