import json
from pathlib import Path
import pytest
from agent.runtime import two_tool_input as shared
from agent.training.grpo_input import InputBuilder, execution_receipt, swift_request


def fixture(tmp_path):
    contract={'action_dim':7,'action_low':[-1]*7,'action_high':[1]*7}
    profile=shared.make_profile('maniskill','PickCube-v1',{'input_control_contract':{
        'slots':['x']*7,'mode':'pd_ee_delta_pose','tcp':{'frame':'world'}}},contract)
    profile['camera_dimensions']['base_camera']=[2,2]
    obs={'robot':{'joint_positions':[1], 'end_effector_pose':{'xyz':[0,0,0],'quat_xyzw':[0,0,0,1]},
                  'gripper_state':{'openness':.5},'hidden_success':True},
         'objects':[{'position':[9,9,9]}], 'reward':100,'handle':'secret',
         'cameras':[{'frame_id':'base_camera','rgb':[[[25,30,40]]*2]*2}]}
    return InputBuilder(profile,role='alice',task='Move cube',directory=tmp_path),obs


def build(b,limit=100000,encode=None):
    return b.build(remaining_turns=10,remaining_control_steps=50,generation_budget=768,
                  context_limit=limit,encode=encode or (lambda d:list(range(len(json.dumps(d))))))


def test_shared_messages_history_dedup_and_privacy(tmp_path):
    b,o=fixture(tmp_path)
    for turn in range(7):
        b.observe(o,observation_id=f'o{turn}',turn=turn)
        if turn<6:
            b.record('<think>PRIVATE</think><decision>answer</decision>',execution_receipt(
                decision_id=turn+1,command={'tool':'observe_env'},executed_steps=0))
    x=build(b)
    assert x['packet']['history_decision_ids']==[3,4,5,6]
    assert len(x['request']['images'])==1
    assert [r['observation_id'] for r in x['packet']['images'][0]['references']]==['o5','o6']
    text=json.dumps(x)
    for hidden in ('PRIVATE','hidden_success','secret','objects'):
        assert hidden not in text
    assert x['request']==swift_request(x['packet'],False)
    assert x['request']['messages'][-1]['content'].count('<image>')==1


def test_measured_delta_and_unknown_receipt(tmp_path):
    b,o=fixture(tmp_path);b.observe(o,observation_id='o0',turn=0)
    r=execution_receipt(decision_id=1,command={'tool':'step_env','action':[1]*7,'num_steps':3},
                        executed_steps=0,outcome_unknown=True,error='transport')
    assert shared.execution_feedback(r)['last_action']['executed_steps'] is None
    b.record('invalid answer',r)
    o['robot']['end_effector_pose']['xyz']=[.01,0,0]
    b.observe(o,observation_id='o1',turn=1)
    assert build(b)['packet']['dynamic']['robot_delta']['tcp_displacement_m']==[.01,0,0]


def test_budget_preserves_feedback(tmp_path):
    b,o=fixture(tmp_path);b.observe(o,observation_id='o0',turn=0)
    b.record('xml',execution_receipt(decision_id=1,command=None,executed_steps=0,error='invalid'))
    o['cameras'][0]['rgb'][0][0]=[255,0,0]
    b.observe(o,observation_id='o1',turn=1)
    def encode(d):return list(range(1000+100*len(d['messages'])+100*len(d['images'])))
    x=build(b,limit=2100,encode=encode)
    assert not x['packet']['history_decision_ids'] and len(x['request']['images'])==1
    assert x['packet']['dynamic']['execution']=='rejected'
    with pytest.raises(ValueError,match='minimal input'):build(b,limit=1800,encode=encode)


def test_missing_camera_and_partial_think(tmp_path):
    b,o=fixture(tmp_path);o['cameras'][0]['frame_id']='render'
    with pytest.raises(ValueError,match='missing'):b.observe(o,observation_id='o0',turn=0)
    o['cameras'][0]['frame_id']='base_camera';b.observe(o,observation_id='o0',turn=0)
    b.record('<think>secret',execution_receipt(decision_id=1,command=None,executed_steps=0,error='invalid'))
    assert 'secret' not in json.dumps(build(b))
