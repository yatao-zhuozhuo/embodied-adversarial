import asyncio
import json
import pytest
from agent.runtime.two_tool_xml import parse_decision, serialize_decision, validate_action


def contract(dim=4):return {'action_dim':dim,'action_low':[-1]*dim,'action_high':[1]*dim}
def wire(dim=4,reason=''):
    return serialize_decision('tool_call','step_env',reason,{'action':[0]*dim,'num_steps':3},contract(dim))


@pytest.mark.parametrize('dim',[4,7,12])
def test_roundtrip_dimensions_and_reasoning(dim):
    reason='a < b & c > d; ]]>\nsecond line'
    parsed=parse_decision(wire(dim,reason),contract(dim))
    assert parsed['valid'] and parsed['decision']['reasoning']==reason
    assert parsed['execution_request']['parameters']=={'action':[0]*dim,'num_steps':3}


@pytest.mark.parametrize('name,kind',[('observe_env','tool_call'),('task_complete','response')])
def test_empty_parameters_and_stop(name,kind):
    p=parse_decision(serialize_decision(kind,name,'',{},contract()),contract())
    assert p['valid']
    assert (p['execution_request'] is None)==(name=='task_complete')


@pytest.mark.parametrize('mutate',[
 lambda x:x+x,
 lambda x:'```xml\n'+x+'\n```',
 lambda x:'prefix '+x,
 lambda x:x.replace('<kind>','<kind>tool_call</kind><kind>',1),
 lambda x:x.replace('</parameters>','<handle>h</handle></parameters>'),
 lambda x:x.replace('step_env','reset_env'),
 lambda x:x.replace('tool_call','tool_batch'),
 lambda x:x.replace('<action type="array">'+('<item type="number">0</item>'*4)+'</action>','<action>[0,0,0,0]</action>'),
 lambda x:x.replace('<num_steps type="integer">3</num_steps>',''),
 lambda x:x.replace('type="integer">3','type="number">3.0'),
 lambda x:x.replace('type="integer">3','type="integer">true'),
 lambda x:x.replace('type="integer">3','type="integer">11'),
 lambda x:x.replace('<item type="number">0</item>','<item type="number">NaN</item>',1),
 lambda x:x.replace('<item type="number">0</item>','<item type="number">Infinity</item>',1),
 lambda x:x.replace('<item type="number">0</item>','<item type="boolean">true</item>',1),
 lambda x:x.replace('<item type="number">0</item>','<item type="number">2</item>',1),
 lambda x:x.replace('</action>',''),
 lambda x:x.replace('<reasoning />','<reasoning>bad & unescaped</reasoning>'),
 lambda x:'<think>'+x,
])
def test_rejects_invalid_without_execution_or_repair(mutate):
    result=parse_decision(mutate(wire()),contract())
    assert not result['valid'] and result['execution_request'] is None
    assert not result['repair_applied']


def test_budget_truncation_and_thinking():
    assert not parse_decision(wire(),contract(),remaining_steps=2)['valid']
    assert not parse_decision(wire(),contract(),finish_reason='length')['valid']
    raw='<think>'+wire(7)+'</think>'+wire()
    assert not parse_decision(raw,contract())['valid']
    result=parse_decision(raw,contract(),allow_thinking=True)
    assert result['valid'] and len(result['decision']['parameters']['action'])==4
    assert result['thinking']==wire(7)
    assert not parse_decision('<think>'+wire()+'</think>',contract(),allow_thinking=True)['valid']


def test_no_parameters_for_stop():
    text='<decision><kind>response</kind><name>task_complete</name><reasoning/><parameters><message>done</message></parameters></decision>'
    assert not parse_decision(text,contract())['valid']


def test_no_json_or_missing_steps_fallback():
    for p in [{'action':'[0,0,0,0]','num_steps':1},{'action':[0]*4}]:
        with pytest.raises(ValueError):validate_action(p,contract())


def test_actual_steps_stop_at_terminal_without_mutating_cache(monkeypatch):
    from sim import bench_worker as worker
    from types import SimpleNamespace
    import numpy as np
    import gymnasium as gym
    h='xml-count-test';terminal={'terminated':True,'truncated':False,'reward':0}
    calls=[]
    def step(*args,**kwargs):
        if h in worker._done_handles:return terminal
        calls.append(1)
        if len(calls)==2:
            worker._done_handles.add(h);return terminal
        return {'terminated':False,'truncated':False,'reward':0}
    async def run(fn):return fn()
    class Request:
        path_params={'handle':h};method='POST'
        async def body(self):return json.dumps({'action':[0]*4,'num_steps':10}).encode()
    monkeypatch.setitem(worker._envs,h,SimpleNamespace(action_space=gym.spaces.Box(-1,1,(4,),dtype=np.float32)))
    monkeypatch.setattr(worker,'_step_with_image',step);monkeypatch.setattr(worker,'_run_sim_call',run)
    try:
        result=json.loads(asyncio.run(worker.step_env(Request())).body)
        assert result['steps_executed']==2 and result['steps_requested']==10
        result=json.loads(asyncio.run(worker.step_env(Request())).body)
        assert result['steps_executed']==0 and len(calls)==2
        assert 'steps_executed' not in terminal
    finally:worker._done_handles.discard(h)


def test_original_planner_numeric_inference_is_allowed_but_canonical_output_is_typed():
    text=wire().replace(' type="array"','').replace(' type="number"','').replace(' type="integer"','')
    result=parse_decision(text,contract())
    assert result['valid'] and result['decision']['parameters']['num_steps']==3
    assert 'type="array"' in wire()
