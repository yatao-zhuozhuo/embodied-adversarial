"""Actual Swift multimodal encoding audit, no model weights or GPU required."""
import json
import os
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['MAX_PIXELS']='65536'
os.environ['IMAGE_MAX_TOKEN_NUM']='64'
from agent.training.conditional_turns import ConditionalTurnsMixin
from swift.model import get_model_processor
from swift.template import get_template
from swift.rl_core.data import OnPolicySample
from swift.rlhf_trainers.utils import encode_sample
from agent.runtime import two_tool_input as shared
from agent.runtime.two_tool_xml import serialize_decision
from agent.training.grpo_input import InputBuilder,execution_receipt

_,processor=get_model_processor('/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B',load_model=False)
infer=get_template(processor,template_type='qwen3_5',enable_thinking=False)
infer.set_mode('transformers')
train=get_template(processor,template_type='qwen3_5',enable_thinking=False,loss_scale='last_round+ignore_empty_think')
train.set_mode('train')
profile=shared.make_profile('maniskill','PickCube-v1',{'input_control_contract':{'slots':['x']*7,'mode':'pd_ee_delta_pose'}},
    {'action_dim':7,'action_low':[-1]*7,'action_high':[1]*7})
reports=[]
with tempfile.TemporaryDirectory() as directory:
 b=InputBuilder(profile,role='alice',task='Move the cube.',directory=directory)
 for t in range(7):
    obs={'robot':{'end_effector_pose':{'xyz':[t/100,0,0],'quat_xyzw':[0,0,0,1]},'gripper_state':{'openness':.5}},
         'cameras':[{'frame_id':'base_camera','rgb':[[[30+t,60,90]]*256]*256}]}
    b.observe(obs,observation_id=f'obs{t}',turn=t)
    record=b.build(remaining_turns=32-t,remaining_control_steps=96-t,generation_budget=768,context_limit=28672,
        encode=lambda d:infer.encode(d)['input_ids'])
    response=serialize_decision('step_env',{'action':[0]*7,'num_steps':1})
    ids=processor.tokenizer.encode(response,add_special_tokens=False)
    request=record['request']
    sample=OnPolicySample(messages=deepcopy(request['messages'])+[{'role':'assistant','content':response}],
        images=request['images'],extra={'chat_template_kwargs':{'enable_thinking':False}},
        response_token_ids=[ids],response_loss_mask=[[1]*len(ids)],add_eos=False)
    encoded=encode_sample(deepcopy(sample),train)
    active=[i for i,label in enumerate(encoded['labels']) if label!=-100]
    actual=encoded['input_ids'][:active[0]]
    expected=record['prompt_token_ids']
    if actual!=expected:
        i=next((i for i,(a,e) in enumerate(zip(actual,expected)) if a!=e),min(len(actual),len(expected)))
        raise ValueError(f'prompt mismatch turn={t} index={i} actual={actual[max(0,i-8):i+8]} expected={expected[max(0,i-8):i+8]} lengths={len(actual)},{len(expected)}')
    assert [encoded['input_ids'][i] for i in active]==ids
    assert sample.response_token_ids==[ids], 'audit must not mutate raw generated IDs'
    sample.rollout_logprobs=[[-.1]*len(ids)]
    sample.extra['input_audit']={'prompt_token_ids':expected,'turn':t,'episode_id':'fixture'}
    class Parent:
        def _prepare_batch_inputs(self,samples):return samples
    class Harness(ConditionalTurnsMixin,Parent):
        def _template_context(self,template):return nullcontext()
    harness=Harness();harness.template=train
    harness.args=SimpleNamespace(output_dir=directory)
    harness.state=SimpleNamespace(global_step=0)
    harness.accelerator=SimpleNamespace(process_index=0)
    os.environ['OPENETA_INPUT_PROTOCOL']='openeta.two_tool_input.v1'
    harness._prepare_batch_inputs([sample])
    assert sample.response_token_ids==[ids]
    assert len(encoded['image_grid_thw'])==len(request['images'])
    # Native batch preparation mutates token fields; the chunked multimodal
    # forward then re-encodes that sample. Verify this second encoding too.
    native = deepcopy(sample)
    prepared = encode_sample(native,train)
    chunked = train.encode(native.to_template_dict())
    assert chunked['input_ids'] == prepared['input_ids'] == encoded['input_ids']
    assert chunked['labels'] == prepared['labels'] == encoded['labels']
    reports.append({'turn':t+1,'history':record['packet']['history_decision_ids'],'images':len(request['images']),
        'prompt_tokens':len(actual),'target_tokens':len(ids),'exact_condition_and_mask':True,
        'native_chunk_reencoding_identical':True})
    b.record(response,execution_receipt(decision_id=t+1,command={'tool':'step_env','action':[0]*7,'num_steps':1},executed_steps=1))
path=Path('runs/input_protocol_validation/swift_encoding.json');path.parent.mkdir(exist_ok=True,parents=True)
path.write_text(json.dumps(reports,indent=2)+'\n')
print(json.dumps(reports))
