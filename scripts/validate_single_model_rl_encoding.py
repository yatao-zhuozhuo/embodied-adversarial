"""Audit real four-benchmark images with Swift's actual per-turn training encoder."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['MAX_PIXELS']='65536'
os.environ['HF_HUB_OFFLINE']='1'
from swift.model import get_model_processor
from swift.template import get_template
from swift.rl_core.data import OnPolicySample
from swift.rlhf_trainers.utils import encode_sample
from agent.runtime.two_tool_input import build_bounded_input
from agent.runtime.two_tool_xml import serialize_decision
from agent.training.grpo_input import swift_request, execution_receipt
from agent.training.single_model_rollout import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--preflight-dir',type=Path,required=True)
p.add_argument('--ycb-dir',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
_,processor=get_model_processor('/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B',load_model=False)
infer=get_template(processor,template_type='qwen3_5',enable_thinking=False)
infer.set_mode('transformers')
train=get_template(processor,template_type='qwen3_5',enable_thinking=False,loss_scale='last_round')
train.set_mode('train')
reports=[]
paths=[a.preflight_dir/bench/task/'reset-0.json' for bench,task in [
 ('metaworld','reach-v3'),('libero','libero_spatial_task6'),('maniskill','PokeCube-v1'),('robocasa','CoffeeSetupMug')]]
paths.append(a.ycb_dir/'maniskill/PickSingleYCB-v1/reset-0.json')
for path in paths:
 d=json.loads(path.read_text());profile=d['profile'];history=[]
 response=serialize_decision('step_env',{'action':[0.]*profile['action_contract']['action_dim'],'num_steps':1})
 ids=processor.tokenizer.encode(response,add_special_tokens=False)
 for turn in range(1,8):
  package,request,prompt=build_bounded_input(task='Complete the native task',role='task solver',profile=profile,
   current=d['before'] if turn==1 else d['after'],previous=None if turn==1 else d['before'],history=history,
   turn=turn,max_turns=60,remaining_steps=400-turn+1,generation_budget=768,context_limit=16384,
   render_request=lambda pkg:swift_request(pkg,False),encode_request=lambda req:infer.encode(req)['input_ids'])
  sample=OnPolicySample(messages=deepcopy(request['messages'])+[{'role':'assistant','content':response}],
   images=request['images'],extra={'chat_template_kwargs':{'enable_thinking':False}},response_token_ids=[ids],
   response_loss_mask=[[1]*len(ids)],add_eos=False)
  encoded=encode_sample(sample,train)
  active=[i for i,v in enumerate(encoded['labels']) if v!=-100]
  assert active and encoded['input_ids'][:active[0]]==prompt
  assert [encoded['input_ids'][i] for i in active]==ids
  reports.append({'profile':profile['id'],'turn':turn,'history':package['history_decision_ids'],
                  'images':len(request['images']),'prompt_tokens':len(prompt),'target_tokens':len(ids),'passed':True})
  history.append({**execution_receipt(decision_id=turn,command={'tool':'step_env','action':[0.]*profile['action_contract']['action_dim'],'num_steps':1},executed_steps=1),
                  'model_message':{'content':response}})
write_json(a.output,{'kind':'real_images_with_fixture_decisions_encoding_only','reports':reports})
print(json.dumps({'passed':len(reports),'profiles':5,'output':str(a.output)}))
