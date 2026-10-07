"""Replay saved, real sampled contexts through the Swift training encoder."""
import os
os.environ.update(MAX_PIXELS='65536', HF_HUB_OFFLINE='1')
import argparse
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from swift.model import get_model_processor
from swift.template import get_template
from swift.rl_core.data import OnPolicySample
from swift.rlhf_trainers.utils import encode_sample
from agent.training.conditional_turns import split_turn_samples
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--episodes',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--latest',type=int,default=32)
a=p.parse_args()
_,processor=get_model_processor('/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B',load_model=False)
template=get_template(processor,template_type='qwen3_5',enable_thinking=False,loss_scale='last_round')
template.set_mode('train')
files=sorted(a.episodes.glob('*.json'),key=lambda x:x.stat().st_mtime)[-a.latest:]
reports=[];before_failures=[]
for path in files:
 d=json.loads(path.read_text()); records=d['visible_inputs'];rid=d['final_info']['request_id']
 for record in records:
  sample=OnPolicySample(request_id=rid,messages=[],images=[],extra={},add_eos=False)
  old=split_turn_samples([sample],{rid:[record]})[0]
  before=encode_sample(old,template)
  active=[i for i,x in enumerate(before['labels']) if x!=-100]
  if before['input_ids'][:active[0]]!=record['prompt_token_ids']:
   before_failures.append({'episode':rid,'turn':record['turn'],
    'prompt_length_delta':len(before['input_ids'][:active[0]])-len(record['prompt_token_ids'])})
  new=split_turn_samples([sample],{rid:[record]},template)[0]
  after=encode_sample(new,template)
  active=[i for i,x in enumerate(after['labels']) if x!=-100]
  assert after['input_ids'][:active[0]]==record['actual_prompt_token_ids']==record['prompt_token_ids'],(rid,record['turn'],'prompt')
  assert [after['input_ids'][i] for i in active]==record['response_token_ids'],(rid,record['turn'],'targets')
  assert len(active)==len(record['rollout_logprobs']),(rid,record['turn'],'logprobs')
  reports.append({'episode':rid,'turn':record['turn'],'passed':True})
 print(json.dumps({'episode':rid,'turns':len(records),'verified':len(reports)}),flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'before_failures':before_failures,'after_verified':len(reports),'episodes':len(files),'reports':reports},indent=2)+'\n')
print(json.dumps({'before_failures':before_failures,'after_verified':len(reports),'episodes':len(files)}),flush=True)
