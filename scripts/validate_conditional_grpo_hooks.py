"""Exercise actual staged dispatch and weighted backward without loading weights."""
import os
import sys
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ['OPENETA_INPUT_PROTOCOL']='openeta.two_tool_input.v1'
import torch
from agent.training.embodied_staged_colocate import StagedEmbodiedGRPOTrainer
from agent.training.conditional_turns import ConditionalTurnsMixin

seen=[]
engine=SimpleNamespace(enable_lora=True,default_adapter_request='alice',infer=lambda *a,**kw:['generated'])
obj=SimpleNamespace(engine=engine,_disable_sp_context=nullcontext,
    _verify_prompts=lambda requests,outputs:seen.append((requests,outputs)) or outputs)
assert StagedEmbodiedGRPOTrainer._engine_infer_with_adapter(obj,['request'],None,None)==['generated']
assert seen==[(['request'],['generated'])]
assert engine.enable_lora and engine.default_adapter_request=='alice'

values=torch.tensor([2.,4.,10.],requires_grad=True)
samples=[SimpleNamespace(request_id=rid,response_loss_mask=[[1]*n]) for rid,n in [('a',2),('a',6),('b',3)]]
obj=SimpleNamespace(args=SimpleNamespace(loss_type='grpo'),model=SimpleNamespace(training=True),
    get_chunked_inputs=lambda mi,gb,a,b,origin:({'i':a},None),
    _compute_loss_and_metrics=lambda m,inputs,b:(values[inputs['i']],{}),
    _aggregate_and_update_metrics=lambda *args:None)
loss=ConditionalTurnsMixin._compute_loss_chunked(obj,None,{},None,samples)
assert abs(loss.item()-6.75)<1e-6
loss.backward()
assert torch.allclose(values.grad,torch.tensor([.125,.375,.5]))
print('staged generation audit dispatch: passed; episode-weighted loss/backward: passed')

from unittest.mock import patch
samples=[SimpleNamespace() for _ in range(4)]
batch=SimpleNamespace(completion_mask=torch.ones((4,2),dtype=torch.bool),teacher_per_token_logps=None)
obj=SimpleNamespace(_has_teacher=True,_has_teacher_explicit=lambda:False,template=SimpleNamespace(sequence_parallel_size=1),
    _compute_advantages=lambda *a:torch.arange(7,dtype=torch.float),_rewards_per_func=None,
    accelerator=SimpleNamespace(process_index=1,device='cpu'),_logs={'advantages':[]},
    split_by_mini_batches=lambda samples:[samples])
with patch('accelerate.utils.gather_object',return_value=[3,4]):
    ConditionalTurnsMixin._postprocess_batch(obj,samples,[{'grpo_batch':batch}])
assert torch.equal(batch.advantages,torch.tensor([[3.,3.],[4.,4.],[5.,5.],[6.,6.]]))
print('unequal turn counts across ranks route advantages correctly: passed')

async def start(requests):
    requests[0].messages=[{'role':'user','content':'new observation'}]
    requests[0].images=['live256.png']
    requests[0].chat_template_kwargs={'enable_thinking':False}
sample=SimpleNamespace(messages=[],images=['stale64.png'],extra={})
request=SimpleNamespace(messages=[],images=['stale64.png'])
def rollout(samples,config,global_inputs):
    assert samples[0].images==['live256.png']
    assert samples[0].messages==request.messages
    assert samples[0].extra['chat_template_kwargs']=={'enable_thinking':False}
    return ['first turn']
obj=SimpleNamespace(multi_turn_scheduler=SimpleNamespace(on_trajectory_start=start),
    enable_server_multi_turn=False,samples2requests=lambda samples:[request],_rollout=rollout,
    _colocate_multi_turn_infer=lambda samples,outputs,config,requests:outputs)
assert ConditionalTurnsMixin._infer_single_or_multi_turn(obj,[sample],None)==['first turn']
print('ordinary Bob first-turn live images and template kwargs preserved: passed')

from swift.infer_engine import RequestConfig
from plugins.embodied_swift_grpo import EmbodiedTwoToolBobScheduler
class EngineParent:
    def _engine_infer(self, requests, config, **kwargs):
        assert config.max_tokens==768 and config.logprobs and config.return_details
        return ['bounded']
class EngineHarness(ConditionalTurnsMixin,EngineParent):
    def _verify_prompts(self,requests,outputs):return outputs
obj=EngineHarness()
obj.multi_turn_scheduler=EmbodiedTwoToolBobScheduler(max_turns=6)
original=RequestConfig(max_tokens=8192,logprobs=False)
assert obj._engine_infer([],original)==['bounded']
assert original.max_tokens==8192
print('ordinary Bob uses independent per-turn generation budget: passed')

request=SimpleNamespace(snapshot={'metadata':{'env_id':'PickCube-v1'}},seed=1,
    goal_predicate={'tolerance':.025},max_steps=32,proposal_id='proposal',
    evaluation_id='evaluation',bob_policy_version='base_model')
proposal=SimpleNamespace(alice_rollout_info={'camera_resolution':256,'thinking_enabled':False})
for adapter_name in ['base_model','openeta_frozen_bob']:
    obj=SimpleNamespace(_bob_adapter_name=adapter_name)
    sample=StagedEmbodiedGRPOTrainer._bob_sample(obj,request,proposal)
    assert sample.extra['adapter_name']==adapter_name
    assert sample.extra['env_request']['camera_resolution']==256
print('valid Alice proposal builds frozen Bob evaluation sample: passed')
