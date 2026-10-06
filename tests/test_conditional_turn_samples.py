from types import SimpleNamespace
import pytest
from agent.training.conditional_turns import split_turn_samples


def test_turns_preserve_episode_reward_and_raw_targets():
    sample=SimpleNamespace(request_id='episode1',prompt_id='prompt',rollout_infos={'reward':.3},extra={})
    records=[]
    for i in range(6):
        records.append({'prompt_verified':True,'request':{'messages':[{'role':'user','content':str(i)}],
            'images':[f'image{i}'],'chat_template_kwargs':{'enable_thinking':False}},
            'answer':'xml','response_token_ids':[7,8],'response_loss_mask':[1,1],
            'rollout_logprobs':[-.1,-.2],'finish_reason':'stop','prompt_token_ids':[i],
            'turn':i,'episode_id':'episode1'})
    split=split_turn_samples([sample],{'episode1':records})
    assert len(split)==6
    assert {s.request_id for s in split}=={'episode1'}
    assert all(s.rollout_infos['reward']==.3 for s in split)
    assert split[4].images==['image4']
    assert split[4].response_token_ids==[[7,8]]
    records[0]['prompt_verified']=False
    with pytest.raises(ValueError,match='not verified'):split_turn_samples([sample],{'episode1':records})


def test_loss_weights_preserve_mean_tokens_per_episode():
    from agent.training.conditional_turns import episode_token_weights
    samples=[SimpleNamespace(request_id=rid,response_loss_mask=[[1]*count])
             for rid,count in [('a',2),('a',6),('b',3)]]
    assert episode_token_weights(samples)==[.25,.75,1.]

@pytest.mark.parametrize('remote_count', [3, 6])
def test_sequential_backward_preserves_episode_gradient_and_handles_dummy_turns(monkeypatch, remote_count):
    import torch
    from collections import defaultdict
    from contextlib import nullcontext
    from agent.training.conditional_turns import ConditionalTurnsMixin
    monkeypatch.setenv('OPENETA_INPUT_PROTOCOL', 'openeta.two_tool_input.v1')
    monkeypatch.setattr('accelerate.utils.gather_object', lambda values: [values[0], remote_count])
    parameter = torch.nn.Parameter(torch.tensor(1.))
    values = [2., 4., 10.]
    samples = [SimpleNamespace(request_id=rid, response_loss_mask=[[1]*n])
               for rid,n in [('a',2),('a',6),('b',3)]]
    backwards = []
    syncs = []
    def backward(loss):
        loss.backward()
        backwards.append(float(parameter.grad))
    def compute(model, inputs, batch):
        # No second decision forward can occur before the preceding backward.
        assert len(backwards) == len(syncs)
        syncs.append(inputs['i'])
        return parameter * values[inputs['i']], {}
    obj = SimpleNamespace(
        args=SimpleNamespace(async_generate=False,loss_type='grpo'),
        accelerator=SimpleNamespace(distributed_type=SimpleNamespace(value='NO'),device='cpu',
            no_sync=lambda model:nullcontext(),backward=backward),
        optimizer=SimpleNamespace(param_groups=[]),current_gradient_accumulation_steps=1,
        _prepare_inputs=lambda inputs:{'model_inputs':{},'grpo_batch':None,'_origin_data':samples},
        get_chunked_inputs=lambda mi,gb,a,b,origin:({'i':a},None),
        compute_loss_context_manager=nullcontext,_compute_loss_and_metrics=compute,
        _aggregate_and_update_metrics=lambda *args:None,
        _step=0,_current_train_step_time=0.,_metrics={'train':defaultdict(list)})
    loss = ConditionalTurnsMixin.training_step(obj, torch.nn.Linear(1,1), None)
    assert loss.item() == pytest.approx(6.75)
    assert parameter.grad.item() == pytest.approx(6.75)
    assert len(backwards) == remote_count
    assert obj._step == 1


def test_restored_adam_moments_follow_parameter_dtype_without_resetting_step():
    import torch
    from agent.training.conditional_turns import align_adam_moments
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([parameter])
    step = torch.tensor(5., dtype=torch.float32)
    optimizer.state[parameter] = {'step': step,
        'exp_avg': torch.tensor([.25, .5]), 'exp_avg_sq': torch.tensor([.125, .25])}
    assert align_adam_moments(optimizer) == 2
    assert optimizer.state[parameter]['step'] is step
    assert optimizer.state[parameter]['exp_avg'].tolist() == [.25, .5]
    assert optimizer.state[parameter]['exp_avg_sq'].dtype == parameter.dtype
    assert align_adam_moments(optimizer) == 0
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert step.item() == 6
