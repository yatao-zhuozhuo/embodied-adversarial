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
