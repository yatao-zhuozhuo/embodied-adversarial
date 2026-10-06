"""Check actual Swift request-aware advantages are invariant to turn duplication."""
import json
import torch
from swift.rl_core.advantage import compute_advantages_dynamic

base=torch.tensor([[float(i)] for i in range(8)])
ids=[str(i) for i in range(8)]
a,_=compute_advantages_dynamic(base,torch.ones(1),['p']*8,ids)
expanded_ids=[rid for i,rid in enumerate(ids) for _ in range(i+1)]
b,rewards=compute_advantages_dynamic(torch.tensor([[float(i)] for i,rid in enumerate(ids) for _ in range(i+1)]),
                                    torch.ones(1),['p']*len(expanded_ids),expanded_ids)
assert torch.allclose(b,torch.tensor([a[int(rid)] for rid in expanded_ids]))
assert rewards.tolist()==[float(rid) for rid in expanded_ids]
print(json.dumps({'swift_episode_advantages_invariant_to_turn_count':True,'unique_episodes':8,'expanded_turns':len(expanded_ids)}))
