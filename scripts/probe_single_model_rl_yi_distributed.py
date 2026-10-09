"""Verify all 16 GPUs and the cross-node control store before real rollouts."""
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
dist.init_process_group('nccl', timeout=timedelta(seconds=180))
assert dist.get_world_size() == 16
value = torch.tensor([rank], device='cuda', dtype=torch.float32)
dist.all_reduce(value)
assert value.item() == sum(range(16))
store = dist.distributed_c10d._get_default_store()
store.set(f'yi-probe-rank-{rank}', socket.gethostname())
for peer in range(16):
    assert store.get(f'yi-probe-rank-{peer}')
root = Path(os.environ['RUN_ROOT']) / 'diagnostics' / 'distributed'
root.mkdir(parents=True, exist_ok=True)
report = {'rank':rank,'local_rank':local_rank,'world_size':16,'hostname':socket.gethostname(),
          'gpu':torch.cuda.get_device_name(local_rank),'all_reduce_sum':value.item(),'control_store':'passed'}
(root/f'rank-{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
dist.barrier()
dist.destroy_process_group()
print(json.dumps(report),flush=True)
