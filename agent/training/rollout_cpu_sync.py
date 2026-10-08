"""CPU collectives for rollout control; NCCL remains the gradient backend.

A rank reaching an NCCL object gather early can occupy a GPU needed by a
slower rank's simulator renderer. A tensor-free CPU store avoids that circular resource wait.
"""
from datetime import timedelta


class RolloutCpuStore:
    """A tensor-free rendezvous for the tiny rollout control messages.

    Avoid object collectives altogether: framework/device binding can make
    an apparently CPU process group touch CUDA in a colocated engine.
    """
    def __init__(self,store,rank,world_size):
        self.store=store
        self.rank=rank
        self.world_size=world_size
        self.sequence=0

    def gather(self,values):
        import json
        if any(type(value) is not bool for value in values):
            raise ValueError('Rollout control only supports boolean flags')
        sequence=self.sequence
        self.sequence+=1
        keys=[f'{sequence}/{rank}' for rank in range(self.world_size)]
        self.store.set(keys[self.rank],json.dumps(list(values)))
        self.store.wait(keys,timedelta(minutes=15))
        return [value for key in keys for value in json.loads(self.store.get(key))]


def create_rollout_cpu_group():
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size()==1:
        return None
    # This PyTorch-pinned runtime exposes its rendezvous store through c10d.
    # Prefixing isolates our keys from DDP/vLLM; no group/device is modified.
    store=dist.PrefixStore('openeta.rollout.control.v1/',dist.distributed_c10d._get_default_store())
    return RolloutCpuStore(store,dist.get_rank(),dist.get_world_size())


def gather_rollout_flags(values,group):
    import torch.distributed as dist
    if group is None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size()>1:
            raise ValueError('Distributed rollout requires an explicit CPU store')
        return list(values)
    if not isinstance(group,RolloutCpuStore):
        raise ValueError('Rollout control requires a tensor-free CPU store')
    return group.gather(values)


class CpuRolloutControlMixin:
    def _rollout_cpu_ready(self):
        # Finish simulator work everywhere before entering any GPU inference
        # or weight-transfer collective, including the initial rollout.
        gather_rollout_flags([True],self._openeta_rollout_cpu_group)

    def _colocate_multi_turn_infer(self,samples,first_turn_rollout_outputs,request_config,requests=None):
        from swift.rollout import run_multi_turn
        if requests is None:
            requests=self.samples2requests(samples)
        def infer(reqs,cfg):
            self._rollout_cpu_ready()
            return self._rollout(reqs,cfg)
        outputs=run_multi_turn(requests=requests,first_turn_outputs=first_turn_rollout_outputs,
            scheduler=self.multi_turn_scheduler,
            rollout_fn=infer,request_config=request_config,
            max_turns=self.args.max_turns,
            gather_fn=lambda values:gather_rollout_flags(values,self._openeta_rollout_cpu_group))
        return self._postprocess_rollout_outputs(samples,outputs)
