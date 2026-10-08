"""CPU collectives for rollout control; NCCL remains the gradient backend.

A rank reaching an NCCL object gather early can occupy a GPU needed by a
slower rank's simulator renderer. Gloo avoids that circular resource wait.
"""
from datetime import timedelta


def create_rollout_cpu_group():
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size()==1:
        return None
    return dist.new_group(backend='gloo',timeout=timedelta(minutes=15))


def gather_rollout_flags(values,group):
    import torch.distributed as dist
    if group is None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size()>1:
            raise ValueError('Distributed rollout requires an explicit CPU group')
        return list(values)
    if dist.get_backend(group)!='gloo':
        raise ValueError('Rollout control must not use a CUDA collective')
    gathered=[None]*dist.get_world_size(group)
    dist.all_gather_object(gathered,list(values),group=group)
    return [value for rank_values in gathered for value in rank_values]


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
