import pytest
from types import SimpleNamespace
from agent.training.rollout_cpu_sync import gather_rollout_flags,CpuRolloutControlMixin


def test_control_collective_is_cpu_and_preserves_rank_order(monkeypatch):
    import torch.distributed as dist
    group=object()
    monkeypatch.setattr(dist,'get_backend',lambda g:'gloo')
    monkeypatch.setattr(dist,'get_world_size',lambda g:3)
    def gather(output,values,group=None):
        assert values==[True]
        output[:]=[[True],[],[False,True]]
    monkeypatch.setattr(dist,'all_gather_object',gather)
    assert gather_rollout_flags([True],group)==[True,False,True]
    monkeypatch.setattr(dist,'get_backend',lambda g:'nccl')
    with pytest.raises(ValueError,match='CUDA collective'):gather_rollout_flags([True],group)


def test_no_distributed_group_cannot_fall_back_to_nccl(monkeypatch):
    import torch.distributed as dist
    monkeypatch.setattr(dist,'is_initialized',lambda:False)
    assert gather_rollout_flags([True],None)==[True]
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    monkeypatch.setattr(dist,'get_world_size',lambda:8)
    with pytest.raises(ValueError,match='explicit CPU group'):gather_rollout_flags([True],None)


def test_cpu_barrier_precedes_next_inference(monkeypatch):
    import swift.rollout
    events=[]
    def driver(**kwargs):
        assert kwargs['requests']==['request']
        assert kwargs['gather_fn']([True])==[True]
        return kwargs['rollout_fn'](['request'],'config')
    monkeypatch.setattr(swift.rollout,'run_multi_turn',driver)
    monkeypatch.setattr('agent.training.rollout_cpu_sync.gather_rollout_flags',
        lambda v,g:events.append('cpu') or v)
    class Harness(CpuRolloutControlMixin):pass
    obj=Harness();obj._openeta_rollout_cpu_group=None
    obj.args=SimpleNamespace(max_turns=48);obj.multi_turn_scheduler=object()
    obj._rollout=lambda req,cfg:events.append('inference') or ['output']
    obj._postprocess_rollout_outputs=lambda samples,output:output
    assert obj._colocate_multi_turn_infer([],[],'config',['request'])==['output']
    assert events==['cpu','cpu','inference']
