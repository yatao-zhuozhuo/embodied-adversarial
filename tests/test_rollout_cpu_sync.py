import pytest
from types import SimpleNamespace
from agent.training.rollout_cpu_sync import gather_rollout_flags,CpuRolloutControlMixin


def test_control_store_is_tensor_free_and_preserves_rank_order(monkeypatch):
    import torch.distributed as dist
    from agent.training.rollout_cpu_sync import RolloutCpuStore
    import json
    class Store:
        def __init__(self):self.values={}
        def set(self,k,v):self.values[k]=v
        def wait(self,keys,timeout):
            self.values['0/1']=json.dumps([])
            self.values['0/2']=json.dumps([False,True])
        def get(self,k):return self.values[k]
    def forbidden(*args,**kwargs):raise AssertionError('CUDA/tensor collective must not run')
    monkeypatch.setattr(dist,'all_gather_object',forbidden)
    monkeypatch.setattr(dist,'all_gather',forbidden)
    control=RolloutCpuStore(Store(),0,3)
    assert gather_rollout_flags([True],control)==[True,False,True]
    with pytest.raises(ValueError,match='tensor-free CPU store'):
        gather_rollout_flags([True],object())


def test_no_distributed_group_cannot_fall_back_to_nccl(monkeypatch):
    import torch.distributed as dist
    monkeypatch.setattr(dist,'is_initialized',lambda:False)
    assert gather_rollout_flags([True],None)==[True]
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    monkeypatch.setattr(dist,'get_world_size',lambda:8)
    with pytest.raises(ValueError,match='explicit CPU store'):gather_rollout_flags([True],None)


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
