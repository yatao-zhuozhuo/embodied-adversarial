import asyncio
import threading
from sim import bench_worker


def test_mujoco_benches_use_an_owning_thread():
    assert {'metaworld','libero','robocasa'} <= bench_worker._THREAD_AFFINE_BENCHES


def test_http_calls_share_simulator_owner_thread(monkeypatch):
    executor=bench_worker._MainThreadExecutor()
    monkeypatch.setattr(bench_worker.app.state,'sim_executor',executor,raising=False)
    stop=threading.Event()
    owner=[];seen=[]
    def loop():
        while not stop.is_set():executor.run_once(timeout=.01)
    thread=threading.Thread(target=loop)
    thread.start()
    def create():owner.append(threading.get_ident())
    def simulate(kind):
        seen.append((kind,threading.get_ident()))
        assert threading.get_ident()==owner[0]
        return kind
    async def run():
        await bench_worker._run_sim_call(create)
        result=await asyncio.gather(*[bench_worker._run_sim_call(simulate,kind)
            for kind in ['reset','step','render','observe','close']])
        assert result==['reset','step','render','observe','close']
    try:asyncio.run(run())
    finally:stop.set();thread.join(timeout=2)
    assert len({tid for _,tid in seen})==1
    assert owner[0]!=threading.get_ident()


def test_mujoco_worker_cap_does_not_reduce_episode_group(monkeypatch):
    from sim.mcp_server.worker_mgr import _pool_max
    monkeypatch.setenv('OPENETA_WORKER_POOL_MAX','32')
    monkeypatch.setenv('OPENETA_WORKER_POOL_MAX_METAWORLD','8')
    assert _pool_max('metaworld')==8
    assert _pool_max('maniskill')==32
    assert _pool_max()==32
