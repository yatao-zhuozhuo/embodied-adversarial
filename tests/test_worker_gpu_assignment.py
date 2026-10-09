from types import SimpleNamespace

from sim.mcp_server import worker_mgr


def manager(monkeypatch, mode):
    monkeypatch.setattr(worker_mgr, '_detect_gpus', lambda: list(range(8)))
    monkeypatch.setenv('OPENETA_WORKER_GPU_ASSIGNMENT', mode)
    return worker_mgr.BenchWorkerManager()


def add(mgr, bench):
    gpu = mgr._select_worker_gpu(bench)
    mgr._pools.setdefault(bench, []).append(SimpleNamespace(gpu=gpu))
    return gpu


def test_interleaved_preflight_does_not_duplicate_training_gpu(monkeypatch):
    mgr = manager(monkeypatch, 'per_bench')
    add(mgr, 'metaworld')
    for bench in ['maniskill', 'robocasa', 'libero', 'maniskill', 'maniskill', 'maniskill']:
        add(mgr, bench)
    for _ in range(7):
        add(mgr, 'metaworld')
    assert sorted(w.gpu for w in mgr._pools['metaworld']) == list(range(8))


def test_replacement_and_larger_pools_remain_balanced(monkeypatch):
    mgr = manager(monkeypatch, 'per_bench')
    for _ in range(8):
        add(mgr, 'metaworld')
    mgr._pools['metaworld'].pop(3)
    add(mgr, 'libero')
    assert add(mgr, 'metaworld') == 3
    for _ in range(32):
        add(mgr, 'maniskill')
    assert all(sum(w.gpu == gpu for w in mgr._pools['maniskill']) == 4 for gpu in range(8))


def test_default_retains_global_round_robin(monkeypatch):
    mgr = manager(monkeypatch, '')
    assert [add(mgr, bench) for bench in ['metaworld', 'libero', 'metaworld']] == [0, 1, 2]
