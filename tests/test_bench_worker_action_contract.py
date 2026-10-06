import asyncio
import json

import numpy as np
import pytest

from sim import bench_worker


@pytest.mark.parametrize("dimension", [4, 7, 12])
def test_full_action_contract_and_strict_execution(monkeypatch, dimension):
    gym = pytest.importorskip("gymnasium")

    class Environment:
        action_space = gym.spaces.Box(-1, 1, shape=(dimension,), dtype=np.float32)

    class Request:
        method = "POST"
        path_params = {}
        payload = {"env_id": "test-environment"}

        async def json(self):
            return self.payload

        async def body(self):
            return json.dumps(self.payload).encode()

    async def run(function, *args, **kwargs):
        return function(*args, **kwargs)

    executed = []

    def step(env, action, **kwargs):
        executed.append(action.tolist())
        return {"observation": {}, "terminated": False, "truncated": False}

    monkeypatch.setattr(bench_worker, "_run_sim_call", run)
    monkeypatch.setattr(bench_worker, "_make_env_locked", lambda *args, **kwargs: Environment())
    monkeypatch.setattr(bench_worker, "_step_with_image", step)
    request = Request()
    created = json.loads(asyncio.run(bench_worker.create_env(request)).body)
    handle = created["handle"]
    try:
        assert created["action_dim"] == dimension
        assert len(created["action_low"]) == len(created["action_high"]) == dimension
        request.path_params = {"handle": handle}
        request.payload = {"action": [0.25] * dimension, "num_steps": 2, "strict": True}
        response = asyncio.run(bench_worker.step_env(request))
        assert response.status_code == 200
        assert json.loads(response.body)["executed_control_steps"] == 2
        assert executed == [[0.25] * dimension] * 2
        for invalid in ([0] * (dimension + 1), [2] * dimension, [True] * dimension):
            request.payload["action"] = invalid
            assert asyncio.run(bench_worker.step_env(request)).status_code == 400
        assert len(executed) == 2
    finally:
        bench_worker._envs.pop(handle, None)
        bench_worker._obs_locks.pop(handle, None)
