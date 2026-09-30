import sys
from types import ModuleType, SimpleNamespace

import numpy as np

from adapter.maniskill_sim import ManiSkillSimulatorAdapter


def _fake_maniskill_modules(monkeypatch, calls):
    gymnasium = ModuleType("gymnasium")

    def make(env_id, **kwargs):
        calls.append((env_id, kwargs))
        return SimpleNamespace(action_space=SimpleNamespace(shape=(7,)))

    gymnasium.make = make
    mani_skill = ModuleType("mani_skill")
    mani_skill.__path__ = []
    mani_skill_envs = ModuleType("mani_skill.envs")
    mani_skill.envs = mani_skill_envs
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "mani_skill", mani_skill)
    monkeypatch.setitem(sys.modules, "mani_skill.envs", mani_skill_envs)


def test_adapter_routes_rendering_to_cpu_by_default(monkeypatch, tmp_path):
    calls = []
    _fake_maniskill_modules(monkeypatch, calls)

    adapter = ManiSkillSimulatorAdapter(snapshot_dir=tmp_path)

    assert adapter.render_backend == "sapien_cpu"
    assert calls[0][1]["render_backend"] == "sapien_cpu"
    assert calls[0][1]["num_envs"] == 1
    assert np.prod(adapter._env.action_space.shape) == 7


def test_adapter_allows_render_backend_override(monkeypatch, tmp_path):
    calls = []
    _fake_maniskill_modules(monkeypatch, calls)
    monkeypatch.setenv("OPENETA_MANISKILL_RENDER_BACKEND", "sapien_cuda:3")

    adapter = ManiSkillSimulatorAdapter(snapshot_dir=tmp_path)

    assert adapter.render_backend == "sapien_cuda:3"
    assert calls[0][1]["render_backend"] == "sapien_cuda:3"
