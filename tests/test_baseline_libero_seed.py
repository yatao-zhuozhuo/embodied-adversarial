from sim.env_registry import _LibEnvWrapper
from types import ModuleType, SimpleNamespace


def test_libero_reset_seeds_before_sampling_and_keeps_unseeded_reset():
    events = []

    class Native:
        def seed(self, value):
            events.append(("seed", value))

        def reset(self):
            events.append(("reset",))
            return {"agentview_image": "frame"}

    env = _LibEnvWrapper(Native())
    obs, info = env.reset(seed=17)
    assert events == [("seed", 17), ("reset",)]
    assert obs["agentview_image"] == "frame"
    assert info == {}
    assert env.render() == "frame"
    env.reset()
    assert events[-1] == ("reset",)
    assert len(events) == 3


def test_metaworld_seeds_task_pool_at_construction(monkeypatch):
    import sys
    import sim.env_registry as registry

    calls = []
    monkeypatch.setitem(sys.modules, "metaworld", SimpleNamespace(register_mw_envs=lambda: None))

    def make(env_id, **kwargs):
        calls.append((env_id, kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(registry.gym, "make", make)
    registry._make_metaworld_direct("reach-v3", seed=20261002)
    assert calls[0][0] == "Meta-World/MT1"
    assert calls[0][1]["seed"] == 20261002


def test_maniskill_renderer_override_survives_camera_config_retry(monkeypatch):
    import sys
    import sim.env_registry as registry

    package = ModuleType("mani_skill")
    package.__path__ = []
    envs = ModuleType("mani_skill.envs")
    package.envs = envs
    monkeypatch.setitem(sys.modules, "mani_skill", package)
    monkeypatch.setitem(sys.modules, "mani_skill.envs", envs)
    monkeypatch.setenv("OPENETA_MANISKILL_RENDER_BACKEND", "sapien_cpu")
    calls = []
    expected = object()

    def make(env_id, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("unsupported camera override")
        return expected

    monkeypatch.setattr(registry.gym, "make", make)
    actual = registry._make_maniskill_direct("PickCube-v1", image_width=512)
    assert actual is expected
    assert len(calls) == 2
    assert all(c["render_backend"] == "sapien_cpu" for c in calls)


def test_robocasa_uses_renderer_with_hard_reset_resource_cleanup(monkeypatch):
    import sys
    import numpy as np
    from sim.envs.robocasa.direct_env import RoboCasaDirectEnv

    calls = []

    def create_env(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(reset=lambda: {}, action_spec=(-np.ones(12), np.ones(12)))

    monkeypatch.setitem(sys.modules, "robocasa", ModuleType("robocasa"))
    monkeypatch.setitem(sys.modules, "robosuite", SimpleNamespace(macros=SimpleNamespace(IMAGE_CONVENTION="opengl")))
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry",
                        SimpleNamespace(TASK_SET_REGISTRY={"all_tasks": ["TurnOffStove"]}))
    monkeypatch.setitem(sys.modules, "robocasa.utils.dataset_registry_utils",
                        SimpleNamespace(get_task_horizon=lambda _: 450))
    monkeypatch.setitem(sys.modules, "robocasa.utils.env_utils", SimpleNamespace(create_env=create_env))
    env = RoboCasaDirectEnv("TurnOffStove", seed=17)
    assert calls[0]["renderer"] == "mujoco"
    assert calls[0]["render_onscreen"] is False
    assert calls[0]["seed"] == 17
    assert env.action_space.shape == (12,)
