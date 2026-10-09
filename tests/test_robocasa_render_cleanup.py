from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from sim.envs.robocasa.rendering import _cleanup_egl_renderer


@pytest.mark.parametrize("previous_value", [None, 11, 22])
@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_renderer_cleanup_uses_owner_and_restores_other_context(previous_value, cleanup_fails):
    owned = ctypes.c_void_p(11)
    previous = ctypes.c_void_p(previous_value)
    current = [previous_value]
    restorations = []

    def restore(display, draw, read, context):
        restorations.append((display, draw, read, context.value))
        current[0] = context.value
        return True

    egl = SimpleNamespace(
        EGL_DRAW="draw", EGL_READ="read",
        eglGetCurrentContext=lambda: previous,
        eglGetCurrentDisplay=lambda: "display",
        eglGetCurrentSurface=lambda surface: surface,
        eglMakeCurrent=restore,
    )

    def make_current():
        current[0] = owned.value

    renderer = SimpleNamespace(gl_ctx=SimpleNamespace(_context=owned, make_current=make_current))

    def cleanup(candidate):
        assert candidate is renderer
        assert current[0] == owned.value
        current[0] = None
        if cleanup_fails:
            raise ValueError("cleanup failure")

    if cleanup_fails:
        with pytest.raises(ValueError, match="cleanup failure"):
            _cleanup_egl_renderer(renderer, cleanup, egl)
    else:
        _cleanup_egl_renderer(renderer, cleanup, egl)
    assert restorations == ([("display", "draw", "read", 22)] if previous_value == 22 else [])
    assert current[0] == (22 if previous_value == 22 else None)


@pytest.mark.skipif(os.environ.get("OPENETA_TEST_ROBOCASA_RENDER") != "1", reason="requires RoboCasa assets and EGL")
def test_live_rgbd_survives_deferred_renderer_garbage_collection():
    root = Path(__file__).resolve().parents[1]
    script = """
import gc
import numpy as np
from sim.envs.robocasa.direct_env import RoboCasaDirectEnv

env = RoboCasaDirectEnv('CoffeeSetupMug', seed=20310000)
try:
    for repeat in range(3):
        env.reset(seed=20310000)
        sim = env.unwrapped_env.sim
        before = {
            name: sim.render(width=256, height=256, camera_name=name, depth=True)
            for name in env._camera_names
        }
        gc.collect()
        for name, (rgb, depth) in before.items():
            actual_rgb, actual_depth = sim.render(width=256, height=256, camera_name=name, depth=True)
            np.testing.assert_array_equal(actual_rgb, rgb)
            np.testing.assert_array_equal(actual_depth, depth)
finally:
    env.close()
"""
    environment = {**os.environ, "MUJOCO_GL": "egl", "PYTHONPATH": str(root)}
    result = subprocess.run(
        [str(root / "sim/venvs/robocasa/bin/python"), "-c", script],
        cwd=root, env=environment, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
