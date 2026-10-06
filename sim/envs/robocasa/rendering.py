"""Keep deferred robosuite renderer cleanup in its owning EGL context."""

from __future__ import annotations

import ctypes


def _cleanup_egl_renderer(renderer, cleanup, egl) -> None:
    previous = egl.eglGetCurrentContext()
    display = egl.eglGetCurrentDisplay()
    draw = egl.eglGetCurrentSurface(egl.EGL_DRAW)
    read = egl.eglGetCurrentSurface(egl.EGL_READ)
    owned = renderer.gl_ctx._context
    restore = bool(previous) and (
        ctypes.cast(previous, ctypes.c_void_p).value
        != ctypes.cast(owned, ctypes.c_void_p).value
    )
    try:
        renderer.gl_ctx.make_current()
        cleanup(renderer)
    finally:
        if restore and not egl.eglMakeCurrent(display, draw, read, previous):
            raise RuntimeError("Cannot restore EGL context after renderer cleanup")


def install_egl_cleanup() -> None:
    """Patch RoboCasa's EGL renderer destructor once, leaving other backends alone."""
    from robosuite.utils import binding_utils

    renderer_class = binding_utils.MjRenderContext
    if binding_utils._MUJOCO_GL != "egl" or getattr(renderer_class, "_openeta_egl_cleanup", False):
        return
    from robosuite.renderers.context.egl_context import EGL

    original_cleanup = renderer_class.__del__

    def cleanup(renderer):
        context = getattr(renderer, "gl_ctx", None)
        if context is None or not getattr(context, "_context", None):
            return original_cleanup(renderer)
        return _cleanup_egl_renderer(renderer, original_cleanup, EGL)

    renderer_class.__del__ = cleanup
    renderer_class._openeta_egl_cleanup = True
