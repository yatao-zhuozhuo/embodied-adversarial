"""Spawn-isolated ManiSkill adapter for colocated training.

SAPIEN owns a Vulkan/CUDA context.  Keeping that context in the same process
as vLLM's sleep-mode CuMem allocator can invalidate Vulkan fences when the
rollout engine wakes.  This proxy keeps the simulator in a fresh ``spawn``
child and exchanges only host-owned protocol objects over a pipe.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
import traceback
from pathlib import Path
from typing import Any

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from adapter.protocol import EnvAction, EnvObservation, StepResult
from adapter.sim import SimulatorAdapter
from agent.runtime.embodied_snapshot import SnapshotRef


_ALLOWED_OPERATIONS = {
    "reset",
    "observe",
    "step",
    "capture_snapshot",
    "restore_snapshot",
    "render_rgb",
    "task_state",
    "set_goal_position",
}


def _set_parent_death_signal() -> None:
    """Ask Linux to terminate the renderer if its training rank disappears."""

    try:
        import ctypes

        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        # This is lifecycle hygiene, not a correctness requirement.
        pass


def _worker_main(connection: Any, config: dict[str, Any]) -> None:
    """Own one ManiSkill environment without importing parent CUDA state."""

    _set_parent_death_signal()
    adapter: ManiSkillSimulatorAdapter | None = None
    try:
        # Swift's single-device mode rewrites CUDA_VISIBLE_DEVICES separately
        # in every rank.  Override it in this fresh interpreter *before*
        # ManiSkill or torch is imported so logical cuda:0 always means the
        # requested physical render GPU.
        physical_render_gpu = str(config.pop("physical_render_gpu"))
        os.environ["CUDA_VISIBLE_DEVICES"] = physical_render_gpu
        os.environ["LOCAL_RANK"] = "0"
        os.environ.pop("LOCAL_WORLD_SIZE", None)

        adapter = ManiSkillSimulatorAdapter(**config)
        render_device = adapter._env.unwrapped.backend.render_device
        connection.send({
            "kind": "ready",
            "pid": os.getpid(),
            "render_backend": adapter.render_backend,
            "physical_render_gpu": physical_render_gpu,
            "render_device": {
                "name": render_device.name,
                "cuda_id": render_device.cuda_id,
                "pci": render_device.pci_string,
            },
            "action_dim": adapter.action_dim,
        })

        while True:
            request = connection.recv()
            request_id = int(request["id"])
            operation = str(request["operation"])
            if operation == "close":
                adapter.close()
                adapter = None
                connection.send({"kind": "result", "id": request_id, "value": None})
                return
            if operation not in _ALLOWED_OPERATIONS:
                raise ValueError(f"unsupported ManiSkill worker operation: {operation}")
            value = getattr(adapter, operation)(
                *tuple(request.get("args") or ()),
                **dict(request.get("kwargs") or {}),
            )
            connection.send({"kind": "result", "id": request_id, "value": value})
    except EOFError:
        return
    except BaseException as exc:  # noqa: BLE001 - propagate child diagnostics
        try:
            connection.send({
                "kind": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
        except Exception:
            pass
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception:
                pass
        connection.close()


class IsolatedManiSkillSimulatorAdapter(SimulatorAdapter):
    """Synchronous proxy whose SAPIEN renderer lives in a spawn child."""

    def __init__(
        self,
        env_id: str = "PickCube-v1",
        *,
        control_mode: str = "pd_ee_delta_pose",
        render_mode: str = "rgb_array",
        render_backend: str | None = None,
        physical_render_gpu: str | int | None = None,
        camera_resolution: int = 128,
        translation_step_m: float = 0.05,
        fine_translation_step_m: float = 0.01,
        max_episode_steps: int = 120,
        snapshot_dir: str | Path | None = None,
        start_timeout_s: float | None = None,
        request_timeout_s: float | None = None,
    ) -> None:
        self.render_backend = str(
            render_backend
            or os.environ.get("OPENETA_MANISKILL_RENDER_BACKEND")
            or "sapien_cuda:0"
        ).strip()
        self.physical_render_gpu = str(
            physical_render_gpu
            if physical_render_gpu is not None
            else os.environ.get("OPENETA_MANISKILL_RENDER_GPU", "0")
        ).strip()
        if not self.physical_render_gpu:
            raise ValueError("physical_render_gpu must be non-empty")
        self._start_timeout_s = float(
            start_timeout_s
            if start_timeout_s is not None
            else os.environ.get("OPENETA_MANISKILL_WORKER_START_TIMEOUT", "180")
        )
        self._request_timeout_s = float(
            request_timeout_s
            if request_timeout_s is not None
            else os.environ.get("OPENETA_MANISKILL_WORKER_REQUEST_TIMEOUT", "600")
        )
        if self._start_timeout_s <= 0 or self._request_timeout_s <= 0:
            raise ValueError("ManiSkill worker timeouts must be positive")

        config = {
            "env_id": env_id,
            "control_mode": control_mode,
            "render_mode": render_mode,
            "render_backend": self.render_backend,
            "physical_render_gpu": self.physical_render_gpu,
            "camera_resolution": int(camera_resolution),
            "translation_step_m": float(translation_step_m),
            "fine_translation_step_m": float(fine_translation_step_m),
            "max_episode_steps": int(max_episode_steps),
            "snapshot_dir": str(Path(snapshot_dir or "runs/maniskill_snapshots").resolve()),
        }
        context = mp.get_context("spawn")
        self._connection, child_connection = context.Pipe(duplex=True)
        self._process = context.Process(
            target=_worker_main,
            args=(child_connection, config),
            daemon=True,
            name=f"openeta-maniskill-gpu{self.physical_render_gpu}",
        )
        self._closed = False
        self._request_id = 0
        self._process.start()
        child_connection.close()
        ready = self._receive(self._start_timeout_s, phase="startup")
        if ready.get("kind") != "ready":
            self._raise_worker_error(ready, phase="startup")
        self.worker_pid = int(ready["pid"])
        self._action_dim = int(ready["action_dim"])
        self.render_device = dict(ready["render_device"])
        print(
            "[OpenETA] ManiSkill worker ready: "
            f"pid={self.worker_pid}, physical_gpu={self.physical_render_gpu}, "
            f"backend={self.render_backend}, device={self.render_device}",
            flush=True,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _receive(self, timeout_s: float, *, phase: str) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._terminate()
                raise TimeoutError(
                    f"ManiSkill worker timed out during {phase} after {timeout_s:.1f}s"
                )
            if self._connection.poll(min(remaining, 0.2)):
                try:
                    response = self._connection.recv()
                except EOFError as exc:
                    exitcode = self._process.exitcode
                    raise RuntimeError(
                        f"ManiSkill worker closed its pipe during {phase}; exitcode={exitcode}"
                    ) from exc
                if not isinstance(response, dict):
                    raise RuntimeError(f"invalid ManiSkill worker response during {phase}")
                return response
            if not self._process.is_alive():
                raise RuntimeError(
                    f"ManiSkill worker exited during {phase}; exitcode={self._process.exitcode}"
                )

    @staticmethod
    def _raise_worker_error(response: dict[str, Any], *, phase: str) -> None:
        detail = response.get("traceback") or response.get("error") or repr(response)
        raise RuntimeError(f"ManiSkill worker failed during {phase}:\n{detail}")

    def _call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            raise RuntimeError("ManiSkill worker is closed")
        self._request_id += 1
        request_id = self._request_id
        try:
            self._connection.send({
                "id": request_id,
                "operation": operation,
                "args": args,
                "kwargs": kwargs,
            })
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise RuntimeError(
                f"ManiSkill worker is unavailable; exitcode={self._process.exitcode}"
            ) from exc
        response = self._receive(self._request_timeout_s, phase=operation)
        if response.get("kind") == "error":
            self._raise_worker_error(response, phase=operation)
        if response.get("kind") != "result" or int(response.get("id", -1)) != request_id:
            raise RuntimeError(f"invalid ManiSkill worker response for {operation}: {response!r}")
        return response.get("value")

    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        return self._call("reset", task=task, seed=seed)

    def observe(self) -> EnvObservation:
        return self._call("observe")

    def step(self, action: EnvAction) -> StepResult:
        return self._call("step", action)

    def capture_snapshot(self) -> SnapshotRef:
        return self._call("capture_snapshot")

    def restore_snapshot(self, snapshot: SnapshotRef) -> None:
        self._call("restore_snapshot", snapshot)

    def render_rgb(self) -> Any:
        return self._call("render_rgb")

    def task_state(self) -> dict[str, Any]:
        return self._call("task_state")

    def set_goal_position(self, position: list[float] | tuple[float, float, float]) -> None:
        self._call("set_goal_position", position)

    def _terminate(self) -> None:
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        if self._process.is_alive() and hasattr(self._process, "kill"):
            self._process.kill()
            self._process.join(timeout=5)

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._call("close")
                self._process.join(timeout=10)
        finally:
            self._closed = True
            self._terminate()
            self._connection.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
