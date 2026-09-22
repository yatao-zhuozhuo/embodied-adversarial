"""Simulator-side adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adapter.protocol import EnvAction, EnvObservation, StepResult
from agent.runtime.embodied_snapshot import SnapshotRef


class SimulatorAdapter(ABC):
    """Common interface for RLinf-backed and dummy simulators."""

    @abstractmethod
    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        """Reset the simulator and return the first observation."""

    @abstractmethod
    def observe(self) -> EnvObservation:
        """Return the latest simulator observation without stepping."""

    @abstractmethod
    def step(self, action: EnvAction) -> StepResult:
        """Apply an agent action and return the environment result."""

    def capture_snapshot(self) -> SnapshotRef:
        """Capture a replayable state for strict Alice/Bob evaluation.

        Backends must opt in explicitly.  A seed-only reset is not silently
        treated as an equivalent snapshot.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support embodied snapshots"
        )

    def restore_snapshot(self, snapshot: SnapshotRef) -> None:
        """Restore a previously captured state or raise on unsupported backends."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support embodied snapshots"
        )

    def close(self) -> None:
        """Release simulator resources."""
        return None
