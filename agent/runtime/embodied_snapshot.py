"""Host-owned snapshot boundary for Alice/Bob task replay.

The concrete simulator adapter is responsible for serialising physics state.
This module only defines the runtime-facing reference and lifecycle contract so
Alice and Bob cannot silently fall back to a different reset state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SnapshotRef:
    """Content-addressed reference to a simulator initial state."""

    snapshot_id: str
    env_id: str
    state_uri: str
    state_sha256: str
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "env_id", "state_uri", "state_sha256"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "env_id": self.env_id,
            "state_uri": self.state_uri,
            "state_sha256": self.state_sha256,
            "seed": self.seed,
            "metadata": dict(self.metadata),
        }


class SnapshotCapable(Protocol):
    """Optional simulator capability required for strict Alice/Bob replay."""

    def capture_snapshot(self) -> SnapshotRef:
        """Persist and return the current simulator state."""

    def restore_snapshot(self, snapshot: SnapshotRef) -> None:
        """Restore exactly the referenced state or raise on hash mismatch."""
