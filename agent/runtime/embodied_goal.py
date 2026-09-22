"""Host-owned goal checking boundary for Alice/Bob embodied tasks.

The checker is deliberately a protocol: simulator-specific state extraction and
predicate implementations stay in the simulator adapter, while the runtime owns
the result schema and never accepts an agent-declared success as truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True, slots=True)
class GoalEvaluation:
    success: bool
    score: float
    predicate_type: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("goal score must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "score": self.score,
            "predicate_type": self.predicate_type,
            "details": dict(self.details),
        }


class GoalChecker(Protocol):
    """Host-owned interface used after every candidate terminal state."""

    def evaluate(
        self,
        *,
        state: Mapping[str, Any],
        predicate: Mapping[str, Any],
    ) -> GoalEvaluation:
        """Evaluate a predicate against trusted simulator state."""
