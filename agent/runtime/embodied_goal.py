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


class InfoGoalChecker:
    """Small trusted checker for simulator ``info`` receipts.

    ManiSkill's PickCube task exposes ``is_obj_placed`` in ``info``.  The
    agent cannot set this field; the host passes the environment receipt here
    after each step.
    """

    def evaluate(
        self,
        *,
        state: Mapping[str, Any],
        predicate: Mapping[str, Any],
    ) -> GoalEvaluation:
        predicate_type = str(predicate.get("type", ""))
        if predicate_type != "is_obj_placed":
            raise ValueError(f"unsupported info predicate: {predicate_type!r}")
        raw = state.get("is_obj_placed", state.get("success", False))
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else False
        success = bool(raw)
        return GoalEvaluation(
            success=success,
            score=1.0 if success else 0.0,
            predicate_type=predicate_type,
            details={"source": predicate.get("source", "environment_info")},
        )
