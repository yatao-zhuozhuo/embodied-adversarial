"""Host-owned goal checking boundary for Alice/Bob embodied tasks.

The checker is deliberately a protocol: simulator-specific state extraction and
predicate implementations stay in the simulator adapter, while the runtime owns
the result schema and never accepts an agent-declared success as truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
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


class PositionGoalChecker:
    """Check a compiled object-position predicate against trusted task state."""

    def evaluate(
        self,
        *,
        state: Mapping[str, Any],
        predicate: Mapping[str, Any],
    ) -> GoalEvaluation:
        predicate_type = str(predicate.get("type", ""))
        if predicate_type != "cube_at_position":
            raise ValueError(f"unsupported position predicate: {predicate_type!r}")
        actual = list(state.get("cube_position") or [])
        target = list(predicate.get("position") or [])
        if len(actual) != 3 or len(target) != 3:
            raise ValueError("cube_at_position requires actual and target xyz")
        tolerance = float(predicate.get("tolerance", 0.025))
        if tolerance <= 0:
            raise ValueError("position tolerance must be positive")
        distance = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(actual, target)))
        success = distance <= tolerance
        score = max(0.0, min(1.0, 1.0 - distance / max(tolerance * 4.0, 1e-9)))
        return GoalEvaluation(
            success=success,
            score=score,
            predicate_type=predicate_type,
            details={"distance": distance, "tolerance": tolerance},
        )
