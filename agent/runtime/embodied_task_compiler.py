"""Compile Alice's trusted final simulator state into a replayable task."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from agent.runtime.embodied_snapshot import SnapshotRef


@dataclass(frozen=True, slots=True)
class CompiledTask:
    task_id: str
    instruction: str
    goal_predicate: dict[str, Any]
    goal_cell: str
    displacement: float
    valid: bool
    reason: str


def goal_cell(position: list[float], *, resolution: float = 0.05) -> str:
    if resolution <= 0:
        raise ValueError("goal cell resolution must be positive")
    indices = tuple(round(float(value) / resolution) for value in position)
    return ":".join(map(str, indices))


def compile_cube_position_task(
    *,
    snapshot: SnapshotRef,
    initial_state: Mapping[str, Any],
    final_state: Mapping[str, Any],
    min_displacement: float = 0.04,
    tolerance: float = 0.025,
) -> CompiledTask:
    initial = list(initial_state.get("cube_position") or [])
    final = list(final_state.get("cube_position") or [])
    if len(initial) != 3 or len(final) != 3:
        return CompiledTask("invalid", "", {}, "", 0.0, False, "missing cube position")
    displacement = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(initial, final)))
    rounded = [round(float(value), 3) for value in final]
    bounds_ok = -0.35 <= rounded[0] <= 0.35 and -0.35 <= rounded[1] <= 0.35 and 0.015 <= rounded[2] <= 0.45
    if displacement < min_displacement:
        valid, reason = False, "goal state is too close to initial state"
    elif not bounds_ok:
        valid, reason = False, "goal state is outside the supported workspace"
    else:
        valid, reason = True, "valid"
    predicate = {
        "type": "cube_at_position",
        "position": rounded,
        "tolerance": float(tolerance),
        "source": "maniskill_state",
    }
    digest = hashlib.sha256(json.dumps({
        "snapshot": snapshot.state_sha256,
        "predicate": predicate,
    }, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return CompiledTask(
        task_id=f"pickcube-{digest}",
        instruction="Move the red cube to the green target marker.",
        goal_predicate=predicate,
        goal_cell=goal_cell(rounded),
        displacement=displacement,
        valid=valid,
        reason=reason,
    )
