"""Map ManiSkill pd_ee_delta_pose demo actions to the strict OpenETA contract."""

from __future__ import annotations

import numpy as np


def canonicalize_pd_ee_delta_pose(action: list[float]) -> list[float]:
    """Preserve ManiSkill controller semantics for out-of-range 7D actions.

    Its position and gripper controllers clip components to [-1, 1], while
    PDEEPosPoseController clips the *norm* of the 3D rotation vector. Clipping
    rotation components independently changes the requested rotation axis.
    """
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("ManiSkill pd_ee_delta_pose requires a finite 7D action")
    result = np.clip(values, -1.0, 1.0)
    rotation = values[3:6]
    length = float(np.linalg.norm(rotation))
    result[3:6] = rotation / max(1.0, length)
    return result.tolist()
