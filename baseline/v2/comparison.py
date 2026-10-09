"""Fixed, small numerical tolerances for independent simulator instances."""
import math
import numpy as np

ROBOT_ABSOLUTE_TOLERANCE = 1e-7


def robot_equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(robot_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(robot_equal(x, y) for x, y in zip(a, b))
    if type(a) in (float, int) and type(b) in (float, int):
        return math.isfinite(a) and math.isfinite(b) and abs(a-b) <= ROBOT_ABSOLUTE_TOLERANCE
    return type(a) == type(b) and a == b


def pixel_match(a, b, benchmark):
    if a.shape != b.shape:
        return False
    delta = np.abs(a.astype(float)-b.astype(float))
    if benchmark == 'metaworld':
        return float(delta.mean()) <= 1.0
    # A few +/-1 uint8 rounding differences, not scene or object displacement.
    return float(delta.mean()) <= 0.001 and float(delta.max()) <= 1.0
