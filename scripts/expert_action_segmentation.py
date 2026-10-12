"""Budgeted action grouping that preserves gripper transitions and nearby steps."""

import numpy as np


def budgeted_segments(actions, turn_budget, protection_radius=3, drift_weight=2.0):
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or not len(values) or not np.isfinite(values).all():
        raise ValueError("expected finite seven-dimensional LIBERO actions")
    if turn_budget < 1 or protection_radius < 0 or drift_weight < 0:
        raise ValueError("invalid segmentation parameters")
    count = len(values)
    groups = min(count, turn_budget)
    protected = np.zeros(count, dtype=bool)
    for index in range(1, count):
        if values[index, -1] != values[index - 1, -1]:
            protected[max(0, index - protection_radius):min(count, index + protection_radius + 1)] = True
    costs = np.full((count + 1, 11), np.inf)
    weights = np.array([1, 1, 1, 0.5, 0.5, 0.5])
    for end in range(1, count + 1):
        for length in range(1, min(10, end) + 1):
            start = end - length
            window = values[start:end]
            if length > 1 and (protected[start:end].any() or np.any(window[:, -1] != window[0, -1])):
                continue
            deviation = window[:, :6] - window[:, :6].mean(axis=0)
            drift = np.cumsum(deviation, axis=0)
            costs[end, length] = float(np.sum(deviation ** 2 * weights) + drift_weight * np.sum(drift ** 2 * weights))
    score = np.full((groups + 1, count + 1), np.inf)
    previous = np.zeros((groups + 1, count + 1), dtype=np.int16)
    score[0, 0] = 0
    for group in range(1, groups + 1):
        for end in range(group, min(count, group * 10) + 1):
            lengths = np.arange(1, min(10, end - group + 1) + 1)
            candidates = score[group - 1, end - lengths] + costs[end, lengths]
            best = int(np.argmin(candidates))
            score[group, end] = candidates[best]
            previous[group, end] = lengths[best]
    if not np.isfinite(score[groups, count]):
        raise ValueError("turn budget cannot preserve protected actions and ten-step maximum")
    segments = []
    end = count
    for group in range(groups, 0, -1):
        length = int(previous[group, end])
        window = values[end - length:end]
        segments.append({"action": window.mean(axis=0).tolist(), "num_steps": length})
        end -= length
    return list(reversed(segments))
