"""Small, auditable GRPO primitives for image-conditioned embodied rollouts.

This module deliberately contains no trainer framework dependency.  The smoke
trainer in :mod:`scripts.train_embodied_grpo` uses these functions with
Transformers and PEFT, which keeps the environment receipt and policy loss easy
to inspect before the same contract is moved to a distributed trainer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


ALLOWED_ACTIONS = (
    "MOVE",
    "MOVE_X_POS",
    "MOVE_X_NEG",
    "MOVE_Y_POS",
    "MOVE_Y_NEG",
    "MOVE_Z_POS",
    "MOVE_Z_NEG",
    "MOVE_X_POS_FINE",
    "MOVE_X_NEG_FINE",
    "MOVE_Y_POS_FINE",
    "MOVE_Y_NEG_FINE",
    "MOVE_Z_POS_FINE",
    "MOVE_Z_NEG_FINE",
    "GRASP",
    "RELEASE",
    "DONE",
)


@dataclass(slots=True)
class ParsedAction:
    action: str
    valid: bool
    reason: str = ""


@dataclass(slots=True)
class PolicyStep:
    """One generated completion and the trusted environment receipt it caused."""

    prompt: str
    image: np.ndarray = field(repr=False)
    completion_ids: list[int]
    completion: str
    action: str
    valid_action: bool
    environment_reward: float
    success: bool
    is_grasped: bool
    is_obj_placed: bool
    terminated: bool
    truncated: bool
    teacher_forced: bool = False
    teacher_action: str | None = None

    def artifact(self) -> dict[str, Any]:
        return {
            "completion": self.completion,
            "action": self.action,
            "valid_action": self.valid_action,
            "environment_reward": self.environment_reward,
            "success": self.success,
            "is_grasped": self.is_grasped,
            "is_obj_placed": self.is_obj_placed,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "teacher_forced": self.teacher_forced,
            "teacher_action": self.teacher_action,
        }


@dataclass(slots=True)
class PolicyEpisode:
    role: str
    reward: float
    success: bool
    steps: list[PolicyStep]

    def artifact(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "reward": self.reward,
            "success": self.success,
            "steps": [step.artifact() for step in self.steps],
        }


def parse_action(text: str) -> ParsedAction:
    """Parse a bounded action code, with backward compatibility for JSON."""

    raw = str(text or "").strip()
    direct = raw.upper()
    if direct in ALLOWED_ACTIONS:
        return ParsedAction(direct, True, "")
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        # Models occasionally wrap an otherwise valid object in prose.  Accept
        # only the first complete-looking object; execution is still bounded by
        # the action allowlist below.
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end <= start:
            return ParsedAction("DONE", False, "invalid JSON")
        try:
            payload = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return ParsedAction("DONE", False, "invalid JSON")
    if not isinstance(payload, dict):
        return ParsedAction("DONE", False, "action payload is not an object")
    action = str(payload.get("action", "")).upper().strip()
    if action not in ALLOWED_ACTIONS:
        return ParsedAction("DONE", False, f"unsupported action: {action or '<empty>'}")
    return ParsedAction(action, True, str(payload.get("reason", "")).strip())


def receipt_flag(info: dict[str, Any], key: str) -> bool:
    value = info.get(key, False)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else False
    return bool(value)


def episode_reward(steps: Iterable[PolicyStep], *, role: str) -> float:
    """Compute a trusted, bounded reward from simulator receipts.

    Bob gets the task success reward plus small progress shaping.  Alice mode is
    intentionally only a feasibility pretraining reward: adversarial Alice GRPO
    must replace it with the aggregate Bob boundary score from the curriculum.
    """

    records = list(steps)
    if not records:
        return -1.0
    success = any(step.is_obj_placed or step.success for step in records)
    grasped = any(step.is_grasped for step in records)
    valid_fraction = sum(step.valid_action for step in records) / len(records)
    dense = max(0.0, min(0.2, max(step.environment_reward for step in records)))
    invalid_penalty = 0.1 * (1.0 - valid_fraction)
    if role == "bob":
        reward = (1.0 if success else 0.0) + (0.15 if grasped else 0.0) + dense
    elif role == "alice":
        reward = (1.0 if success else 0.0) + (0.25 if grasped else 0.0) + dense
    else:
        raise ValueError(f"unsupported role: {role!r}")
    return max(-1.0, min(1.0, reward - invalid_penalty))


def group_advantages(rewards: Iterable[float], *, epsilon: float = 1e-4) -> list[float]:
    """Return population-standardized GRPO advantages for one prompt group."""

    values = np.asarray(list(rewards), dtype=np.float64)
    if values.size < 2:
        raise ValueError("GRPO requires at least two generations per prompt")
    return ((values - values.mean()) / (values.std(ddof=0) + epsilon)).astype(float).tolist()


def clipped_grpo_objective(
    current_logps: Any,
    old_logps: Any,
    reference_logps: Any,
    advantage: float,
    *,
    clip_epsilon: float,
    beta: float,
) -> Any:
    """Token-level clipped GRPO loss with the standard unbiased KL estimator."""

    import torch

    ratio = torch.exp(current_logps - old_logps)
    unclipped = ratio * float(advantage)
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * float(advantage)
    policy_loss = -torch.minimum(unclipped, clipped)
    log_ratio = reference_logps - current_logps
    per_token_kl = torch.exp(log_ratio) - log_ratio - 1.0
    return (policy_loss + float(beta) * per_token_kl).mean()
