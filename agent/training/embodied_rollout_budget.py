"""Pure request-level token budgeting for embodied multi-turn rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
CONTEXT_BUDGET_EXHAUSTED = "context_budget_exhausted"


@dataclass(frozen=True)
class BudgetDecision:
    """The generation allowance for one request at one environment turn."""

    allowed_tokens: int
    trajectory_remaining: int
    context_remaining: int
    termination_reason: str | None = None


@dataclass
class RolloutBudget:
    """Track generated tokens without counting repeated conversation history."""

    per_turn_limit: int
    trajectory_limit: int
    context_limit: int
    generated_tokens_total: int = 0
    turn_index: int = 0
    termination_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("per_turn_limit", "trajectory_limit", "context_limit"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.generated_tokens_total < 0 or self.turn_index < 0:
            raise ValueError("budget counters must be non-negative")

    def decide(self, encoded_prompt_tokens: int) -> BudgetDecision:
        """Return the exact next-turn allowance after final prompt encoding."""

        if encoded_prompt_tokens < 0:
            raise ValueError("encoded_prompt_tokens must be non-negative")
        trajectory_remaining = max(0, self.trajectory_limit - self.generated_tokens_total)
        context_remaining = max(0, self.context_limit - encoded_prompt_tokens)
        allowed = min(self.per_turn_limit, trajectory_remaining, context_remaining)
        reason = None
        if allowed <= 0:
            reason = (
                TOKEN_BUDGET_EXHAUSTED
                if trajectory_remaining <= 0
                else CONTEXT_BUDGET_EXHAUSTED
            )
        return BudgetDecision(allowed, trajectory_remaining, context_remaining, reason)

    def consume(self, generated_tokens: int, *, allowed_tokens: int) -> None:
        """Commit one sampled turn and reject engine/accounting violations."""

        if generated_tokens < 0 or allowed_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if generated_tokens > allowed_tokens:
            raise RuntimeError(
                f"engine generated {generated_tokens} tokens with allowance {allowed_tokens}"
            )
        self.generated_tokens_total += generated_tokens
        self.turn_index += 1
        if self.generated_tokens_total > self.trajectory_limit:
            raise RuntimeError("trajectory token budget was exceeded")
        if self.generated_tokens_total >= self.trajectory_limit:
            self.termination_reason = TOKEN_BUDGET_EXHAUSTED

    @property
    def remaining_generation_tokens(self) -> int:
        return max(0, self.trajectory_limit - self.generated_tokens_total)


def count_generated_tokens(turn_token_ids: Iterable[Iterable[int]]) -> int:
    """Count each assistant turn exactly once."""

    return sum(len(list(token_ids)) for token_ids in turn_token_ids)


def validate_rollout_alignment(
    response_token_ids: list[list[int]],
    response_loss_mask: list[list[int]],
    rollout_logprobs: list[list[float]],
) -> dict[str, int | bool]:
    """Validate the token/mask/logprob contract used by GRPO training."""

    if len(response_token_ids) != len(response_loss_mask):
        raise RuntimeError("response_token_ids and response_loss_mask turn counts differ")
    for index, (token_ids, loss_mask) in enumerate(zip(response_token_ids, response_loss_mask)):
        if len(token_ids) != len(loss_mask):
            raise RuntimeError(f"turn {index} token IDs and loss mask lengths differ")
    trained_tokens = sum(sum(int(value) for value in mask) for mask in response_loss_mask)
    logprob_tokens = sum(len(values) for values in rollout_logprobs)
    has_logprobs = bool(rollout_logprobs)
    if has_logprobs and trained_tokens != logprob_tokens:
        raise RuntimeError("trained token count and rollout logprob count differ")
    return {
        "generated_tokens_total": count_generated_tokens(response_token_ids),
        "trained_tokens": trained_tokens,
        "rollout_logprob_tokens": logprob_tokens,
        "rollout_logprobs_available": has_logprobs,
    }
