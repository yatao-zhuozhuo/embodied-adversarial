from __future__ import annotations

import pytest

from agent.training.embodied_rollout_budget import (
    CONTEXT_BUDGET_EXHAUSTED,
    TOKEN_BUDGET_EXHAUSTED,
    RolloutBudget,
    validate_rollout_alignment,
)


def test_allowance_is_minimum_of_three_independent_limits() -> None:
    budget = RolloutBudget(1024, 24576, 32768, generated_tokens_total=24556)
    decision = budget.decide(encoded_prompt_tokens=100)
    assert decision.allowed_tokens == 20
    budget.consume(20, allowed_tokens=decision.allowed_tokens)
    assert budget.remaining_generation_tokens == 0
    assert budget.termination_reason == TOKEN_BUDGET_EXHAUSTED


def test_context_exhaustion_never_creates_crash_guard_tokens() -> None:
    budget = RolloutBudget(1024, 24576, 32768)
    decision = budget.decide(encoded_prompt_tokens=32768)
    assert decision.allowed_tokens == 0
    assert decision.termination_reason == CONTEXT_BUDGET_EXHAUSTED


def test_engine_cannot_return_more_than_allowance() -> None:
    budget = RolloutBudget(1024, 24576, 32768)
    with pytest.raises(RuntimeError, match="allowance"):
        budget.consume(21, allowed_tokens=20)


def test_rollout_alignment_counts_final_turn_once() -> None:
    summary = validate_rollout_alignment(
        [[1, 2], [3]],
        [[1, 1], [1]],
        [[-0.1, -0.2], [-0.3]],
    )
    assert summary == {
        "generated_tokens_total": 3,
        "trained_tokens": 3,
        "rollout_logprob_tokens": 3,
        "rollout_logprobs_available": True,
    }


def test_rollout_alignment_rejects_mismatched_mask() -> None:
    with pytest.raises(RuntimeError, match="mask lengths"):
        validate_rollout_alignment([[1, 2]], [[1]], [])
