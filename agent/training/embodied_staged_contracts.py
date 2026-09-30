"""Serializable contracts for staged embodied Alice/Bob evaluation.

The functions in this module deliberately do not import ManiSkill, vLLM, or
ms-swift.  They are the trust boundary between Alice proposal generation and
the frozen-Bob evaluation phase and can therefore be parity-tested with small
JSON fixtures.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


ALICE_PROPOSAL_SCHEMA = "openeta.alice_proposal.v1"
BOB_EVAL_REQUEST_SCHEMA = "openeta.bob_eval_request.v1"
BOB_EVAL_RECEIPT_SCHEMA = "openeta.bob_eval_receipt.v1"


@dataclass(slots=True)
class AliceProposal:
    proposal_id: str
    alice_policy_version: str
    bob_policy_version: str
    snapshot_sha256: str
    snapshot: dict[str, Any]
    goal_predicate: dict[str, Any]
    compiled_valid: bool
    replay_equal: bool
    repeated: bool
    target_success_rate: float
    alice_rollout_info: dict[str, Any]
    alice_response_token_ids: list[list[int]] = field(default_factory=list)
    alice_response_loss_mask: list[list[int]] = field(default_factory=list)
    alice_rollout_logprobs: list[list[float]] = field(default_factory=list)
    schema_version: str = ALICE_PROPOSAL_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ALICE_PROPOSAL_SCHEMA:
            raise ValueError(f"unsupported Alice proposal schema: {self.schema_version}")
        if not self.proposal_id:
            raise ValueError("proposal_id must not be empty")
        if not 0.0 < float(self.target_success_rate) < 1.0:
            raise ValueError("target_success_rate must be in (0, 1)")
        if self.compiled_valid and not self.goal_predicate:
            raise ValueError("a valid proposal requires goal_predicate")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AliceProposal":
        return cls(**deepcopy(value))


@dataclass(slots=True)
class BobEvalRequest:
    evaluation_id: str
    proposal_id: str
    evaluation_index: int
    bob_policy_version: str
    snapshot_sha256: str
    snapshot: dict[str, Any]
    goal_predicate: dict[str, Any]
    max_steps: int
    seed: int
    schema_version: str = BOB_EVAL_REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BOB_EVAL_REQUEST_SCHEMA:
            raise ValueError(f"unsupported Bob request schema: {self.schema_version}")
        if self.evaluation_index < 0:
            raise ValueError("evaluation_index must be non-negative")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BobEvalRequest":
        return cls(**deepcopy(value))


@dataclass(slots=True)
class BobEvalReceipt:
    evaluation_id: str
    proposal_id: str
    evaluation_index: int
    bob_policy_version: str
    snapshot_sha256: str
    goal_predicate: dict[str, Any]
    success: bool
    reward: float
    step_count: int
    infrastructure_error: str | None = None
    role: str = "bob"
    phase: str = "bob_evaluation"
    adapter_name: str = "openeta_frozen_bob"
    rank: int = 0
    schema_version: str = BOB_EVAL_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BOB_EVAL_RECEIPT_SCHEMA:
            raise ValueError(f"unsupported Bob receipt schema: {self.schema_version}")
        if self.evaluation_index < 0:
            raise ValueError("evaluation_index must be non-negative")
        if self.step_count < 0:
            raise ValueError("step_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BobEvalReceipt":
        return cls(**deepcopy(value))


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def boundary_score(success_rate: float, target_success_rate: float) -> float:
    """Score tasks highest at Bob's configured competence boundary."""

    if not 0.0 <= success_rate <= 1.0:
        raise ValueError("success_rate must be in [0, 1]")
    if not 0.0 < target_success_rate < 1.0:
        raise ValueError("target_success_rate must be in (0, 1)")
    scale = max(target_success_rate, 1.0 - target_success_rate)
    return max(0.0, 1.0 - abs(success_rate - target_success_rate) / scale)


def build_bob_eval_requests(
    proposal: AliceProposal,
    evaluations: int,
) -> list[BobEvalRequest]:
    """Expand one valid proposal into deterministic frozen-Bob requests."""

    if evaluations < 1:
        raise ValueError("evaluations must be positive")
    if not proposal.compiled_valid:
        return []
    snapshot_seed = proposal.snapshot.get("seed")
    base_seed = int(snapshot_seed) if snapshot_seed is not None else 0
    max_steps = int(proposal.alice_rollout_info.get("bob_max_steps", 32))
    return [
        BobEvalRequest(
            evaluation_id=f"{proposal.proposal_id}-bob-eval-{index:02d}",
            proposal_id=proposal.proposal_id,
            evaluation_index=index,
            bob_policy_version=proposal.bob_policy_version,
            snapshot_sha256=proposal.snapshot_sha256,
            snapshot=deepcopy(proposal.snapshot),
            goal_predicate=deepcopy(proposal.goal_predicate),
            max_steps=max_steps,
            seed=base_seed + index,
        )
        for index in range(evaluations)
    ]


def validate_bob_receipts(
    proposal: AliceProposal,
    receipts: Iterable[BobEvalReceipt],
    *,
    evaluations: int,
) -> list[BobEvalReceipt]:
    """Return deterministically ordered receipts or raise on trust violations."""

    materialized = list(receipts)
    if len(materialized) != evaluations:
        raise ValueError(f"expected {evaluations} Bob receipts, got {len(materialized)}")
    by_index: dict[int, BobEvalReceipt] = {}
    for receipt in materialized:
        if receipt.proposal_id != proposal.proposal_id:
            raise ValueError("Bob receipt proposal_id mismatch")
        expected_id = f"{proposal.proposal_id}-bob-eval-{receipt.evaluation_index:02d}"
        if receipt.evaluation_id != expected_id:
            raise ValueError("Bob receipt evaluation_id mismatch")
        if receipt.evaluation_index in by_index:
            raise ValueError("duplicate Bob evaluation_index")
        if receipt.bob_policy_version != proposal.bob_policy_version:
            raise ValueError("Bob receipt policy version mismatch")
        if receipt.snapshot_sha256 != proposal.snapshot_sha256:
            raise ValueError("Bob receipt snapshot hash mismatch")
        if _canonical(receipt.goal_predicate) != _canonical(proposal.goal_predicate):
            raise ValueError("Bob receipt goal predicate mismatch")
        by_index[receipt.evaluation_index] = receipt
    expected_indexes = set(range(evaluations))
    if set(by_index) != expected_indexes:
        raise ValueError("Bob receipt evaluation indexes are incomplete")
    return [by_index[index] for index in range(evaluations)]


def finalize_alice_reward(
    proposal: AliceProposal,
    receipts: Iterable[BobEvalReceipt],
    *,
    evaluator_error: str | None = None,
) -> dict[str, Any]:
    """Finalize trusted Alice reward while preserving the server-mode formula.

    Invalid proposals have already received feasibility shaping in
    ``alice_rollout_info`` and must never launch Bob.  Any missing, malformed,
    or infrastructure-error receipt is treated as an evaluator outage, not as
    a Bob policy failure.
    """

    result = deepcopy(proposal.alice_rollout_info)
    result["reward_pending"] = False
    if not proposal.compiled_valid:
        result.setdefault("bob_evaluation", None)
        result.setdefault("bob_evaluator_error", None)
        return result

    evaluations = int(result.get("bob_evaluations", 0))
    if evaluations < 1:
        raise ValueError("proposal alice_rollout_info must declare bob_evaluations")
    ordered: list[BobEvalReceipt] = []
    error = evaluator_error
    if error is None:
        try:
            ordered = validate_bob_receipts(proposal, receipts, evaluations=evaluations)
            infrastructure_errors = [
                receipt.infrastructure_error for receipt in ordered if receipt.infrastructure_error
            ]
            if infrastructure_errors:
                error = "; ".join(str(item) for item in infrastructure_errors)
        except Exception as exc:  # validation failures are evaluator failures
            error = f"{type(exc).__name__}: {exc}"

    format_penalty = float(result.get("format_penalty", 0.0))
    if error is not None:
        # This is identical to the legacy synchronous path: evaluator outage is
        # neutral before the independently computed format penalty is applied.
        reward = -format_penalty
        result.update({
            "reward": max(-1.0, min(1.0, reward)),
            "reward_stage": "bob_evaluator_error",
            "bob_evaluation": None,
            "bob_evaluator_error": error,
            "bob_eval_receipts": [receipt.to_dict() for receipt in ordered],
        })
        return result

    successes = sum(bool(receipt.success) for receipt in ordered)
    success_rate = successes / len(ordered)
    difficulty = boundary_score(success_rate, proposal.target_success_rate)
    novelty = 0.0 if proposal.repeated else 1.0
    repetition_penalty = 0.15 if proposal.repeated else 0.0
    reward = difficulty + 0.25 * novelty - repetition_penalty - format_penalty
    result.update({
        "reward": max(-1.0, min(1.0, reward)),
        "reward_stage": "frozen_bob_boundary",
        "bob_evaluation": {
            "policy_version": proposal.bob_policy_version,
            "episodes": len(ordered),
            "successes": successes,
            "success_rate": success_rate,
            "mean_reward": sum(float(receipt.reward) for receipt in ordered) / len(ordered),
            "request_ids": [receipt.evaluation_id for receipt in ordered],
        },
        "bob_evaluator_error": None,
        "bob_eval_receipts": [receipt.to_dict() for receipt in ordered],
    })
    return result


def preserve_rollout_num_turns(
    final_info: dict[str, Any],
    source_rollout_info: dict[str, Any],
) -> dict[str, Any]:
    """Keep Swift's scalar multi-turn metadata across staged reward finalization.

    ``run_multi_turn`` adds ``num_turns`` to the outer RolloutOutput only after
    the Alice scheduler has built its proposal.  A valid proposal therefore
    does not carry that field inside ``alice_rollout_info``.  Replacing the
    sample metadata with the finalized proposal reward used to drop the field
    on valid-proposal ranks while invalid-proposal ranks retained it.  Swift's
    rank-local presence check then desynchronized its distributed gather.
    """

    result = deepcopy(final_info)
    value = source_rollout_info.get("num_turns", result.get("step_count"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("rollout num_turns must be a positive integer")
    result["num_turns"] = value
    return result
