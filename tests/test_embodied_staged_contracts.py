import json
from copy import deepcopy

import pytest

from agent.training.embodied_staged_contracts import (
    AliceProposal,
    BobEvalReceipt,
    boundary_score,
    build_bob_eval_requests,
    finalize_alice_reward,
    preserve_rollout_num_turns,
)
from scripts.create_embodied_swift_dataset import _task_from_artifact
from scripts.summarize_embodied_formal_round import _alice_infos


def _proposal(**overrides):
    values = {
        "proposal_id": "round-0002-prompt-0003-sample-05",
        "alice_policy_version": "alice/checkpoint-12",
        "bob_policy_version": "bob/checkpoint-9",
        "snapshot_sha256": "abc123",
        "snapshot": {
            "snapshot_id": "snapshot-1",
            "env_id": "openeta/maniskill_PickCube-v1-v0",
            "state_uri": "/tmp/snapshot.pt",
            "state_sha256": "abc123",
            "seed": 31003,
            "metadata": {"env_id": "PickCube-v1"},
        },
        "goal_predicate": {
            "type": "cube_at_position",
            "position": [0.03, -0.04, 0.12],
            "tolerance": 0.025,
        },
        "compiled_valid": True,
        "replay_equal": True,
        "repeated": False,
        "target_success_rate": 0.5,
        "alice_rollout_info": {
            "schema_version": "openeta.embodied_rollout.v1",
            "role": "alice",
            "reward": 0.0,
            "reward_stage": "awaiting_frozen_bob",
            "reward_pending": True,
            "format_penalty": 0.1,
            "bob_evaluations": 2,
            "bob_max_steps": 32,
        },
    }
    values.update(overrides)
    return AliceProposal(**values)


def _receipt(proposal, index, *, success=False, **overrides):
    values = {
        "evaluation_id": f"{proposal.proposal_id}-bob-eval-{index:02d}",
        "proposal_id": proposal.proposal_id,
        "evaluation_index": index,
        "bob_policy_version": proposal.bob_policy_version,
        "snapshot_sha256": proposal.snapshot_sha256,
        "goal_predicate": deepcopy(proposal.goal_predicate),
        "success": success,
        "reward": 1.0 if success else 0.0,
        "step_count": 12,
    }
    values.update(overrides)
    return BobEvalReceipt(**values)


def test_build_bob_requests_is_deterministic_and_complete():
    proposal = _proposal()
    requests = build_bob_eval_requests(proposal, 2)
    assert [request.evaluation_index for request in requests] == [0, 1]
    assert [request.seed for request in requests] == [31003, 31004]
    assert all(request.proposal_id == proposal.proposal_id for request in requests)


def test_finalize_reward_uses_boundary_novelty_and_format_formula():
    proposal = _proposal()
    result = finalize_alice_reward(
        proposal,
        [_receipt(proposal, 0, success=True), _receipt(proposal, 1)],
    )
    # success_rate == target => boundary 1.0; +0.25 novelty -0.1 format,
    # then clamp to the legacy reward range.
    assert result["reward"] == 1.0
    assert result["reward_stage"] == "frozen_bob_boundary"
    assert result["bob_evaluation"]["success_rate"] == 0.5
    assert result["reward_pending"] is False


def test_repeated_proposal_uses_same_legacy_formula():
    proposal = _proposal(repeated=True, target_success_rate=0.45)
    result = finalize_alice_reward(
        proposal,
        [_receipt(proposal, 0), _receipt(proposal, 1)],
    )
    expected = boundary_score(0.0, 0.45) - 0.15 - 0.1
    assert result["reward"] == pytest.approx(expected)


def test_receipt_infrastructure_error_is_neutral_before_format_penalty():
    proposal = _proposal()
    result = finalize_alice_reward(
        proposal,
        [
            _receipt(proposal, 0, infrastructure_error="simulator unavailable"),
            _receipt(proposal, 1),
        ],
    )
    assert result["reward"] == pytest.approx(-0.1)
    assert result["reward_stage"] == "bob_evaluator_error"
    assert "simulator unavailable" in result["bob_evaluator_error"]


def test_mismatched_policy_version_is_rejected_as_evaluator_error():
    proposal = _proposal()
    result = finalize_alice_reward(
        proposal,
        [
            _receipt(proposal, 0, bob_policy_version="wrong/checkpoint"),
            _receipt(proposal, 1),
        ],
    )
    assert result["reward_stage"] == "bob_evaluator_error"
    assert "policy version mismatch" in result["bob_evaluator_error"]


def test_invalid_proposal_never_requires_bob_receipts():
    info = deepcopy(_proposal().alice_rollout_info)
    info.update({
        "reward": -0.7,
        "reward_stage": "invalid_proposal_shaping",
        "reward_pending": False,
    })
    proposal = _proposal(
        compiled_valid=False,
        goal_predicate={},
        alice_rollout_info=info,
    )
    assert build_bob_eval_requests(proposal, 2) == []
    result = finalize_alice_reward(proposal, [])
    assert result["reward"] == -0.7
    assert result["reward_stage"] == "invalid_proposal_shaping"


def test_staged_reward_finalization_preserves_scalar_num_turns():
    finalized = preserve_rollout_num_turns(
        {"role": "alice", "step_count": 23, "reward": 0.5},
        {"role": "alice", "num_turns": 23},
    )
    assert finalized["num_turns"] == 23
    assert isinstance(finalized["num_turns"], int)


def test_staged_reward_finalization_rejects_non_scalar_num_turns():
    with pytest.raises(ValueError, match="positive integer"):
        preserve_rollout_num_turns(
            {"role": "alice", "step_count": 23},
            {"role": "alice", "num_turns": {"rank": 0}},
        )


def test_staged_proposal_can_feed_next_round_bob_dataset(tmp_path):
    proposal = _proposal()
    proposal.alice_rollout_info["compiled_task"] = {
        "task_id": "alice-task-1",
        "valid": True,
        "goal_predicate": deepcopy(proposal.goal_predicate),
    }
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal.to_dict()), encoding="utf-8")
    snapshot, compiled = _task_from_artifact(path)
    assert snapshot["state_sha256"] == proposal.snapshot_sha256
    assert compiled["task_id"] == "alice-task-1"


def test_round_summary_prefers_finalized_staged_proposal(tmp_path):
    pending = deepcopy(_proposal().alice_rollout_info)
    pending.update({"role": "alice", "proposal_id": "proposal-1", "reward": 0.0})
    finalized = deepcopy(pending)
    finalized.update({"reward_pending": False, "reward": 0.75})
    (tmp_path / "rollout.json").write_text(
        json.dumps({"final_info": pending}), encoding="utf-8"
    )
    (tmp_path / "proposal.json").write_text(
        json.dumps({"alice_rollout_info": finalized}), encoding="utf-8"
    )
    infos = _alice_infos(tmp_path)
    assert len(infos) == 1
    assert infos[0]["reward"] == 0.75
