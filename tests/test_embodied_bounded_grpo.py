from __future__ import annotations

from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest

from agent.training.embodied_bounded_grpo import BoundedEmbodiedGRPOTrainer
from plugins.embodied_swift_grpo import _align_rollout_output


class _Scheduler:
    def __init__(self, remaining: dict[str, int]):
        self.remaining = remaining
        self.allowed: dict[str, int] = {}

    def remaining_generation_tokens(self, request: SimpleNamespace) -> int:
        return self.remaining[request.request_id]

    def note_generation_allowance(self, request: SimpleNamespace, allowed: int) -> None:
        self.allowed[request.request_id] = allowed


def test_bounded_engine_groups_independent_budgets_and_restores_order() -> None:
    trainer = object.__new__(BoundedEmbodiedGRPOTrainer)
    trainer.samples2requests = lambda values: values
    calls: list[tuple[int, list[str]]] = []

    def infer(_self, requests, config, _adapter):
        calls.append((config.max_tokens, [request.request_id for request in requests]))
        return [f"output-{request.request_id}" for request in requests]

    trainer._engine_infer_with_adapter = MethodType(infer, trainer)
    requests = [SimpleNamespace(request_id="long"), SimpleNamespace(request_id="short")]
    scheduler = _Scheduler({"long": 1024, "short": 20})
    config = SimpleNamespace(max_tokens=1024)

    outputs = trainer._bounded_rollout_with_adapter(requests, config, scheduler, object())

    assert outputs == ["output-long", "output-short"]
    assert calls == [(20, ["short"]), (1024, ["long"])]
    assert scheduler.allowed == {"long": 1024, "short": 20}
    assert config.max_tokens == 1024


def test_rollout_session_sleeps_and_preserves_original_exception() -> None:
    trainer = object.__new__(BoundedEmbodiedGRPOTrainer)
    events: list[str] = []

    class Engine:
        def reset_prefix_cache(self):
            events.append("reset")

        def sleep(self, level: int):
            events.append(f"sleep-{level}")

        def wake_up(self, tags=None):
            events.append(f"wake-{tags}")

    trainer.args = SimpleNamespace(sleep_level=2)
    trainer.engine = SimpleNamespace(
        inner_model_executor=SimpleNamespace(is_sleeping=False), engine=Engine()
    )
    trainer.accelerator = SimpleNamespace(device="cpu")
    trainer.state = SimpleNamespace(global_step=3)
    trainer._last_loaded_step = 3
    trainer._move_model_to_vllm = lambda: events.append("sync")
    trainer.enable_offload = False
    trainer.offload_context = nullcontext

    with pytest.raises(LookupError, match="rollout failed"):
        with trainer._rollout_engine_session():
            raise LookupError("rollout failed")

    assert events[-2:] == ["reset", "sleep-2"]


def test_final_turn_alignment_is_idempotent() -> None:
    choice = SimpleNamespace(token_ids=[7, 8])
    item = SimpleNamespace(
        messages=[{"role": "assistant", "content": "DONE"}],
        response_token_ids=[],
        response_loss_mask=[],
        rollout_logprobs=[],
        response=SimpleNamespace(choices=[choice]),
    )
    scheduler = SimpleNamespace(_extract_logprobs_from_choice=lambda _choice: [])

    _align_rollout_output(scheduler, item)
    _align_rollout_output(scheduler, item)

    assert item.response_token_ids == [[7, 8]]
    assert len(item.response_loss_mask) == 1
