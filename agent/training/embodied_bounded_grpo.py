"""Shared bounded rollout orchestration for Alice and Bob GRPO training."""

from __future__ import annotations

import inspect
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext, suppress
from copy import deepcopy
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

import torch
from accelerate.utils import gather_object
from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest, RolloutOutput
from swift.rlhf_trainers import GRPOTrainer
from swift.rlhf_trainers.utils import aggressive_empty_cache, set_expandable_segments
from swift.rl_core.data import OnPolicySample
from swift.rollout import invoke_async_hook, run_multi_turn

from agent.training.embodied_rollout_budget import validate_rollout_alignment
from plugins.embodied_swift_grpo import _align_rollout_output, _find_rollout_info


CURRENT_LORA_ID = 1
ROLE_LORA_NAMES = {
    "alice": "openeta_current_alice",
    "bob": "openeta_current_bob",
}


def register_bounded_trainer(role: str) -> None:
    """Register the role-specific trainer without patching site-packages."""

    role = role.strip().lower()
    if role not in ROLE_LORA_NAMES:
        raise ValueError("OPENETA_TRAIN_ROLE must be alice or bob")
    from swift.infer_engine import grpo_vllm_engine
    from swift.rlhf_trainers import rollout_mixin
    from swift.trainers.trainer_factory import TrainerFactory

    lora_name = ROLE_LORA_NAMES[role]
    rollout_mixin.VLLM_LORA_INT_ID = CURRENT_LORA_ID
    rollout_mixin.VLLM_LORA_NAME = lora_name
    grpo_vllm_engine.VLLM_LORA_INT_ID = CURRENT_LORA_ID
    grpo_vllm_engine.VLLM_LORA_NAME = lora_name
    trainer = (
        "agent.training.embodied_staged_colocate.StagedEmbodiedGRPOTrainer"
        if role == "alice"
        else "agent.training.embodied_bounded_grpo.BoundedEmbodiedGRPOTrainer"
    )
    TrainerFactory.TRAINER_MAPPING["grpo"] = trainer
    print(
        "[openeta] bounded embodied trainer: "
        f"role={role} trainer={trainer} per_turn={os.environ.get('OPENETA_THINKING_MAX_TOKENS', '1024')} "
        f"trajectory={os.environ.get('OPENETA_TRAJECTORY_MAX_TOKENS', '24576')} "
        f"context={os.environ.get('OPENETA_CONTEXT_MAX_TOKENS', '32768')} "
        f"history={os.environ.get('OPENETA_HISTORY_MODE', 'full')}"
    )


class BoundedEmbodiedGRPOTrainer(GRPOTrainer):
    """Stock GRPO optimization with bounded, request-aware embodied rollout."""

    def _prepare_scheduler(self) -> None:
        super()._prepare_scheduler()
        if self.args.vllm_mode != "colocate":
            raise ValueError("bounded embodied training requires --vllm_mode colocate")
        if self.args.vllm_tensor_parallel_size != 1:
            raise ValueError("bounded embodied training requires --vllm_tensor_parallel_size 1")
        if getattr(self.args, "async_generate", False):
            raise ValueError("bounded embodied collectives are incompatible with --async_generate")
        self._train_role = os.environ.get("OPENETA_TRAIN_ROLE", "").strip().lower()
        if self._train_role not in ROLE_LORA_NAMES:
            raise ValueError("OPENETA_TRAIN_ROLE must be alice or bob")
        scheduler_role = getattr(self.multi_turn_scheduler, "role", None)
        if scheduler_role != self._train_role:
            raise ValueError(
                f"role={self._train_role} cannot use scheduler role={scheduler_role!r}"
            )
        history_mode = os.environ.get("OPENETA_HISTORY_MODE", "full").strip().lower()
        if history_mode != "full":
            raise ValueError("the first bounded rollout implementation requires OPENETA_HISTORY_MODE=full")
        configured_completion_limit = int(self.max_completion_length)
        effective_context_limit = min(
            int(getattr(self.args, "max_length", 32768) or 32768),
            int(getattr(self.args, "vllm_max_model_len", 32768) or 32768),
        )
        self._trajectory_limit = int(
            os.environ.get("OPENETA_TRAJECTORY_MAX_TOKENS", self.max_completion_length)
        )
        self._context_limit = int(
            os.environ.get(
                "OPENETA_CONTEXT_MAX_TOKENS",
                effective_context_limit,
            )
        )
        if self._trajectory_limit <= 0 or self._context_limit <= 0:
            raise ValueError("trajectory/context token limits must be positive")
        if self._trajectory_limit > configured_completion_limit:
            raise ValueError("OPENETA_TRAJECTORY_MAX_TOKENS exceeds --max_completion_length")
        if self._context_limit > effective_context_limit:
            raise ValueError("OPENETA_CONTEXT_MAX_TOKENS exceeds the effective engine context")
        self._bounded_group_index = 0
        self._train_forward_seconds = 0.0
        self._optimizer_window: list[dict[str, float]] = []
        configured_root = os.environ.get("OPENETA_STAGED_ARTIFACT_ROOT")
        self._bounded_artifact_root = Path(
            configured_root or Path(self.args.output_dir).resolve().parent
        ).resolve()

    def compute_loss(
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        started = time.monotonic()
        try:
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        finally:
            self._train_forward_seconds += time.monotonic() - started

    def training_step(
        self,
        model: Any,
        inputs: dict[str, Any],
        num_items_in_batch: Any = None,
    ) -> torch.Tensor:
        self._train_forward_seconds = 0.0
        started = time.monotonic()
        result = super().training_step(model, inputs, num_items_in_batch)
        total_seconds = time.monotonic() - started
        self._optimizer_window.append({
            "train_forward_seconds": self._train_forward_seconds,
            "train_backward_seconds": max(0.0, total_seconds - self._train_forward_seconds),
        })
        return result

    def create_optimizer(self, model: Any = None) -> Any:
        optimizer = super().create_optimizer(model)
        if getattr(optimizer, "_openeta_timing_wrapped", False):
            return optimizer
        original_step = optimizer.step
        trainer = self

        def timed_step(_optimizer: Any, *args: Any, **kwargs: Any) -> Any:
            started = time.monotonic()
            try:
                return original_step(*args, **kwargs)
            finally:
                window, trainer._optimizer_window = trainer._optimizer_window, []
                trainer._append_phase_timing({
                    "event": "training",
                    "role": trainer._train_role,
                    "global_step": trainer.state.global_step,
                    "group_index": max(0, trainer._bounded_group_index - 1),
                    "rank": trainer.accelerator.process_index,
                    "micro_steps": len(window),
                    "train_forward_seconds": sum(x["train_forward_seconds"] for x in window),
                    "train_backward_seconds": sum(x["train_backward_seconds"] for x in window),
                    "optimizer_step_seconds": time.monotonic() - started,
                    **trainer._cuda_memory("after_optimizer"),
                })

        optimizer.step = MethodType(timed_step, optimizer)
        optimizer._openeta_timing_wrapped = True
        return optimizer

    def _current_adapter_request(self) -> Any:
        if not self.rollout_enable_lora:
            raise ValueError("bounded embodied routing requires --vllm_enable_lora true")
        from swift.rlhf_trainers import rollout_mixin
        from vllm.lora.request import LoRARequest

        return LoRARequest(
            lora_name=ROLE_LORA_NAMES[self._train_role],
            lora_int_id=CURRENT_LORA_ID,
            lora_path=rollout_mixin.VLLM_LORA_PATH,
        )

    def _engine_infer_with_adapter(
        self,
        infer_requests: list[RolloutInferRequest],
        request_config: RequestConfig,
        adapter_request: Any,
    ) -> list[RolloutOutput]:
        with self._disable_sp_context():
            return self.engine.infer(
                infer_requests,
                request_config,
                use_tqdm=False,
                adapter_request=adapter_request,
            )

    def _bounded_rollout_with_adapter(
        self,
        requests_or_samples: list[OnPolicySample] | list[RolloutInferRequest],
        request_config: RequestConfig,
        scheduler: Any,
        adapter_request: Any,
    ) -> list[RolloutOutput]:
        requests = self.samples2requests(requests_or_samples)
        if not requests:
            return self._engine_infer_with_adapter(requests, request_config, adapter_request)
        indexed_groups: dict[int, list[tuple[int, RolloutInferRequest]]] = defaultdict(list)
        for index, request in enumerate(requests):
            remaining = int(scheduler.remaining_generation_tokens(request))
            allowed = min(int(request_config.max_tokens), remaining)
            if allowed <= 0:
                raise RuntimeError(
                    f"request {getattr(request, 'request_id', index)!r} reached zero budget before scheduler finish"
                )
            scheduler.note_generation_allowance(request, allowed)
            indexed_groups[allowed].append((index, request))
        ordered: list[RolloutOutput | None] = [None] * len(requests)
        for allowed, group in sorted(indexed_groups.items()):
            bounded = deepcopy(request_config)
            bounded.max_tokens = allowed
            outputs = self._engine_infer_with_adapter(
                [request for _, request in group], bounded, adapter_request
            )
            if len(outputs) != len(group):
                raise RuntimeError("engine output count differs from bounded request group")
            for (index, _), output in zip(group, outputs):
                ordered[index] = output
        if any(output is None for output in ordered):
            raise RuntimeError("bounded rollout failed to restore output order")
        return [output for output in ordered if output is not None]

    def _run_bounded_scheduler_phase(
        self,
        samples: list[OnPolicySample],
        scheduler: Any,
        adapter_request: Any,
    ) -> list[OnPolicySample]:
        requests = self.samples2requests(samples)
        invoke_async_hook(scheduler.on_trajectory_start(requests))
        for request, sample in zip(requests, samples):
            sample.messages = request.messages
        request_config = scheduler.prepare_request_config(self._get_request_config())
        first_outputs = self._bounded_rollout_with_adapter(
            requests, request_config, scheduler, adapter_request
        )
        rollout_outputs = run_multi_turn(
            requests=requests,
            first_turn_outputs=first_outputs,
            scheduler=scheduler,
            rollout_fn=lambda reqs, cfg: self._bounded_rollout_with_adapter(
                reqs, cfg, scheduler, adapter_request
            ),
            request_config=request_config,
            max_turns=self.args.max_turns,
            gather_fn=gather_object,
        )
        for output in rollout_outputs:
            _align_rollout_output(scheduler, output)
        outputs = self._postprocess_rollout_outputs(samples, rollout_outputs)
        self._annotate_and_validate_rollouts(outputs)
        return outputs

    def _annotate_and_validate_rollouts(self, samples: list[OnPolicySample]) -> None:
        per_turn_limit = int(self.multi_turn_scheduler.per_turn_token_limit)
        for sample in samples:
            token_ids = list(sample.response_token_ids or [])
            for index, turn_ids in enumerate(token_ids):
                if len(turn_ids) > per_turn_limit:
                    raise RuntimeError(
                        f"turn {index} generated {len(turn_ids)} tokens; limit is {per_turn_limit}"
                    )
            summary = validate_rollout_alignment(
                token_ids,
                list(sample.response_loss_mask or []),
                list(sample.rollout_logprobs or []),
            )
            if int(summary["generated_tokens_total"]) > self._trajectory_limit:
                raise RuntimeError("postprocessed rollout exceeded trajectory token budget")
            info = _find_rollout_info(sample.rollout_infos)
            if info is not None:
                info["rollout_budget"] = {
                    "schema_version": "openeta.rollout_budget.v1",
                    "per_turn_limit": per_turn_limit,
                    "trajectory_limit": self._trajectory_limit,
                    "context_limit": self._context_limit,
                    **summary,
                }

    def _collect_role_rollouts(self, samples: list[OnPolicySample]) -> list[OnPolicySample]:
        return self._run_bounded_scheduler_phase(
            samples, self.multi_turn_scheduler, self._current_adapter_request()
        )

    @staticmethod
    def _cuda_memory(stage: str, device: Any = None) -> dict[str, int]:
        if not torch.cuda.is_available():
            return {f"gpu_{stage}_{name}_bytes": 0 for name in ("allocated", "reserved", "peak")}
        device = device if device is not None else torch.cuda.current_device()
        return {
            f"gpu_{stage}_allocated_bytes": torch.cuda.memory_allocated(device),
            f"gpu_{stage}_reserved_bytes": torch.cuda.memory_reserved(device),
            f"gpu_{stage}_peak_bytes": torch.cuda.max_memory_allocated(device),
        }

    @contextmanager
    def _rollout_engine_session(self) -> Iterator[dict[str, Any]]:
        args = self.args
        device = self.accelerator.device
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        timing: dict[str, Any] = self._cuda_memory("before_wake", device)
        original_error: BaseException | None = None
        try:
            started = time.monotonic()
            if args.sleep_level > 0 and self.engine.inner_model_executor.is_sleeping:
                wake_kwargs = {}
                if "tags" in inspect.signature(self.engine.engine.wake_up).parameters:
                    wake_kwargs = {"tags": ["weights"]}
                aggressive_empty_cache()
                self.engine.engine.wake_up(**wake_kwargs)
            timing["vllm_wake_seconds"] = time.monotonic() - started
            timing.update(self._cuda_memory("after_wake", device))

            started = time.monotonic()
            if self.state.global_step != self._last_loaded_step or args.sleep_level == 2:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            timing["weight_sync_seconds"] = time.monotonic() - started

            context = self.offload_context if self.enable_offload else nullcontext
            with context():
                if (
                    self.engine.inner_model_executor.is_sleeping
                    and "tags" in inspect.signature(self.engine.engine.wake_up).parameters
                ):
                    aggressive_empty_cache()
                    set_expandable_segments(False)
                    self.engine.engine.wake_up(tags=["kv_cache"])
                yield timing
                timing.update(self._cuda_memory("after_rollout", device))
        except BaseException as exc:
            original_error = exc
            timing["rollout_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            started = time.monotonic()
            try:
                if args.sleep_level > 0:
                    self.engine.engine.reset_prefix_cache()
                    self.engine.engine.sleep(level=args.sleep_level)
                    aggressive_empty_cache()
                    set_expandable_segments(True)
            except BaseException as cleanup_error:
                timing["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
                if original_error is None:
                    raise
            finally:
                timing["vllm_sleep_seconds"] = time.monotonic() - started
                timing["vllm_is_sleeping_after"] = bool(
                    getattr(self.engine.inner_model_executor, "is_sleeping", False)
                )
                timing.update(self._cuda_memory("after_sleep", device))
                if original_error is not None:
                    timing.update({
                        "event": "rollout_error",
                        "role": getattr(self, "_train_role", "unknown"),
                        "global_step": self.state.global_step,
                        "group_index": getattr(self, "_bounded_group_index", 0),
                        "rank": getattr(self.accelerator, "process_index", 0),
                    })
                    with suppress(Exception):
                        self._append_phase_timing(timing)

    def _append_phase_timing(self, timing: dict[str, Any]) -> None:
        path = self._bounded_artifact_root / (
            f"phase_timing.rank-{self.accelerator.process_index:02d}.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(timing, ensure_ascii=False) + "\n")

    def _after_rollout(
        self,
        outputs: list[OnPolicySample],
        timing: dict[str, Any],
    ) -> None:
        del outputs
        self._append_phase_timing(timing)

    def _fast_infer(self, samples: list[OnPolicySample]) -> list[OnPolicySample]:
        started = time.monotonic()
        with self._rollout_engine_session() as timing:
            outputs = self._collect_role_rollouts(samples)
        timing.update({
            "event": "rollout",
            "role": self._train_role,
            "global_step": self.state.global_step,
            "group_index": self._bounded_group_index,
            "rank": self.accelerator.process_index,
            "rollout_seconds": time.monotonic() - started,
            "trajectories": len(outputs),
        })
        self._after_rollout(outputs, timing)
        self._bounded_group_index += 1
        return outputs
