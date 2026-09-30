"""Three-phase Alice rollout / frozen-Bob evaluation / GRPO trainer.

Only rollout construction is customized.  Reward normalization, GRPO
advantages, reference-policy/KL handling, loss masking, backward, gradient
accumulation, and optimizer updates remain in ms-swift's ``GRPOTrainer``.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from accelerate.utils import gather_object
from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest, RolloutOutput
from swift.rlhf_trainers import GRPOTrainer
from swift.rlhf_trainers.utils import aggressive_empty_cache, set_expandable_segments
from swift.rl_core.data import OnPolicySample
from swift.rollout import invoke_async_hook, run_multi_turn

from agent.training.embodied_staged_contracts import (
    AliceProposal,
    BobEvalReceipt,
    BobEvalRequest,
    build_bob_eval_requests,
    finalize_alice_reward,
    preserve_rollout_num_turns,
)
from plugins.embodied_swift_grpo import (
    BOB_TASK,
    EmbodiedAliceScheduler,
    EmbodiedBobScheduler,
    _find_rollout_info,
)


ALICE_LORA_ID = 1
BOB_LORA_ID = 2
ALICE_LORA_NAME = "openeta_current_alice"
BOB_LORA_NAME = "openeta_frozen_bob"


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def shard_bob_requests(
    requests: list[BobEvalRequest],
    rank: int,
    world_size: int,
) -> list[BobEvalRequest]:
    """Deterministic rank-strided Bob work assignment."""

    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("invalid distributed rank/world_size")
    ordered = sorted(requests, key=lambda item: item.evaluation_id)
    return ordered[rank::world_size]


def register_staged_trainer() -> None:
    """Register this trainer without modifying the installed Swift package."""

    if os.environ.get("OPENETA_STAGED_COLOCATE", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        return
    from swift.infer_engine import grpo_vllm_engine
    from swift.rlhf_trainers import rollout_mixin
    from swift.trainers.trainer_factory import TrainerFactory

    # Swift's online trainable adapter uses module-level constants.  Staged
    # mode reserves ID 1 for it and ID 2 for the immutable Bob checkpoint.
    rollout_mixin.VLLM_LORA_INT_ID = ALICE_LORA_ID
    rollout_mixin.VLLM_LORA_NAME = ALICE_LORA_NAME
    grpo_vllm_engine.VLLM_LORA_INT_ID = ALICE_LORA_ID
    grpo_vllm_engine.VLLM_LORA_NAME = ALICE_LORA_NAME
    TrainerFactory.TRAINER_MAPPING["grpo"] = (
        "agent.training.embodied_staged_colocate.StagedEmbodiedGRPOTrainer"
    )


class StagedEmbodiedGRPOTrainer(GRPOTrainer):
    """Use all data-parallel ranks for Alice, Bob, then unchanged GRPO."""

    def _prepare_scheduler(self) -> None:
        super()._prepare_scheduler()
        if self.args.vllm_mode != "colocate":
            raise ValueError("staged embodied training requires --vllm_mode colocate")
        if self.args.vllm_tensor_parallel_size != 1:
            raise ValueError("staged embodied training requires --vllm_tensor_parallel_size 1")
        if getattr(self.args, "async_generate", False):
            raise ValueError("staged phase collectives are incompatible with --async_generate")
        # Swift may import an external plugin under a generated module name,
        # so class identity is not stable even though the registered scheduler
        # implements the same protocol.  The explicit role is the invariant.
        if getattr(self.multi_turn_scheduler, "role", None) != "alice":
            raise ValueError(
                "staged Alice training requires --multi_turn_scheduler embodied_alice_scheduler"
            )
        self.alice_scheduler = self.multi_turn_scheduler
        tokenizer = getattr(self, "processing_class", None)
        self.bob_scheduler = EmbodiedBobScheduler(
            max_turns=self.args.max_turns,
            tokenizer=tokenizer,
        )
        self._staged_group_index = 0
        self._train_forward_seconds = 0.0
        self._optimizer_window: list[dict[str, float]] = []
        self._bob_adapter_path = Path(
            os.environ.get("OPENETA_FROZEN_BOB_ADAPTER", "")
        ).expanduser().resolve()
        if not self._bob_adapter_path.is_dir():
            raise ValueError(
                "OPENETA_FROZEN_BOB_ADAPTER must point to a frozen LoRA checkpoint directory"
            )
        if not (self._bob_adapter_path / "adapter_config.json").is_file():
            raise ValueError("frozen Bob adapter has no adapter_config.json")
        self._bob_policy_version = os.environ.get(
            "OPENETA_BOB_POLICY_VERSION", str(self._bob_adapter_path)
        )
        configured_root = os.environ.get("OPENETA_STAGED_ARTIFACT_ROOT")
        self._staged_artifact_root = Path(
            configured_root or Path(self.args.output_dir).resolve().parent
        ).resolve()

    @contextmanager
    def multi_turn_completion_length_context(self):
        """Keep total-trajectory and per-turn token budgets independent.

        Swift's stock ``total`` context derives the temporary engine window
        from ``request_config.max_tokens``.  Our embodied scheduler must first
        replace that value with the per-turn thinking limit (normally 1024),
        so the stock context accidentally treats 1024 as the *whole-trajectory*
        budget.  Later turns then lose almost all generation space and an
        unfinished ``<think>`` is safely parsed as invalid DONE.

        Preserve Swift's algorithm, but derive the window from
        ``max_completion_length`` (normally 24576).  The scheduler's request
        config continues to enforce the independent per-turn limit.
        """

        if (
            not self.multi_turn_scheduler
            or not self.use_fast_infer
            or self.vllm_mode == "server"
            or self.completion_length_limit_scope == "per_round"
        ):
            with super().multi_turn_completion_length_context():
                yield
            return

        original_fn = self.engine.set_default_max_tokens
        original_engine_max_len = int(self.engine.max_model_len or 8192)
        total_completion_budget = int(self.max_completion_length)
        if total_completion_budget < 1:
            raise ValueError("max_completion_length must be positive")
        initialized = False
        window_info: dict[str, int] = {}

        def set_default_max_tokens(
            engine_self: Any,
            request_config: RequestConfig,
            inputs: dict[str, Any],
        ) -> None:
            nonlocal initialized
            prompt_tokens = int(engine_self._get_num_tokens(inputs))
            if not initialized:
                engine_self.max_model_len = min(
                    original_engine_max_len,
                    prompt_tokens + total_completion_budget,
                )
                window_info.update({
                    "engine_max_model_len": original_engine_max_len,
                    "initial_prompt_tokens": prompt_tokens,
                    "total_completion_budget": total_completion_budget,
                    "effective_total_window": int(engine_self.max_model_len),
                })
                initialized = True
            elif int(engine_self.max_model_len) <= prompt_tokens:
                # Match Swift's crash guard at the exhausted total budget.
                # This does not alter the configured policy budget; it only
                # permits a short terminal response instead of a negative
                # max_tokens value inside the inference engine.
                crash_guard_tokens = 10
                engine_self.max_model_len = min(
                    original_engine_max_len,
                    prompt_tokens + crash_guard_tokens,
                )
                request_config.max_tokens = max(
                    1, int(engine_self.max_model_len) - prompt_tokens
                )
            original_fn(request_config, inputs)

        try:
            self.engine.set_default_max_tokens = MethodType(
                set_default_max_tokens, self.engine
            )
            yield
        finally:
            self.engine.set_default_max_tokens = original_fn
            self.engine.max_model_len = original_engine_max_len
            self._last_completion_window = window_info

    def compute_loss(
        self,
        model: Any,
        inputs: Any,
        return_outputs: bool = False,
        num_items_in_batch: Any = None,
    ) -> Any:
        """Time Swift's unchanged GRPO forward/loss implementation."""

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
        """Measure forward/backward without changing Swift's training step."""

        self._train_forward_seconds = 0.0
        started = time.monotonic()
        result = super().training_step(model, inputs, num_items_in_batch)
        total_seconds = time.monotonic() - started
        forward_seconds = self._train_forward_seconds
        self._optimizer_window.append({
            "train_forward_seconds": forward_seconds,
            # HF's training_step surrounds compute_loss with backward and
            # gradient plumbing. This residual is the least invasive way to
            # time that phase while leaving GRPO loss/backward untouched.
            "train_backward_seconds": max(0.0, total_seconds - forward_seconds),
        })
        return result

    def create_optimizer(self, model: Any = None) -> Any:
        """Wrap only optimizer.step so phase timings include its wall time."""

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
                optimizer_seconds = time.monotonic() - started
                window = trainer._optimizer_window
                trainer._optimizer_window = []
                trainer._append_phase_timing({
                    "event": "training",
                    "global_step": trainer.state.global_step,
                    "group_index": max(0, trainer._staged_group_index - 1),
                    "rank": trainer.accelerator.process_index,
                    "micro_steps": len(window),
                    "train_forward_seconds": sum(
                        item["train_forward_seconds"] for item in window
                    ),
                    "train_backward_seconds": sum(
                        item["train_backward_seconds"] for item in window
                    ),
                    "optimizer_step_seconds": optimizer_seconds,
                })

        # LambdaLR expects optimizer.step to remain a bound method and reads
        # ``__func__`` while installing its own step counter.
        optimizer.step = MethodType(timed_step, optimizer)
        optimizer._openeta_timing_wrapped = True
        return optimizer

    def _prepare_vllm_engine(self):
        """Ask Swift to construct its normal engine with two LoRA slots."""

        from swift.infer_engine import GRPOVllmEngine

        original_init = GRPOVllmEngine.__init__

        def two_lora_init(engine_self: Any, *args: Any, **kwargs: Any) -> None:
            if kwargs.get("enable_lora"):
                kwargs["max_loras"] = 2
                engine_kwargs = dict(kwargs.get("engine_kwargs") or {})
                engine_kwargs.setdefault("max_cpu_loras", 2)
                kwargs["engine_kwargs"] = engine_kwargs
            original_init(engine_self, *args, **kwargs)

        GRPOVllmEngine.__init__ = two_lora_init
        try:
            engine = super()._prepare_vllm_engine()
        finally:
            GRPOVllmEngine.__init__ = original_init
        if not self.rollout_enable_lora:
            raise ValueError("staged Alice/Bob routing requires --vllm_enable_lora true")
        return engine

    def _lora_requests(self) -> tuple[Any, Any]:
        from vllm.lora.request import LoRARequest
        from swift.rlhf_trainers import rollout_mixin

        alice = LoRARequest(
            lora_name=ALICE_LORA_NAME,
            lora_int_id=ALICE_LORA_ID,
            lora_path=rollout_mixin.VLLM_LORA_PATH,
        )
        bob = LoRARequest(
            lora_name=BOB_LORA_NAME,
            lora_int_id=BOB_LORA_ID,
            lora_path=str(self._bob_adapter_path),
        )
        return alice, bob

    def _ensure_bob_adapter(self, bob_request: Any) -> None:
        loaded = set(self.engine.engine.list_loras())
        if BOB_LORA_ID not in loaded:
            self.engine.engine.add_lora(bob_request)
        if BOB_LORA_ID not in set(self.engine.engine.list_loras()):
            raise RuntimeError("frozen Bob LoRA was not loaded as adapter ID 2")

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

    def _rollout_with_adapter(
        self,
        samples: list[OnPolicySample] | list[RolloutInferRequest],
        request_config: RequestConfig,
        adapter_request: Any,
    ) -> list[RolloutOutput]:
        # TP=1 is an enforced invariant above: each DDP rank owns one engine.
        requests = self.samples2requests(samples)
        return self._engine_infer_with_adapter(requests, request_config, adapter_request)

    def _run_scheduler_phase(
        self,
        samples: list[OnPolicySample],
        scheduler: EmbodiedAliceScheduler | EmbodiedBobScheduler,
        adapter_request: Any,
    ) -> list[OnPolicySample]:
        requests = self.samples2requests(samples)
        invoke_async_hook(scheduler.on_trajectory_start(requests))
        for request, sample in zip(requests, samples):
            sample.messages = request.messages
        request_config = scheduler.prepare_request_config(self._get_request_config())
        # Use the exact requests initialized by the scheduler.  Rebuilding
        # them from OnPolicySample here would discard the first observation
        # image and the scheduler's chat-template overrides.
        first_outputs = self._rollout_with_adapter(requests, request_config, adapter_request)
        rollout_outputs = run_multi_turn(
            requests=requests,
            first_turn_outputs=first_outputs,
            scheduler=scheduler,
            rollout_fn=lambda reqs, cfg: self._rollout_with_adapter(
                reqs, cfg, adapter_request
            ),
            request_config=request_config,
            max_turns=self.args.max_turns,
            gather_fn=gather_object,
        )
        return self._postprocess_rollout_outputs(samples, rollout_outputs)

    def _assign_proposal_ids(self, samples: list[OnPolicySample]) -> None:
        rank = self.accelerator.process_index
        descriptors = [
            {
                "rank": rank,
                "local_index": index,
                "prompt_id": sample.prompt_id,
                "row_id": str(sample.extra.get("row_id", sample.prompt_id)),
            }
            for index, sample in enumerate(samples)
        ]
        all_descriptors = gather_object(descriptors)
        all_descriptors.sort(key=lambda item: (item["rank"], item["local_index"]))
        sample_indexes: dict[tuple[int, int], int] = {}
        counts: dict[str, int] = {}
        for item in all_descriptors:
            prompt_id = str(item["prompt_id"])
            sample_indexes[(int(item["rank"]), int(item["local_index"]))] = counts.get(
                prompt_id, 0
            )
            counts[prompt_id] = counts.get(prompt_id, 0) + 1

        round_id = _safe_id(os.environ.get("OPENETA_ROUND_ID", "round-0000"))
        group_id = f"group-{self.state.global_step:06d}-{self._staged_group_index:04d}"
        alice_version = os.environ.get(
            "OPENETA_ALICE_POLICY_VERSION",
            f"{self.args.output_dir}@step-{self.state.global_step}",
        )
        for local_index, sample in enumerate(samples):
            sample_index = sample_indexes[(rank, local_index)]
            proposal_id = (
                f"{round_id}-{group_id}-{_safe_id(sample.prompt_id)}-sample-{sample_index:02d}"
            )
            env_request = deepcopy(sample.extra.get("env_request") or {})
            env_request.update({
                "proposal_id": proposal_id,
                "alice_policy_version": alice_version,
                "bob_policy_version": self._bob_policy_version,
            })
            sample.extra["env_request"] = env_request

    def _extract_local_proposals(
        self, samples: list[OnPolicySample]
    ) -> list[AliceProposal]:
        proposals: list[AliceProposal] = []
        for sample in samples:
            info = _find_rollout_info(sample.rollout_infos)
            if not info or info.get("role") != "alice":
                raise RuntimeError("Alice phase returned no trusted rollout info")
            raw = info.get("proposal")
            if raw is None:
                # Invalid proposals finish locally and never enter Bob phase.
                env_request = sample.extra["env_request"]
                compiled = info.get("compiled_task") or {}
                fallback_info = deepcopy(info)
                fallback_info.setdefault("reward", 0.0)
                fallback_info.setdefault("reward_stage", "alice_infrastructure_error")
                fallback_info.setdefault("reward_pending", False)
                fallback_info.setdefault("format_penalty", 0.0)
                fallback_info.setdefault(
                    "bob_evaluations", int(os.environ.get("OPENETA_BOB_EVALUATIONS", "2"))
                )
                fallback_info.setdefault(
                    "bob_max_steps", int(env_request.get("bob_max_steps", self.args.max_turns))
                )
                fallback_info.setdefault("target_success_rate", 0.45)
                fallback_info.setdefault("replay_equal", False)
                fallback_info.setdefault("repeated", False)
                fallback_info.setdefault("proposal_id", env_request["proposal_id"])
                raw = {
                    "proposal_id": env_request["proposal_id"],
                    "alice_policy_version": env_request.get(
                        "alice_policy_version", "unknown"
                    ),
                    "bob_policy_version": self._bob_policy_version,
                    "snapshot_sha256": env_request["snapshot"]["state_sha256"],
                    "snapshot": deepcopy(env_request["snapshot"]),
                    "goal_predicate": deepcopy(compiled.get("goal_predicate") or {}),
                    "compiled_valid": False,
                    "replay_equal": bool(fallback_info["replay_equal"]),
                    "repeated": bool(fallback_info["repeated"]),
                    "target_success_rate": float(fallback_info["target_success_rate"]),
                    "alice_rollout_info": fallback_info,
                }
            proposal = AliceProposal.from_dict(raw)
            proposal.alice_rollout_info["rank"] = self.accelerator.process_index
            proposal.alice_response_token_ids = deepcopy(sample.response_token_ids)
            proposal.alice_response_loss_mask = deepcopy(sample.response_loss_mask)
            proposal.alice_rollout_logprobs = deepcopy(sample.rollout_logprobs)
            proposals.append(proposal)
        return proposals

    @staticmethod
    def _bob_sample(request: BobEvalRequest, proposal: AliceProposal) -> OnPolicySample:
        source_info = proposal.alice_rollout_info
        env_request = {
            "role": "bob",
            "env_id": str(
                request.snapshot.get("metadata", {}).get("env_id") or "PickCube-v1"
            ),
            "seed": request.seed,
            "snapshot": deepcopy(request.snapshot),
            "instruction": BOB_TASK,
            "goal_predicate": deepcopy(request.goal_predicate),
            "max_steps": request.max_steps,
            "camera_resolution": int(source_info.get("camera_resolution", 128)),
            "goal_tolerance": float(request.goal_predicate.get("tolerance", 0.025)),
        }
        return OnPolicySample(
            messages=[{"role": "user", "content": "Environment bootstrap pending."}],
            prompt_id=request.proposal_id,
            request_id=request.evaluation_id,
            extra={
                "role": "bob",
                "env_request": env_request,
                "chat_template_kwargs": {
                    "enable_thinking": bool(source_info.get("thinking_enabled", False))
                },
                "phase": "bob_evaluation",
                "adapter_name": BOB_LORA_NAME,
                "policy_version": request.bob_policy_version,
            },
        )

    def _receipt_from_sample(
        self,
        request: BobEvalRequest,
        sample: OnPolicySample,
    ) -> BobEvalReceipt:
        info = _find_rollout_info(sample.rollout_infos) or {}
        receipt_error = info.get("infrastructure_error")
        if info.get("role") != "bob":
            receipt_error = receipt_error or "Bob rollout returned no trusted Bob receipt"
        actual_snapshot_sha256 = str(info.get("snapshot_sha256") or "")
        actual_goal = info.get("goal_predicate")
        if not isinstance(actual_goal, dict):
            actual_goal = {}
        if actual_snapshot_sha256 != request.snapshot_sha256:
            receipt_error = receipt_error or "Bob rollout restored a different snapshot"
        if actual_goal != request.goal_predicate:
            receipt_error = receipt_error or "Bob rollout evaluated a different goal"
        return BobEvalReceipt(
            evaluation_id=request.evaluation_id,
            proposal_id=request.proposal_id,
            evaluation_index=request.evaluation_index,
            bob_policy_version=request.bob_policy_version,
            # Record what the scheduler actually restored/evaluated.  The
            # pure finalizer compares these fields with the request contract;
            # copying the expected values here would hide routing mistakes.
            snapshot_sha256=actual_snapshot_sha256,
            goal_predicate=deepcopy(actual_goal),
            success=bool(info.get("success", False)),
            reward=float(info.get("reward", 0.0)),
            step_count=int(info.get("step_count", 0)),
            infrastructure_error=receipt_error,
            rank=self.accelerator.process_index,
        )

    def _write_json(self, relative: Path, payload: dict[str, Any]) -> None:
        path = self._staged_artifact_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".rank{self.accelerator.process_index}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(path)

    def _append_phase_timing(self, timing: dict[str, Any]) -> None:
        rank_timing = self._staged_artifact_root / (
            f"phase_timing.rank-{self.accelerator.process_index:02d}.jsonl"
        )
        rank_timing.parent.mkdir(parents=True, exist_ok=True)
        with rank_timing.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(timing, ensure_ascii=False) + "\n")

    def _persist_group(
        self,
        proposals: list[AliceProposal],
        receipts: list[BobEvalReceipt],
        timing: dict[str, Any],
    ) -> None:
        for proposal in proposals:
            self._write_json(
                Path("alice_rollout/proposals") / f"{_safe_id(proposal.proposal_id)}.json",
                proposal.to_dict(),
            )
        for receipt in receipts:
            self._write_json(
                Path("bob_evaluation/receipts") / f"{_safe_id(receipt.evaluation_id)}.json",
                receipt.to_dict(),
            )
        self._append_phase_timing(timing)

    def _write_manifest(
        self,
        proposals: list[AliceProposal],
        receipts: list[BobEvalReceipt],
    ) -> None:
        if not self.accelerator.is_main_process:
            return
        by_proposal: dict[str, list[str]] = {}
        for receipt in receipts:
            by_proposal.setdefault(receipt.proposal_id, []).append(receipt.evaluation_id)
        row = {
            "schema_version": "openeta.staged_group_manifest.v1",
            "round": os.environ.get("OPENETA_ROUND_ID", "round-0000"),
            "global_step": self.state.global_step,
            "group_index": self._staged_group_index,
            "alice_adapter": ALICE_LORA_NAME,
            "bob_adapter": BOB_LORA_NAME,
            "bob_policy_version": self._bob_policy_version,
            "proposals": [
                {
                    "proposal_id": proposal.proposal_id,
                    "compiled_valid": proposal.compiled_valid,
                    "receipt_ids": sorted(by_proposal.get(proposal.proposal_id, [])),
                }
                for proposal in proposals
            ],
        }
        path = self._staged_artifact_root / "staged_group_manifest.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _fast_infer(self, samples: list[OnPolicySample]) -> list[OnPolicySample]:
        """Run Alice -> frozen Bob -> finalize before returning to Swift GRPO."""

        args = self.args
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.accelerator.device)
        phase_start = time.monotonic()
        if args.sleep_level > 0 and self.engine.inner_model_executor.is_sleeping:
            wake_kwargs = {}
            if "tags" in inspect.signature(self.engine.engine.wake_up).parameters:
                wake_kwargs = {"tags": ["weights"]}
            aggressive_empty_cache()
            self.engine.engine.wake_up(**wake_kwargs)
        wake_seconds = time.monotonic() - phase_start

        sync_start = time.monotonic()
        if self.state.global_step != self._last_loaded_step or args.sleep_level == 2:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
        weight_sync_seconds = time.monotonic() - sync_start
        alice_adapter, bob_adapter = self._lora_requests()

        context = self.offload_context if self.enable_offload else nullcontext
        with context():
            if (
                self.engine.inner_model_executor.is_sleeping
                and "tags" in inspect.signature(self.engine.engine.wake_up).parameters
            ):
                aggressive_empty_cache()
                set_expandable_segments(False)
                self.engine.engine.wake_up(tags=["kv_cache"])

            self._assign_proposal_ids(samples)
            alice_start = time.monotonic()
            with self.multi_turn_completion_length_context():
                alice_outputs = self._run_scheduler_phase(
                    samples, self.alice_scheduler, alice_adapter
                )
            alice_seconds = time.monotonic() - alice_start
            local_proposals = self._extract_local_proposals(alice_outputs)
            all_proposal_dicts = gather_object(
                [proposal.to_dict() for proposal in local_proposals]
            )
            all_proposals = sorted(
                (AliceProposal.from_dict(value) for value in all_proposal_dicts),
                key=lambda item: item.proposal_id,
            )

            bob_requests: list[BobEvalRequest] = []
            for proposal in all_proposals:
                evaluations = int(proposal.alice_rollout_info.get("bob_evaluations", 0))
                bob_requests.extend(build_bob_eval_requests(proposal, evaluations))
            local_requests = shard_bob_requests(
                bob_requests,
                self.accelerator.process_index,
                self.accelerator.num_processes,
            )
            proposals_by_id = {proposal.proposal_id: proposal for proposal in all_proposals}

            adapter_start = time.monotonic()
            if bob_requests:
                self._ensure_bob_adapter(bob_adapter)
            adapter_switch_seconds = time.monotonic() - adapter_start
            bob_start = time.monotonic()
            retry_limit = int(os.environ.get("OPENETA_BOB_EVAL_RETRIES", "1"))
            if retry_limit < 0:
                raise ValueError("OPENETA_BOB_EVAL_RETRIES must be non-negative")
            pending_requests = list(local_requests)
            local_receipts_by_id: dict[str, BobEvalReceipt] = {}
            if bob_requests:
                for _attempt in range(retry_limit + 1):
                    bob_samples = [
                        self._bob_sample(request, proposals_by_id[request.proposal_id])
                        for request in pending_requests
                    ]
                    if bob_samples:
                        with self.multi_turn_completion_length_context():
                            bob_outputs = self._run_scheduler_phase(
                                bob_samples, self.bob_scheduler, bob_adapter
                            )
                    else:
                        # Ranks with no local shard must still enter run_multi_turn
                        # collectives while peers work.  Swift's length context
                        # assumes at least one local engine request and otherwise
                        # deletes an attribute that was never created.
                        bob_outputs = self._run_scheduler_phase(
                            bob_samples, self.bob_scheduler, bob_adapter
                        )
                    attempt_receipts = [
                        self._receipt_from_sample(request, output)
                        for request, output in zip(pending_requests, bob_outputs)
                    ]
                    for receipt in attempt_receipts:
                        local_receipts_by_id[receipt.evaluation_id] = receipt
                    pending_requests = [
                        request
                        for request, receipt in zip(pending_requests, attempt_receipts)
                        if receipt.infrastructure_error
                    ]
            local_receipts = [
                local_receipts_by_id[request.evaluation_id] for request in local_requests
            ]
            bob_seconds = time.monotonic() - bob_start
            all_receipt_dicts = gather_object(
                [receipt.to_dict() for receipt in local_receipts]
            )
            all_receipts = sorted(
                (BobEvalReceipt.from_dict(value) for value in all_receipt_dicts),
                key=lambda item: item.evaluation_id,
            )
            receipts_by_proposal: dict[str, list[BobEvalReceipt]] = {}
            for receipt in all_receipts:
                receipts_by_proposal.setdefault(receipt.proposal_id, []).append(receipt)

            for sample, proposal in zip(alice_outputs, local_proposals):
                final_info = finalize_alice_reward(
                    proposal, receipts_by_proposal.get(proposal.proposal_id, [])
                )
                source_rollout_info = _find_rollout_info(sample.rollout_infos) or {}
                final_info = preserve_rollout_num_turns(
                    final_info, source_rollout_info
                )
                # Persist the finalized reward beside the immutable proposal
                # contract. The scheduler's trajectory artifact is written at
                # the Alice/Bob phase boundary and intentionally remains an
                # awaiting-receipts record.
                proposal.alice_rollout_info = deepcopy(final_info)
                final_info["proposal"] = proposal.to_dict()
                sample.rollout_infos = final_info

            sleep_start = time.monotonic()
            if args.sleep_level > 0:
                self.engine.engine.reset_prefix_cache()
                self.engine.engine.sleep(level=args.sleep_level)
                aggressive_empty_cache()
                set_expandable_segments(True)
            sleep_seconds = time.monotonic() - sleep_start

        successful_receipts = [receipt for receipt in all_receipts if not receipt.infrastructure_error]
        timing = {
            "event": "rollout",
            "global_step": self.state.global_step,
            "group_index": self._staged_group_index,
            "rank": self.accelerator.process_index,
            "alice_phase_seconds": alice_seconds,
            "alice_trajectories": len(local_proposals),
            "alice_valid_proposals": sum(p.compiled_valid for p in local_proposals),
            "bob_phase_seconds": bob_seconds,
            "bob_episodes": len(local_receipts),
            "bob_success_rate_mean": (
                sum(receipt.success for receipt in successful_receipts)
                / len(successful_receipts)
                if successful_receipts else None
            ),
            "adapter_switch_seconds": adapter_switch_seconds,
            "weight_sync_seconds": weight_sync_seconds,
            "vllm_wake_seconds": wake_seconds,
            "vllm_sleep_seconds": sleep_seconds,
            "gpu_peak_memory_bytes": (
                torch.cuda.max_memory_allocated(self.accelerator.device)
                if torch.cuda.is_available() else 0
            ),
            "gpu_peak_memory_by_rank": {
                str(self.accelerator.process_index): (
                    torch.cuda.max_memory_allocated(self.accelerator.device)
                    if torch.cuda.is_available() else 0
                )
            },
            # vLLM 0.19.1 does not expose a stable per-request prefix-cache
            # hit counter through GRPOVllmEngine; keep the field explicit
            # rather than inventing a value.
            "prefix_cache_hit_rate": None,
        }
        self._persist_group(local_proposals, local_receipts, timing)
        self._write_manifest(all_proposals, all_receipts)
        self._staged_group_index += 1
        return alice_outputs
