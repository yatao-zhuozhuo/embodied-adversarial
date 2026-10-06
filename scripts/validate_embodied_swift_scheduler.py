#!/usr/bin/env python3
"""Run the real Swift scheduler and ManiSkill with a deterministic fake engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swift.infer_engine.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    RequestConfig,
    RolloutInferRequest,
    UsageInfo,
)

from plugins.embodied_swift_grpo import (
    EmbodiedAliceScheduler, EmbodiedBobScheduler,
    EmbodiedTwoToolAliceScheduler, EmbodiedTwoToolBobScheduler,
)

DEFAULT_MODEL = "/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B"


class _FakeEngine:
    def __init__(self, model: str, actions: list[str]):
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self._actions = iter(actions)

    async def infer_async(
        self,
        infer_request: RolloutInferRequest,
        request_config: RequestConfig,
        **kwargs: Any,
    ) -> ChatCompletionResponse:
        del infer_request, request_config, kwargs
        action = next(self._actions)
        token_ids = self.tokenizer.encode(action, add_special_tokens=False)
        return ChatCompletionResponse(
            model="deterministic-fake-engine",
            choices=[ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=action),
                finish_reason="stop",
                token_ids=token_ids,
                logprobs={
                    "content": [
                        {"token": str(token_id), "logprob": -0.1}
                        for token_id in token_ids
                    ]
                },
            )],
            usage=UsageInfo(
                prompt_tokens=0,
                completion_tokens=len(token_ids),
                total_tokens=len(token_ids),
            ),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--actions", default="MOVE_X_POS_FINE,DONE")
    parser.add_argument("--commands-file", type=Path,
                        help="JSONL model commands for the two-tool MCP protocol")
    parser.add_argument(
        "--alice-artifact",
        type=Path,
        help="Replay actions and snapshot from a known-valid Alice trajectory artifact.",
    )
    parser.add_argument("--bob-evaluator-url", default="http://127.0.0.1:8111")
    parser.add_argument("--bob-evaluations", type=int, default=2)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.alice_artifact:
        artifact = json.loads(args.alice_artifact.read_text(encoding="utf-8"))
        actions = [str(step["action"]).upper() for step in artifact["steps"]]
        snapshot = dict(artifact["initial_snapshot"])
        row = {
            "messages": [{"role": "user", "content": "Environment bootstrap pending."}],
            "role": "alice",
            "env_request": {
                "role": "alice",
                "env_id": str((snapshot.get("metadata") or {}).get("env_id") or "PickCube-v1"),
                "seed": snapshot.get("seed"),
                "snapshot": snapshot,
                "max_steps": len(actions),
                "camera_resolution": 128,
                "min_goal_displacement": 0.04,
                "goal_tolerance": 0.025,
                "replay_tolerance": 0.005,
                "bob_evaluator_url": args.bob_evaluator_url,
                "bob_evaluations": args.bob_evaluations,
                "bob_max_steps": 4,
                "bob_policy_version": "qwen3.5-4b-base-canary",
                "target_success_rate": 0.45,
            },
        }
    else:
        row = json.loads(args.dataset.read_text(encoding="utf-8").splitlines()[0])
        actions = ([line.strip() for line in args.commands_file.read_text().splitlines() if line.strip()]
                   if args.commands_file else
                   [value.strip().upper() for value in args.actions.split(",") if value.strip()])
    engine = _FakeEngine(args.model, actions)
    if args.commands_file:
        scheduler_type = (EmbodiedTwoToolBobScheduler if row["role"] == "bob"
                          else EmbodiedTwoToolAliceScheduler)
    else:
        scheduler_type = EmbodiedBobScheduler if row["role"] == "bob" else EmbodiedAliceScheduler
    scheduler = scheduler_type(infer_engine=engine, max_turns=len(actions))
    request = RolloutInferRequest(
        messages=list(row["messages"]),
        data_dict={key: value for key, value in row.items() if key != "messages"},
        uuid="scheduler-canary",
    )
    output = await scheduler.run(
        request,
        RequestConfig(
            max_tokens=12,
            temperature=0.0,
            return_details=True,
            logprobs=True,
        ),
    )
    trained_tokens = sum(sum(mask) for mask in output.response_loss_mask)
    logprob_tokens = sum(len(values) for values in output.rollout_logprobs)
    assistant_turns = sum(
        message.get("role") == "assistant" for message in (output.messages or [])
    )
    assert assistant_turns == len(actions)
    assert len(output.response_token_ids) == len(actions)
    assert len(output.response_loss_mask) == len(actions)
    assert trained_tokens == logprob_tokens
    assert len(output.rollout_infos["images"]) == len(actions)
    return {
        "role": row["role"],
        "actions": actions,
        "assistant_turns": assistant_turns,
        "response_turns": len(output.response_token_ids),
        "trained_tokens": trained_tokens,
        "logprob_tokens": logprob_tokens,
        "image_count": len(output.rollout_infos["images"]),
        "reward": output.rollout_infos["reward"],
        "success": output.rollout_infos["success"],
        "termination_reason": output.rollout_infos["termination_reason"],
        "reward_stage": output.rollout_infos.get("reward_stage"),
        "bob_evaluation": output.rollout_infos.get("bob_evaluation"),
    }


def main() -> None:
    args = parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
