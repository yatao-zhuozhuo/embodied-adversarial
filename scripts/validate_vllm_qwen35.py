#!/usr/bin/env python3
"""Validate that the pinned vLLM runtime can serve the OpenETA Qwen3.5 VLM.

This is intentionally independent from the Swift scheduler.  It distinguishes
model/runtime incompatibility from later environment or GRPO integration bugs.
vLLM starts worker processes with ``spawn``, so the executable entry point must
live in a real file and be guarded by ``if __name__ == '__main__'``.
"""

from __future__ import annotations

import argparse

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

DEFAULT_MODEL = "/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.new("RGB", (128, 128), (32, 32, 32))
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Choose one action. Return only DONE."},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    engine = LLM(
        model=args.model,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    )
    outputs = engine.generate(
        [{"prompt": prompt, "multi_modal_data": {"image": image}}],
        SamplingParams(max_tokens=8, temperature=0.0),
    )
    completion = outputs[0].outputs[0]
    print(f"VLLM_QWEN35_SMOKE_OUTPUT={completion.text!r}")
    print(f"TOKEN_IDS={list(completion.token_ids)}")


if __name__ == "__main__":
    main()
