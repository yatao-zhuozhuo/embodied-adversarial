#!/usr/bin/env python3
"""Role-aware entry for bounded embodied GRPO in synchronous colocate mode."""

from __future__ import annotations

import faulthandler
import os
import signal


def main() -> None:
    # SIGUSR1 prints all Python thread stacks without terminating a long GPU
    # job, which makes distributed initialization stalls diagnosable.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    role = os.environ.get("OPENETA_TRAIN_ROLE", "").strip().lower()
    if role not in {"alice", "bob"}:
        raise RuntimeError("set OPENETA_TRAIN_ROLE=alice or bob")

    # This must run before importing torch/vLLM/ManiSkill in every DDP worker.
    from swift.cli.utils import try_use_single_device_mode

    try_use_single_device_mode()

    import plugins.embodied_swift_grpo  # noqa: F401
    from agent.training.embodied_bounded_grpo import register_bounded_trainer
    from swift.pipelines import rlhf_main

    register_bounded_trainer(role)
    rlhf_main()


if __name__ == "__main__":
    main()
