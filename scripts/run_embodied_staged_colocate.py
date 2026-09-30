#!/usr/bin/env python3
"""Process entry that installs OpenETA's staged trainer before Swift starts."""

from __future__ import annotations

import os


def main() -> None:
    if os.environ.get("OPENETA_STAGED_COLOCATE", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        raise RuntimeError("set OPENETA_STAGED_COLOCATE=true for this entry point")

    # In TP=1 colocate each DDP worker must expose only its assigned physical
    # GPU to vLLM and ManiSkill.  Do this before importing either subsystem.
    from swift.cli.utils import try_use_single_device_mode

    try_use_single_device_mode()

    # Importing the plugin registers the two schedulers and trusted reward
    # functions.  TrainerFactory is changed only in this process and only when
    # the staged feature flag above is present.
    import plugins.embodied_swift_grpo  # noqa: F401
    from agent.training.embodied_staged_colocate import register_staged_trainer
    from swift.pipelines import rlhf_main

    register_staged_trainer()
    rlhf_main()


if __name__ == "__main__":
    main()
