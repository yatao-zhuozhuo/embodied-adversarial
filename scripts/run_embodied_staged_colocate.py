#!/usr/bin/env python3
"""Backward-compatible Alice wrapper for the bounded colocate entry."""

from __future__ import annotations

import os


def main() -> None:
    if os.environ.get("OPENETA_STAGED_COLOCATE", "").strip().lower() not in {
        "1", "true", "yes", "on",
    }:
        raise RuntimeError("set OPENETA_STAGED_COLOCATE=true for this entry point")

    os.environ.setdefault("OPENETA_TRAIN_ROLE", "alice")
    from scripts.run_embodied_bounded_colocate import main as bounded_main

    bounded_main()


if __name__ == "__main__":
    main()
