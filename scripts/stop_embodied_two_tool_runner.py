#!/usr/bin/env python3
"""Stop an explicitly identified OpenETA staged-runner process tree."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pid_file", type=Path)
    args = parser.parse_args()
    pid = int(args.pid_file.read_text().strip())
    root = psutil.Process(pid)
    command = " ".join(root.cmdline())
    allowed = ("run_embodied_two_tool_train_canary.sh",
               "run_embodied_swift_staged_formal5.sh")
    if not any(name in command for name in allowed):
        raise RuntimeError(f"refusing to stop unrelated PID {pid}: {command}")
    descendants = root.children(recursive=True)
    for process in reversed(descendants):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    # Some Inspire kernels reject pidfd_open even though Python exposes it.
    # Poll the captured Process identities instead of psutil.wait_procs/pidfd.
    deadline = time.monotonic() + 15
    alive = descendants
    while alive and time.monotonic() < deadline:
        running = []
        for process in alive:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    running.append(process)
            except psutil.NoSuchProcess:
                pass
        alive = running
        if alive:
            time.sleep(0.2)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        root.terminate()
    except psutil.NoSuchProcess:
        pass
    print(f"stopped staged runner {pid} and {len(descendants)} descendants")


if __name__ == "__main__":
    main()
