#!/usr/bin/env python3
"""Restart only this batch's recorded process trees after a startup fix."""
import os
import signal
import time
from pathlib import Path

root = Path(__file__).resolve().parent
parents = {}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        status = (p / 'status').read_text()
        parents[int(p.name)] = int(next(l for l in status.splitlines() if l.startswith('PPid:')).split()[1])
    except (OSError, StopIteration):
        pass
for file in root.glob('mixed-runs/*/pid'):
    target = int(file.read_text())
    if Path(f'/proc/{target}/cmdline').exists():
        assert b'mixed-supervisor' in Path(f'/proc/{target}/cmdline').read_bytes()
    descendants = {target}
    while True:
        children = {pid for pid, parent in parents.items() if parent in descendants}
        if children <= descendants:
            break
        descendants |= children
    for pid in descendants:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    file.unlink()
time.sleep(3)
for file in root.glob('mixed-runs/*/exit_code'):
    file.unlink()
for file in root.glob('mixed-runs/*/train.log'):
    file.rename(file.with_name(f'startup-{time.time_ns()}.log'))
os.execv('/opt/openeta-swift/bin/python', ['/opt/openeta-swift/bin/python', str(root / 'launch_mixed.py')])
