#!/usr/bin/env python3
"""Resume both interrupted pilot runs from their latest complete checkpoints."""
import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parent
tasks = []
for name in ('Qwen3.5-4B', 'Qwen3.5-9B'):
    folder = root / 'v6-pilot-third-20261011' / name
    pidfile = folder / 'pid'
    oldpid = int(pidfile.read_text())
    command_file = Path(f'/proc/{oldpid}/cmdline')
    if command_file.exists():
        assert b'v6-supervisor' not in command_file.read_bytes(), f'{name} still has an active supervisor'
    checkpoints = []
    for path in folder.glob('checkpoints/*/checkpoint-*'):
        if all((path / file).is_file() for file in ('adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json', 'rng_state.pth')):
            state = json.loads((path / 'trainer_state.json').read_text())
            checkpoints.append((state['global_step'], path))
    step, checkpoint = max(checkpoints, key=lambda row: row[0])
    assert step < 1308, f'{name} already finished'
    tasks.append((name, folder, step, checkpoint))

records = {}
for name, folder, step, checkpoint in tasks:
    logpath = folder / f'resume-{time.time_ns()}.log'
    command = ['bash', '-c', 'bash "$1" "$2" --resume_from_checkpoint "$3"; status=$?; echo "$status" > "$4"; exit "$status"',
               'v6-supervisor-resume', str(root / 'train_v6_pilot.sh'), name, str(checkpoint), str(folder / 'exit_code')]
    (folder / 'exit_code').unlink(missing_ok=True)
    with logpath.open('w') as log:
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=root.parent)
    (folder / 'pid').write_text(str(p.pid) + '\n')
    records[name] = {'pid': p.pid, 'resume_step': step, 'checkpoint': str(checkpoint), 'log': str(logpath)}
(root / 'v6-pilot-third-20261011/resume.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records, indent=2), flush=True)
