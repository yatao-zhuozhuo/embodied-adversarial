#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
result = {}
for name in ('Qwen3.5-4B', 'Qwen3.5-9B', 'Qwen3.8-27B'):
    folder = root / 'mixed-runs' / name
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'pid').exists():
        raise RuntimeError(f'{name} already has a launch record; inspect before relaunch')
    command = ['bash', '-c', 'bash "$1" "$2"; status=$?; echo "$status" > "$3"; exit "$status"',
               'mixed-supervisor', str(root / 'train_mixed.sh'), name, str(folder / 'exit_code')]
    with (folder / 'train.log').open('w') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=root.parent)
    (folder / 'pid').write_text(str(process.pid) + '\n')
    result[name] = process.pid
print(json.dumps(result), flush=True)
