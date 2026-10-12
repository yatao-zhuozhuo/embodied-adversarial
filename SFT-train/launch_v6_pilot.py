#!/usr/bin/env python3
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
result = {}
for name in ('Qwen3.5-4B', 'Qwen3.5-9B'):
    folder = root / 'v6-pilot-third-20261011' / name
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'pid').exists(), 'Inspect previous launch before relaunching'
    command = ['bash', '-c', 'bash "$1" "$2"; status=$?; echo "$status" > "$3"; exit "$status"',
               'v6-supervisor', str(root / 'train_v6_pilot.sh'), name, str(folder / 'exit_code')]
    with (folder / 'train.log').open('w') as log:
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, cwd=root.parent)
    (folder / 'pid').write_text(str(p.pid) + '\n')
    result[name] = p.pid
print(json.dumps(result), flush=True)
