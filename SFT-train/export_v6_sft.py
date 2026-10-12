#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
records = {}
for name, gpu, base in [
    ('Qwen3.5-4B', '6', '/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B'),
    ('Qwen3.5-9B', '7', str(root.parent / '.worktrees/baseline/baseline/.runtime/models/Qwen3.5-9B')),
]:
    run = root / 'v6-pilot-third-20261011' / name
    assert (run / 'exit_code').read_text().strip() == '0'
    checkpoints = sorted(run.glob('checkpoints/*/checkpoint-1308'))
    assert checkpoints
    checkpoint = checkpoints[-1]
    output = root.parent.parent / 'model' / f'{name}-embodied-sft'
    assert not output.exists(), output
    logpath = run / 'export-sft.log'
    command = ['/opt/openeta-swift/bin/swift', 'export', '--model', base, '--model_type', 'qwen3_5',
               '--template', 'qwen3_5', '--load_args', 'false', '--adapters', str(checkpoint),
               '--merge_lora', 'true', '--torch_dtype', 'bfloat16', '--output_dir', str(output)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4')
    env.pop('NPROC_PER_NODE', None)
    with logpath.open('w') as log:
        p = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env=env)
    records[name] = {'pid': p.pid, 'base': base, 'checkpoint': str(checkpoint), 'output': str(output),
                     'log': str(logpath), 'global_step': 1308, 'training_samples': 3487, 'training_episodes': 77}
(root / 'v6-pilot-third-20261011/export-sft.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps(records, indent=2), flush=True)
