#!/usr/bin/env python3
"""Export final mixed-data adapters as standalone Hugging Face models."""
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
destination = root.parent.parent / 'model'
models = {
    'Qwen3.5-4B': ('0', '/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B'),
    'Qwen3.5-9B': ('2', str(root.parent / '.worktrees/baseline/baseline/.runtime/models/Qwen3.5-9B')),
    'Qwen3.8-27B': ('4', '/inspire/hdd/global_public/public_models/Qwen/Qwen3.8-27B'),
}
tasks = []
for name, (gpu, base) in models.items():
    runs = root / 'mixed-runs' / name
    assert (runs / 'exit_code').read_text().strip() == '0', name
    summaries = []
    for file in runs.glob('checkpoints/*/logging.jsonl'):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            if row.get('last_model_checkpoint') and row.get('global_step') == 1320:
                summaries.append(row)
    assert len(summaries) == 1, name
    checkpoint = Path(summaries[0]['last_model_checkpoint'])
    assert (checkpoint / 'adapter_model.safetensors').is_file()
    output = destination / f'{name}-embodied-mixed'
    assert not output.exists(), output
    command = ['/opt/openeta-swift/bin/swift', 'export', '--model', base,
               '--model_type', 'qwen3_5', '--template', 'qwen3_5', '--load_args', 'false',
               '--adapters', str(checkpoint), '--merge_lora', 'true',
               '--torch_dtype', 'bfloat16', '--output_dir', str(output)]
    tasks.append((name, gpu, base, checkpoint, output, command))

record = {}
for name, gpu, base, checkpoint, output, command in tasks:
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpu, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='4')
    env.pop('NPROC_PER_NODE', None)
    env['PYTHONPATH'] = str(root / 'logs/triton-fixed') + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    log_path = root / 'mixed-runs' / name / 'export.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=True)
    record[name] = {'pid': process.pid, 'base': base, 'checkpoint': str(checkpoint),
                    'output': str(output), 'log': str(log_path)}
(root / 'logs/mixed-export.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
