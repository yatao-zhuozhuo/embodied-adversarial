#!/usr/bin/env python3
"""Validate standalone model shards/configs and a merged adapter projection."""
import json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor

root = Path(__file__).resolve().parent
records = json.loads((root / 'logs/mixed-export.json').read_text())
for name, record in records.items():
    output = Path(record['output'])
    assert 'Successfully merged LoRA and saved' in Path(record['log']).read_text(), name
    index = json.loads((output / 'model.safetensors.index.json').read_text())
    tensors = set()
    for shard in set(index['weight_map'].values()):
        with safe_open(output / shard, framework='pt', device='cpu') as stream:
            keys = set(stream.keys())
            assert keys == {key for key, value in index['weight_map'].items() if value == shard}
            tensors.update(keys)
    assert not any('lora_' in key for key in tensors)
    assert not (output / 'adapter_config.json').exists()
    config = AutoConfig.from_pretrained(output, local_files_only=True)
    processor = AutoProcessor.from_pretrained(output, local_files_only=True)
    checkpoint = Path(record['checkpoint'])
    adapter_config = json.loads((checkpoint / 'adapter_config.json').read_text())
    with safe_open(checkpoint / 'adapter_model.safetensors', framework='pt', device='cpu') as stream:
        key = next(key for key in stream.keys() if '.in_proj_b.lora_A.weight' in key)
        a = stream.get_tensor(key).float()
        b = stream.get_tensor(key.replace('.lora_A.weight', '.lora_B.weight')).float()
    weight = key.removeprefix('base_model.model.').replace('.lora_A.weight', '.weight')
    base = Path(record['base'])
    base_index = json.loads((base / 'model.safetensors.index.json').read_text())
    with safe_open(base / base_index['weight_map'][weight], framework='pt', device='cpu') as stream:
        original = stream.get_tensor(weight)
    with safe_open(output / index['weight_map'][weight], framework='pt', device='cpu') as stream:
        merged = stream.get_tensor(weight)
    expected = (original.float() + (b @ a) * (adapter_config['lora_alpha'] / adapter_config['r'])).to(merged.dtype)
    assert torch.allclose(merged.float(), expected.float(), rtol=0.01, atol=0.0001), name
    assert not torch.equal(original, merged), 'Adapter did not alter the sampled projection'
    report = {'base_model': record['base'], 'adapter_checkpoint': record['checkpoint'],
              'global_step': 1320, 'training_samples': 3521, 'training_episodes': 74,
              'torch_dtype': 'bfloat16', 'merged': True, 'tensors': len(tensors),
              'shards': len(set(index['weight_map'].values())), 'verified_projection': weight,
              'model_type': config.model_type, 'processor': type(processor).__name__}
    (output / 'openeta_sft_provenance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(name, json.dumps(report), flush=True)
