#!/usr/bin/env python3
"""Check every sample with the same Swift template used for training."""
import json
from pathlib import Path
from swift.model import get_model_processor
from swift.template import get_template

root = Path(__file__).resolve().parent
_, processor = get_model_processor('/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B', load_model=False)
template = get_template(processor, template_type='qwen3_5',
                        loss_scale='last_round+ignore_empty_think', enable_thinking=True,
                        max_length=8192, truncation_strategy='raise')
template.set_mode('train')
report = []
for line in (root / 'data/train.jsonl').read_text().splitlines():
    row = json.loads(line)
    encoded = template.encode(row)
    supervised = processor.tokenizer.decode([token for token in encoded['labels'] if token != -100])
    assert row['messages'][-1]['content'] in supervised
    assert '<think>' not in supervised and '</think>' not in supervised
    assert len(encoded['image_grid_thw']) == len(row['images'])
    report.append({'sample_id': row['sample_id'], 'tokens': len(encoded['input_ids']),
                   'supervised_tokens': sum(token != -100 for token in encoded['labels'])})
(root / 'data/encoding_audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'samples': len(report), 'max_tokens': max(r['tokens'] for r in report),
                  'min_tokens': min(r['tokens'] for r in report)}), flush=True)
