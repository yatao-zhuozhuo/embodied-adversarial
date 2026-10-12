#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from swift.model import get_model_processor
from swift.template import get_template

root = Path(__file__).resolve().parent
_, processor = get_model_processor(sys.argv[1], model_type='qwen3_5', load_model=False)
template = get_template(processor, template_type='qwen3_5', loss_scale='last_round+ignore_empty_think',
                        enable_thinking=True, max_length=8192, truncation_strategy='raise')
template.set_mode('train')
lengths = []
dataset = Path(sys.argv[3]) if len(sys.argv) > 3 else root / 'mixed-data/train.jsonl'
for i, line in enumerate(dataset.read_text().splitlines()):
    row = json.loads(line)
    encoded = template.encode(row)
    supervised = processor.tokenizer.decode([t for t in encoded['labels'] if t != -100])
    assert row['messages'][-1]['content'] in supervised, row['sample_id']
    assert '<think>' not in supervised and '</think>' not in supervised
    assert len(encoded['image_grid_thw']) == len(row['images'])
    lengths.append(len(encoded['input_ids']))
    if (i + 1) % 500 == 0:
        print('audit', i + 1, 'max_tokens', max(lengths), flush=True)
report = {'model': sys.argv[1], 'samples': len(lengths), 'min_tokens': min(lengths), 'max_tokens': max(lengths)}
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
