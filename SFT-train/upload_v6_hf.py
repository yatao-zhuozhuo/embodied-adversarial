#!/usr/bin/env python3
"""Upload a verified v6 merged model to its private Hugging Face repository."""
import json
import shutil
import sys
from pathlib import Path
from huggingface_hub import HfApi

name = sys.argv[1]
assert name in ('Qwen3.5-4B', 'Qwen3.5-9B')
root = Path(__file__).resolve().parent
record = json.loads((root / 'v6-pilot-third-20261011/export-sft.json').read_text())[name]
folder = Path(record['output'])
assert (folder / 'openeta_sft_provenance.json').is_file(), 'Verify export before upload'
repo = f'YangYongzhuo/{name}-embodied-sft'
token = Path('/root/.config/openeta/hf-upload.token').read_text().strip()
api = HfApi(token=token)
assert api.whoami()['name'] == 'YangYongzhuo'
api.create_repo(repo_id=repo, repo_type='model', private=True, exist_ok=True)
api.update_repo_settings(repo_id=repo, repo_type='model', private=True)
assert api.model_info(repo).private
license_file = Path(record['base']) / 'LICENSE'
if license_file.exists():
    shutil.copy2(license_file, folder / 'LICENSE')
card = f'''---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/{name}
tags:
- embodied-ai
- robotics
- sft
- qwen3_5
---

# {name}-embodied-sft

Standalone BF16 Hugging Face model with a trained LoRA adapter merged into Qwen/{name}.
No separate adapter is required. The tokenizer, chat template and visual processor are included.

## Training

Dataset: OpenETA `expert_replay_unified_v6_reasoning_pilot_third_20261011`.
77 successful expert replay trajectories, 3,487 decision samples:
LIBERO 28 trajectories / 1,402 samples; MetaWorld 33 / 1,579; ManiSkill 16 / 506.
Only this pilot dataset was used, with no automatic validation split.

Inputs contain the task/control contract, recent action/execution history, robot state,
remaining budget and ordered camera observations. Targets are the current XML decision,
including action, repetition count and reasoning. Earlier assistant messages are masked.
Reasoning is Gemini's post-hoc step-purpose annotation of fixed expert actions,
not the expert's original hidden thought process.

Swift 4.4.2, language-module LoRA rank 16 / alpha 32 / dropout 0.05;
visual encoder and aligner frozen. BF16, SDPA, batch size 1, gradient accumulation 8,
3 epochs, 1,308 optimizer steps, peak learning rate 5e-5, cosine schedule and 3% warmup.
Template: qwen3_5. Loss: last_round+ignore_empty_think. Maximum context: 8,192 tokens.
All samples passed encoding checks (1,815–5,623 tokens).
Final export merges checkpoint-1308. Interrupted training was resumed with optimizer
and scheduler state restored.

## Evaluation

No environment rollout evaluation results are reported for this checkpoint.
Training fit does not establish task success or generalization.
'''
(folder / 'README.md').write_text(card)
patterns = ['*.safetensors', 'model.safetensors.index.json', 'config.json', 'generation_config.json', '*tokenizer*', '*processor*',
            '*preprocessor*', '*.jinja', 'vocab.json', 'merges.txt', 'special_tokens_map.json',
            'added_tokens.json', 'LICENSE', 'README.md']
print('Uploading private repository', repo, flush=True)
api.upload_large_folder(repo_id=repo, repo_type='model', folder_path=folder,
                        allow_patterns=patterns, num_workers=4, print_report_every=30)
info = api.model_info(repo, files_metadata=True)
assert info.private
files = {file.rfilename: file for file in info.siblings}
index = json.loads((folder / 'model.safetensors.index.json').read_text())
required = set(index['weight_map'].values()) | {'model.safetensors.index.json', 'config.json',
            'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json', 'README.md'}
assert required <= set(files), required - set(files)
for shard in set(index['weight_map'].values()):
    assert files[shard].size == (folder / shard).stat().st_size, shard
report = {'repo_id': repo, 'url': f'https://huggingface.co/{repo}', 'private': info.private,
          'revision': info.sha, 'files': len(files), 'global_step': 1308}
(root / 'v6-pilot-third-20261011' / name / 'hf-upload.json').write_text(json.dumps(report, indent=2) + '\n')
print('Upload and remote verification completed', json.dumps(report), flush=True)
