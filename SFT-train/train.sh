#!/usr/bin/env bash
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
exec /opt/openeta-swift/bin/swift sft \
  --model /inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B \
  --model_type qwen3_5 --template qwen3_5 \
  --dataset "$TRAIN_ROOT/data/train.jsonl" \
  --output_dir "$TRAIN_ROOT/outputs" \
  --tuner_type lora --target_modules all-linear \
  --freeze_vit true --freeze_aligner true \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --loss_scale last_round+ignore_empty_think --enable_thinking true \
  --max_length 8192 --truncation_strategy delete --strict true \
  --split_dataset_ratio 0 --num_train_epochs 3 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 --warmup_ratio 0.05 \
  --gradient_checkpointing true \
  --logging_steps 1 --save_strategy epoch --save_total_limit 3 \
  --report_to none --dataloader_num_workers 0 --seed 42 "$@"
