#!/usr/bin/env bash
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:?model name required}"
case "$NAME" in
  Qwen3.5-4B) export CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MASTER_PORT=29641; ACCUM=4; MODEL=/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B ;;
  Qwen3.5-9B) export CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 MASTER_PORT=29642; ACCUM=4; MODEL="$TRAIN_ROOT/../.worktrees/baseline/baseline/.runtime/models/Qwen3.5-9B" ;;
  Qwen3.8-27B) export CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 MASTER_PORT=29643; ACCUM=2; MODEL=/inspire/hdd/global_public/public_models/Qwen/Qwen3.8-27B ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$TRAIN_ROOT/logs/triton-fixed${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$TRAIN_ROOT/mixed-runs/$NAME"
/opt/openeta-swift/bin/python "$TRAIN_ROOT/audit_mixed.py" "$MODEL" "$TRAIN_ROOT/mixed-runs/$NAME/encoding.json"
exec /opt/openeta-swift/bin/swift sft \
  --model "$MODEL" --model_type qwen3_5 --template qwen3_5 \
  --dataset "$TRAIN_ROOT/mixed-data/train.jsonl" \
  --output_dir "$TRAIN_ROOT/mixed-runs/$NAME/checkpoints" \
  --tuner_type lora --target_modules all-linear \
  --freeze_vit true --freeze_aligner true \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --loss_scale last_round+ignore_empty_think --enable_thinking true \
  --max_length 8192 --truncation_strategy delete --strict true \
  --split_dataset_ratio 0 --num_train_epochs 3 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$ACCUM" \
  --learning_rate 5e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --gradient_checkpointing true --ddp_find_unused_parameters false \
  --logging_steps 10 --save_strategy steps --save_steps 100 --save_total_limit 5 \
  --check_model false \
  --report_to none --dataloader_num_workers 2 --dataset_num_proc 4 --seed 42
