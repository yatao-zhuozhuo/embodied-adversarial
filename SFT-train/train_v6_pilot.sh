#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:?model name}"
shift
case "$NAME" in
  Qwen3.5-4B) export CUDA_VISIBLE_DEVICES=6; MODEL=/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B ;;
  Qwen3.5-9B) export CUDA_VISIBLE_DEVICES=7; MODEL="$ROOT/../.worktrees/baseline/baseline/.runtime/models/Qwen3.5-9B" ;;
  *) exit 2 ;;
esac
unset NPROC_PER_NODE RANK LOCAL_RANK WORLD_SIZE MASTER_PORT MASTER_ADDR
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT/logs/triton-fixed${PYTHONPATH:+:$PYTHONPATH}"
RUN="$ROOT/v6-pilot-third-20261011/$NAME"
mkdir -p "$RUN"
/opt/openeta-swift/bin/python "$ROOT/audit_mixed.py" "$MODEL" "$RUN/encoding.json" "$ROOT/v6-pilot-third-20261011/train.jsonl"
exec /opt/openeta-swift/bin/swift sft \
  --model "$MODEL" --model_type qwen3_5 --template qwen3_5 \
  --dataset "$ROOT/v6-pilot-third-20261011/train.jsonl" --output_dir "$RUN/checkpoints" \
  --tuner_type lora --target_modules all-linear --freeze_vit true --freeze_aligner true \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 \
  --torch_dtype bfloat16 --attn_impl sdpa \
  --loss_scale last_round+ignore_empty_think --enable_thinking true \
  --max_length 8192 --truncation_strategy delete --strict true \
  --split_dataset_ratio 0 --num_train_epochs 3 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 5e-5 --warmup_ratio 0.03 --lr_scheduler_type cosine \
  --gradient_checkpointing true --logging_steps 10 \
  --save_strategy steps --save_steps 100 --save_total_limit 5 \
  --check_model false --report_to none --dataloader_num_workers 2 --dataset_num_proc 4 --seed 42 "$@"
