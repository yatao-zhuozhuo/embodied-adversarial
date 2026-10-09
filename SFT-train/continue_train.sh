#!/usr/bin/env bash
set -euo pipefail
TRAIN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Continue learned adapter weights with a fresh optimizer and LR schedule.
CHECKPOINT="${CHECKPOINT:-$TRAIN_ROOT/outputs/v0-20261007-024622/checkpoint-51}"
exec bash "$TRAIN_ROOT/train.sh" \
  --adapters "$CHECKPOINT" \
  --output_dir "$TRAIN_ROOT/outputs-continued" \
  --max_steps 300 --learning_rate 2e-5 --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --save_strategy steps --save_steps 50 --save_total_limit 6 "$@"
