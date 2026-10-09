#!/usr/bin/env bash
set -euo pipefail
V2_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$V2_DIR/../scripts/environment.sh"
export NO_PROXY='*'
export no_proxy='*'
export MS_ASSET_DIR="$V2_DIR/assets/maniskill"
export OPENETA_WORKER_LOG_DIR="$V2_DIR/logs"
export BASELINE_SNAPSHOT="$V2_DIR/snapshot"
export OPENETA_WORKER_GPUS="${OPENETA_WORKER_GPUS:-0}"
export OPENETA_WORKER_POOL_MAX=1
mkdir -p "$V2_DIR/logs"
