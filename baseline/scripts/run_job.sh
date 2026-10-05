#!/usr/bin/env bash
# Self-contained one-model task. Uses the prepared shared environments.
set -euo pipefail
REPO=/mnt/shared-storage-user/self-evolve/embodied-adversarial
cd "$REPO"
MODEL="${1:?model required}"
: "${BASELINE_RUN_ROOT:?shared campaign output directory required}"
: "${BASELINE_EXPECTED_SNAPSHOT:?frozen protocol hash required}"
export OPENETA_WORKER_GPUS=0
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=20261002
COORD="$REPO/baseline/.runtime/coordinator/bin/python"
export SWIFT_VENV=/mnt/shared-storage-user/self-evolve/.venv_selfplay_embodied.shared-symlink
# Restore only the machine-local interpreter alias expected by the formal5 venv.
PY312=/home/yangyongzhuo/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12
if [[ ! -x "$PY312" ]]; then
  command -v python3 >/dev/null
  python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "job image must supply Python 3.12"'
  mkdir -p "$(dirname "$PY312")"
  ln -s "$(command -v python3)" "$PY312"
fi
"$COORD" - <<'PY'
import hashlib, os
from baseline.runtime.run import SNAPSHOT, verify_snapshot
assert verify_snapshot() == os.environ['BASELINE_EXPECTED_SNAPSHOT'], 'snapshot mismatch'
PY
mkdir -p "$BASELINE_RUN_ROOT/job-logs"
SIM_PID=""
VLLM_PID=""
cleanup() {
  [[ -z "$SIM_PID" ]] || kill "$SIM_PID" 2>/dev/null || true
  [[ -z "$VLLM_PID" ]] || kill "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
if [[ "$MODEL" == deepseek ]]; then
  # Existing user-designated credential file; credentials never enter the spec/log.
  export BASELINE_DEEPSEEK_ENV_FILE="${BASELINE_DEEPSEEK_ENV_FILE:-/mnt/shared-storage-user/self-evolve/auto-train-pipline/runs/_evaluation_api_bridge/deepseek.env}"
else
  CUDA_VISIBLE_DEVICES=0 bash baseline/scripts/serve_vllm.sh "$MODEL" >"$BASELINE_RUN_ROOT/job-logs/$MODEL-vllm.log" 2>&1 &
  VLLM_PID=$!
fi
bash baseline/scripts/serve_simulator.sh >"$BASELINE_RUN_ROOT/job-logs/$MODEL-simulator.log" 2>&1 &
SIM_PID=$!
"$COORD" - "$MODEL" <<'PY'
import json,sys,time,urllib.request,socket
from baseline.runtime.provider import BASELINE
entry=json.loads((BASELINE/'configs/models.json').read_text())[sys.argv[1]]
for i in range(240):
    try:
        with socket.create_connection(('127.0.0.1',8200),timeout=2): pass
        if entry.get('local'):
            with urllib.request.urlopen(entry['api_base']+'/models',timeout=3) as response: assert response.status==200
        break
    except (OSError,AssertionError): time.sleep(5)
else: raise SystemExit('services did not become ready within 20 minutes')
PY
# Environment-chain gate checks all four native checkers before any scored run.
for BENCH in libero metaworld maniskill3 robocasa; do
  "$COORD" baseline/scripts/native_chain_smoke.py --bench "$BENCH" --output "$BASELINE_RUN_ROOT/job-logs/$MODEL-$BENCH-preflight.json" \
    >"$BASELINE_RUN_ROOT/job-logs/$MODEL-$BENCH-preflight.log" 2>&1
done
# A separate 2-turn model/environment canary cannot contaminate formal scores.
"$COORD" baseline/scripts/launch_model.py --model "$MODEL" \
  --root "$BASELINE_RUN_ROOT/canary" --benchmark metaworld --limit 1 --turns 2 --timeout 300
"$COORD" baseline/scripts/check_canary.py "$BASELINE_RUN_ROOT/canary/$MODEL/baseline-metaworld-00/final.json"
"$COORD" baseline/scripts/launch_model.py --model "$MODEL" --root "$BASELINE_RUN_ROOT/formal"
