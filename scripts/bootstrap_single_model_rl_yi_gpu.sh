#!/usr/bin/env bash
# Run once on the GPU workspace after the network-host setup script.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/single_model_rl_yi_env.sh"
cd "$REPO_ROOT"
python3 - <<'PY'
import os
from pathlib import Path
root = Path.cwd()
python = Path(os.environ['SWIFT_VENV']) / 'bin/python'
alias = Path(os.readlink(python)).parents[1]
runtime = root / '.runtime/yi/python/cpython-3.12.11-linux-x86_64-gnu'
if not python.exists():
    if not (runtime / 'bin/python3.12').is_file():
        raise RuntimeError('Run the network-host setup first: shared Python runtime is missing')
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(runtime, target_is_directory=True)
assets = root / '.runtime/yi/maniskill-assets/mani_skill2_ycb'
if not (assets / 'info_pick_v0.json').is_file():
    raise RuntimeError('Run the network-host setup first: YCB assets are missing')
target = Path.home() / '.maniskill/data/assets/mani_skill2_ycb'
if target.is_dir() and not target.is_symlink() and not any(target.iterdir()):
    target.rmdir()
if not target.exists():
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(assets, target_is_directory=True)
PY
"$SWIFT_VENV/bin/python" -c 'import torch; assert torch.cuda.device_count() == 8; print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(8)])'
