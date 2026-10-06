# Inspire asset paths

Project: `exploration-topic`; project user directory: `ky26060`.

## RoboCasa scene assets

- Canonical asset directory: `/inspire/hdd/project/exploration-topic/ky26060/OpenETA-assets/robocasa-models-assets`.
- OpenETA installation path: `/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA/sim/venvs/robocasa/src/robocasa/robocasa/models/assets` (symlink to the canonical directory).
- Consumers: the RoboCasa benchmark worker and SFT trajectory collection. A workload must mount both this project's qb-ilm and HDD paths; the Python environment remains on qb-ilm.
- The HDD directory is the sole persistent copy of these downloaded RoboCasa scene assets. Do not delete it while RoboCasa collection or replay is needed. If a workload cannot mount HDD, restore the assets to an accessible pool and update the symlink before launching it.

## RoboCasa human demonstrations

- Canonical pretrain dataset directory: `/inspire/hdd/project/exploration-topic/ky26060/OpenETA-assets/robocasa-human-demos-pretrain`.
- RoboCasa installation path: `/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA/sim/venvs/robocasa/src/robocasa/datasets/v1.0/pretrain` (symlink to the canonical directory).
- Consumer: the SFT reference-demonstration collector. The directory contains only the task datasets downloaded so far; new task downloads through RoboCasa's registered path write to HDD through the symlink.
- The HDD directory is the sole persistent copy of these demonstrations. Do not remove it while reference replay or trajectory audit is needed.

## LIBERO demonstrations

- Canonical dataset directory: `/inspire/hdd/project/exploration-topic/ky26060/OpenETA-assets/libero-demonstrations`.
- LIBERO installation path: `/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA/sim/venvs/libero/assets/datasets` (symlink to the canonical directory).
- Consumer: LIBERO SFT reference-demonstration collection. Downloaded HDF5 files are written through this symlink; workloads must mount the project's HDD and qb-ilm paths.
- This is the sole persistent copy of downloaded LIBERO demonstrations. Do not remove it while collection or replay is needed.

## Baseline evaluation checkout

- Project: `exploration-topic`; intended GPU notebook: `embodied-selfplay-baseline` in `分布式训练空间`.
- Independent `feat/baseline` worktree: `/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA/.worktrees/baseline`.
- Models and evaluation runtimes: `<worktree>/baseline/.runtime/`; simulator environments: `<worktree>/sim/venvs/`; outputs: `<worktree>/baseline/results/`.
- These are baseline-owned copies. The parent `OpenETA` checkout remains the SFT workspace; use the baseline worktree as the working directory for baseline commands.

## Single policy RL checkout and runtime

- Project scope: `exploration-topic` filesystem, platform project `前沿课题探索`; GPU notebook `embodied-selfplay-copy` in `分布式训练空间`.
- Independent `feat/rl` worktree: `/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA/.worktrees/rl`.
- Training runtime: `/opt/openeta-swift` on that notebook; Swift 4.4.2 and vLLM 0.19.1. Hopper FLA backend overlay: `/inspire/qb-ilm/project/exploration-topic/ky26060/openeta-runtime/tilelang-0.1.14-py312`.
- RoboCasa Python runtime: `/inspire/hdd/project/exploration-topic/ky26060/OpenETA-assets/runtimes/cpython-3.11.16-linux-x86_64-gnu`; the notebook's `/root/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu` points here to satisfy the shared simulator venv.
- ManiSkill YCB assets: `/inspire/hdd/project/exploration-topic/ky26060/OpenETA-assets/maniskill/mani_skill2_ycb`; mount this directory at the notebook's `/root/.maniskill/data/assets/mani_skill2_ycb` before using PickSingleYCB.
- RL simulator runtimes reuse the parent checkout's `sim/venvs` as read-only dependencies; RL worker source is loaded from the RL worktree. Keep both project HDD and qb-ilm mounts available.
- Extra ManiSkill drawing dependencies: `/inspire/qb-ilm/project/exploration-topic/ky26060/openeta-runtime/rl-maniskill-extra` (`svgpathtools==1.8.0`, `svgwrite==1.4.3`). The RL worker loads this explicit host-owned overlay via `OPENETA_MANISKILL_EXTRA_PATH`; the shared simulator venv remains unchanged.
