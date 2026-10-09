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

## Mixed embodied SFT

- Project: `exploration-topic`; Workspace: `分布式训练空间`; intended notebook: `embodied-selfplay-copy`.
- Training workspace: `/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA/SFT-train`.
- Prepared dataset: `<training workspace>/mixed-data/train.jsonl`; source/filter statistics: `mixed-data/summary.json`.
- Inputs reference images under the parent checkout's `SFT_data/`; retain those directories while this dataset is consumed.
- Model-specific adapters, encoding audits and logs: `<training workspace>/mixed-runs/<model>/`.
- Qwen3.5-4B and Qwen3.8-27B base weights: `/inspire/hdd/global_public/public_models/Qwen/<model>`.
- Qwen3.5-9B base weights: `/inspire/qb-ilm/project/exploration-topic/ky26060/OpenETA/.worktrees/baseline/baseline/.runtime/models/Qwen3.5-9B` (baseline-owned, also consumed by SFT).
- Consumers require both project qb-ilm and public HDD mounts. Machine-local Swift runtime: `/opt/openeta-swift`.
- Standalone merged BF16 models from final step 1320: `/inspire/qb-ilm/project/exploration-topic/ky26060/model/{Qwen3.5-4B,Qwen3.5-9B,Qwen3.8-27B}-embodied-mixed`.
- Each export includes Hugging Face weights, tokenizer/processor files and `openeta_sft_provenance.json` identifying its source adapter and validation. These models can be loaded independently of the training checkout.
