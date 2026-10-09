"""The Yicloud deployment may change paths, but not the training contract."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_yi_curriculum_changes_only_model_location():
    original = json.loads((ROOT / 'configs/single_model_rl_curriculum_mixed.json').read_text())
    deployed = json.loads((ROOT / 'configs/single_model_rl_curriculum_mixed_yi.json').read_text())
    assert deployed['model'].endswith('/Qwen3.5-4B-embodied-mixed')
    deployed['model'] = original['model']
    assert deployed == original


def test_yi_shell_keeps_original_training_and_worker_settings():
    keys = ['NUM_GENERATIONS', 'PER_DEVICE_TRAIN_BATCH_SIZE', 'GRADIENT_ACCUMULATION_STEPS',
            'TRAIN_EPOCHS', 'REPORT_TO', 'TRAIN_GPU', 'MAX_TURNS', 'OPENETA_RL_TURN_TOKENS',
            'OPENETA_RL_EPISODE_TOKENS', 'VLLM_MAX_MODEL_LEN', 'DATASET_SHUFFLE',
            'OPENETA_CURRICULUM_ZERO_LIMIT', 'OPENETA_WORKER_POOL_MAX',
            'OPENETA_WORKER_POOL_MAX_METAWORLD', 'OPENETA_WORKER_POOL_MAX_LIBERO',
            'OPENETA_WORKER_POOL_MAX_ROBOCASA']
    def read(script):
        command = 'source "$1"; for key in "${@:2}"; do printf "%s=%s\\n" "$key" "${!key}"; done'
        return subprocess.check_output(['bash', '-c', command, 'bash', str(ROOT / script), *keys], text=True)
    assert read('scripts/single_model_rl_yi_env.sh') == read('scripts/single_model_rl_curriculum_env.sh')
