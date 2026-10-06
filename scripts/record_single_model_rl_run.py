"""Freeze the actual single-policy run inputs and code identity before training."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.training.single_model_rollout import write_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--dataset',type=Path,required=True)
a=p.parse_args()
config_path=Path(os.environ.get('OPENETA_RL_CONFIG', ROOT/'configs/single_model_rl_20tasks.json'))
paths=[config_path,a.dataset,ROOT/'docs/baseline-40-tasks.json']
paths.append(ROOT/json.loads(config_path.read_text())['task_manifest'])
for pattern in ['agent/training/*.py','agent/runtime/two_tool*.py','plugins/*grpo.py','sim/bench_worker.py',
                'sim/native_state_fingerprint.py','sim/two_tool_contract.py','sim/unified_env.py','sim/env_registry.py','sim/mcp_server/*.py',
                'scripts/*single_model*.py','scripts/*single_model*.sh']:
    paths.extend(ROOT.glob(pattern))
versions={}
for name in ['torch','ms-swift','vllm','transformers','peft','tensorboard']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]=None
payload={'schema':'openeta.single_model_rl.run.v1',
    'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'branch':subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
    'config':json.loads(config_path.read_text()),
    'dataset':str(a.dataset.resolve()),'versions':versions,
    'source_sha256':{str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path):
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    'resolved_environment':{key:os.environ.get(key) for key in [
        'TRAIN_GPU','NPROC_PER_NODE','MAX_TURNS','OPENETA_RL_TURN_TOKENS','OPENETA_RL_EPISODE_TOKENS',
        'VLLM_MAX_MODEL_LEN','MODEL_PATH','TRAIN_ADAPTER_PATH','RESUME_FROM_CHECKPOINT','NUM_GENERATIONS',
        'PER_DEVICE_TRAIN_BATCH_SIZE','GENERATION_BATCH_SIZE','REPORT_TO','TRAIN_STEPS','TRAIN_EPOCHS','BETA','TEMPERATURE','OPENETA_EMBODIED_MCP_URL']}}
if a.output.exists():
    old=json.loads(a.output.read_text())
    if old!=payload:raise ValueError('Run manifest changed; use a new run directory')
else:write_json(a.output,payload)
print(json.dumps({'manifest':str(a.output),'branch':payload['branch'],'revision':payload['code_revision']}))
