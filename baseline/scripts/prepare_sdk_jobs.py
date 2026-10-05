"""Materialize reviewable YiCloud SDK requests; never submit implicitly."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime
import re

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--campaign',required=True)
a=p.parse_args()
if not re.fullmatch('[a-z0-9-]{1,40}',a.campaign): raise SystemExit('invalid campaign id')
directory=ROOT/'baseline/jobs'/a.campaign
directory.mkdir(parents=True,exist_ok=True)
snapshot=hashlib.sha256((ROOT/'baseline/snapshots/native-actions-v1/lock.json').read_bytes()).hexdigest()
results=ROOT/'baseline/results'/a.campaign
for model in ('deepseek','qwen38_27b','qwen35_9b','qwen35_4b'):
    name=f'{a.campaign}-{model.replace("_","-")}'
    env={'BASELINE_RUN_ROOT':str(results),'BASELINE_EXPECTED_SNAPSHOT':snapshot,'PYTHONUNBUFFERED':'1'}
    if model=='deepseek': env['BASELINE_DEEPSEEK_RELAY_DIR']=str(results/'deepseek-relay')
    req={'Name':name,'JobId':name,'Project':'fdj-infra','QuotaGroup':'fdj_gpu_task',
         'JobType':'normal','BackoffLimit':1,'AutoDeleteDuration':'720h','TaskSpecs':{'task1':{
             'Command':['bash',str(ROOT/'baseline/scripts/run_job.sh'),model], 'Env':env,
             'Image':'registry.lg.shzhisuan.com/library/sglang:v0.5.12.post1-cu130',
             'Replicas':1,'CPU':'16','GPU':'1','Memory':'96Gi','EphemeralStorage':'50Gi',
             'HostNetwork':False,'RestartPolicy':'Never','MaxRunningDuration':'48h','MaxWaitDuration':'24h',
             'PositiveTags':['feature/gpfs=yes'],
             'MountGpfsVolumes':[{'Source':'gpfs://gpfs1/self-evolve','Target':'/mnt/shared-storage-user/self-evolve'}]}}}
    path=directory/f'{model}.json'
    if path.exists() and json.loads(path.read_text())!=req:
        raise SystemExit(f'request already exists with different content: {path}; use a new campaign id')
    path.write_text(json.dumps(req,indent=2)+'\n')
print(directory)
