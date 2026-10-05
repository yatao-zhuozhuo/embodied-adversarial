"""Start the API relay and submit four jobs with the local Go SDK client."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--campaign',required=True)
p.add_argument('--credentials',type=Path,help='existing env file containing YICLOUD_PUBLIC_KEY and YICLOUD_SECRET_KEY')
p.add_argument('--dry-run',action='store_true');p.add_argument('--inspect',action='store_true')
a=p.parse_args()
env=os.environ.copy()
if a.credentials:
    for line in a.credentials.read_text().splitlines():
        line=line.strip().removeprefix('export ')
        if '=' not in line: continue
        key,value=line.split('=',1)
        if key in {'YICLOUD_PUBLIC_KEY','YICLOUD_SECRET_KEY','YICLOUD_API_HOST'}:
            env[key]=value.strip().strip('"\'')
requests=ROOT/'baseline/jobs'/a.campaign
binary=ROOT/'baseline/.runtime/baseline-submit'
if not binary.is_file(): raise SystemExit('Build baseline/sdk first; see submission report')
if not a.dry_run and any(not env.get(k) for k in ('YICLOUD_PUBLIC_KEY','YICLOUD_SECRET_KEY')):
    raise SystemExit('Missing YiCloud SDK credentials. No jobs submitted. Provide --credentials /existing/file.env.')
if not a.dry_run and not a.inspect:
    from baseline.runtime.run import verify_snapshot
    expected=verify_snapshot()
    for f in requests.glob('*.json'):
        if f.name.endswith('.receipt.json'): continue
        d=json.loads(f.read_text());assert d['TaskSpecs']['task1']['Env']['BASELINE_EXPECTED_SNAPSHOT']==expected
    deepseek=json.loads((requests/'deepseek.json').read_text())
    relay=Path(deepseek['TaskSpecs']['task1']['Env']['BASELINE_DEEPSEEK_RELAY_DIR'])
    relay.mkdir(parents=True,exist_ok=True)
    health=relay/'health.json'
    live=False
    if health.exists():
        h=json.loads(health.read_text());live=time.time()-h['time']<300
    if not live:
        with (relay/'relay.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-m','baseline.runtime.relay','--root',str(relay),
                '--credential-file','/mnt/shared-storage-user/self-evolve/auto-train-pipline/runs/_evaluation_api_bridge/deepseek.env'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        for _ in range(150):
            if child.poll() is not None: raise SystemExit('DeepSeek relay exited; inspect relay.log')
            if health.exists() and time.time()-json.loads(health.read_text())['time']<10: break
            time.sleep(1)
        else: raise SystemExit('DeepSeek relay did not become ready')
for model in ('deepseek','qwen38_27b','qwen35_9b','qwen35_4b'):
    spec=requests/f'{model}.json';receipt=requests/f'{model}.receipt.json'
    cmd=[str(binary),'-spec',str(spec)]
    if a.dry_run: cmd+=['-dry-run']
    elif a.inspect or receipt.exists(): cmd+=['-inspect']
    else: cmd+=['-receipt',str(receipt)]
    subprocess.run(cmd,env=env,check=True)
