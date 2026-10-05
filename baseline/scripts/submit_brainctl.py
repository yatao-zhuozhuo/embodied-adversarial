"""Submit frozen baseline requests with the existing brainctl login."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser();p.add_argument('--campaign',required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--inspect',action='store_true')
a=p.parse_args()
directory=ROOT/'baseline/jobs'/a.campaign
env={k:v for k,v in os.environ.items() if k.lower() not in {'http_proxy','https_proxy','all_proxy'}}
models=('deepseek','qwen38_27b','qwen35_9b','qwen35_4b')
for model in models:
    d=json.loads((directory/f'{model}.json').read_text());t=d['TaskSpecs']['task1'];quota=d['QuotaGroup'];ns=d['Project'];name=d['JobId']+'-b1'
    labels={'brainpp-log-collector':'fluent-bit','kubebrain.brainpp.cn/creator':'yangyongzhuo',
        'kubebrain.brainpp.cn/creator-id':'e344b3ff5628d23c1f4ad3d234d89309','kubebrain.brainpp.cn/creator-type':'user',
        'kubebrain.brainpp.cn/creator-department':'default','kubebrain.brainpp.cn/resourcetype':'rjob',
        'privatemachine.brainpp.cn/privatemachine':'group','quotagroup.brainpp.cn/preemptible':'no',
        'quotagroup.brainpp.cn/quotagroup':quota,'rjob.brainpp.cn/scheduler-name':'volcano',
        'rjob.brainpp.cn/zs-task-type':'rjob-normal','scheduling.volcano.sh/queue-name':ns+'-'+quota.replace('_','-')}
    annotations={'kubebrain.brainpp.cn/module':quota,'kubebrain.brainpp.cn/showname':name,'qos.quality':'Guaranteed',
        'rjob.brainpp.cn/scheduler-name':'volcano','rjob.brainpp.cn/zs-task-type':'rjob-normal',
        'scheduling.volcano.sh/queue-name':labels['scheduling.volcano.sh/queue-name'],'volcano.brainpp.cn/priority':'5'}
    resources={'cpu':t['CPU'],'memory':t['Memory'],'nvidia.com/gpu':t['GPU']}
    request={'apiVersion':'rjob.brainpp.cn/v1alpha1','kind':'RJob','metadata':{'name':name,'namespace':ns,'labels':labels,'annotations':annotations},
        'spec':{'autoDeleteDuration':d['AutoDeleteDuration'],'backoffLimit':1,'failurePolicy':'None','taskSpecs':{'task1':{
            'maxRunningDuration':t['MaxRunningDuration'],'maxWaitDuration':t['MaxWaitDuration'],'positiveTags':t['PositiveTags'],'replicas':1,
            'rootfsSource':{'kind':'image','metadata':{'creationTimestamp':None}},'updatePolicy':'Active',
            'template':{'metadata':{'labels':{**labels,'internet':'false'},'annotations':{'kubebrain.brainpp.cn/module':quota,
                'mount.brainpp.cn/gpfs':';'.join(v['Source']+':'+v['Target'] for v in t['MountGpfsVolumes'])}},
                'spec':{'containers':[{'name':'task1','image':t['Image'],'command':t['Command'],
                    'env':[{'name':k,'value':v} for k,v in t['Env'].items()],
                    'resources':{'limits':resources,'requests':resources}}],
                    'imagePullSecrets':[{'name':labels['kubebrain.brainpp.cn/creator-id']}],
                    'restartPolicy':'Never','securityContext':{},'terminationGracePeriodSeconds':60}}}}}}
    path=directory/f'{model}.rjob.json';path.write_text(json.dumps(request,indent=2)+'\n')
    if a.prepare_only: print(path);continue
    if not a.inspect and model=='deepseek':
        lock_path=ROOT/'baseline/snapshots/native-actions-v1/lock.json'
        assert hashlib.sha256(lock_path.read_bytes()).hexdigest()==t['Env']['BASELINE_EXPECTED_SNAPSHOT']
        for rel, digest in json.loads(lock_path.read_text())['files'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==digest, rel
        relay=Path(t['Env']['BASELINE_DEEPSEEK_RELAY_DIR']);relay.mkdir(parents=True,exist_ok=True)
        health=relay/'health.json'
        if not health.exists() or time.time()-json.loads(health.read_text())['time']>300:
            with (relay/'relay.log').open('a') as log:
                child=subprocess.Popen([str(ROOT/'baseline/.runtime/coordinator/bin/python'),'-m','baseline.runtime.relay','--root',str(relay),
                    '--credential-file','/mnt/shared-storage-user/self-evolve/auto-train-pipline/runs/_evaluation_api_bridge/deepseek.env'],
                    cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            for _ in range(180):
                if child.poll() is not None: raise RuntimeError('relay exited')
                if health.exists() and time.time()-json.loads(health.read_text())['time']<10:break
                time.sleep(1)
            else: raise TimeoutError('relay startup')
    get=['brainctl','get','rjob',name,'-n',ns,'--request-timeout=20s','-o','json']
    result=subprocess.run(get,env=env,capture_output=True,text=True)
    if result.returncode:
        if a.inspect or 'NotFound' not in result.stderr: raise RuntimeError(result.stderr)
        subprocess.run(['brainctl','create','-f',str(path),'--request-timeout=30s'],env=env,check=True)
        result=subprocess.run(get,env=env,capture_output=True,text=True,check=True)
    live=json.loads(result.stdout)
    receipt={'name':name,'uid':live['metadata']['uid'],'created_at':live['metadata'].get('creationTimestamp'),
        'phase':live.get('status',{}).get('phase'),'status':live.get('status',{}),'submission_method':'brainctl'}
    (directory/f'{model}.brainctl-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('name','uid','phase')}),flush=True)
