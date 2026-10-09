"""Durable 1+1+2 campaign supervisor; only operates this campaign's local processes."""
import concurrent.futures,fcntl,json,os,socket,subprocess,sys,time,urllib.request
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parents[3];V2=C.parents[1]
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
os.environ.update(NO_PROXY='*',no_proxy='*',BASELINE_SNAPSHOT=str(C/'snapshot'))
PY=str(ROOT/'baseline/.runtime/coordinator/bin/python')
MODELS=json.loads((C/'models.json').read_text());(C/'logs').mkdir(exist_ok=True)

def save(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def state(stage,**fields):
 data={'stage':stage,'updated_at':time.time(),'host':socket.gethostname(),'gpu_allocation':{k:e['gpus'] for k,e in MODELS.items()},**fields}
 save(C/'status.json',data);print(json.dumps(data),flush=True)

def get(url):
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url,timeout=3) as r:return json.loads(r.read())

def healthy(entry):
 try:return any(m['id']==entry['model'] for m in get(entry['api_base']+'/models')['data'])
 except Exception:return False

def listening(port):
 try:
  with socket.create_connection(('127.0.0.1',port),timeout=1):return True
 except OSError:return False

def launch(name,cmd,**env):
 with (C/'logs'/(name+'.log')).open('a') as out:
  p=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**env},stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 return p

def run(name,cmd):
 p=launch(name,cmd);code=p.wait()
 if code:raise RuntimeError(f'{name} exited {code}; see logs/{name}.log')

def summary(model):
 p=C/'formal'/(model+'-summary.json')
 return json.loads(p.read_text()) if p.exists() else {'completed':0,'planned':207}

def main():
 assert socket.gethostname().startswith('embodied-selfplay-baseline--'),'Wrong host: refuse to allocate GPUs'
 lock=(C/'supervisor.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 from baseline.v2.runtime import verify_snapshot
 state('starting_services',snapshot_sha256=verify_snapshot())
 if all(summary(m).get('completed')==207 and not summary(m).get('infrastructure_errors') for m in MODELS):
  state('complete',summaries={m:summary(m) for m in MODELS});return
 children={};simulators={};evaluators={}
 for m,e in MODELS.items():
  if not healthy(e):
   assert not listening(e['port']),f'Unknown service on {e["port"]}'
   children[m]=launch('model-'+m,[PY,str(C/'serve_model.py'),m])
  if not listening(e['sim_port']):
   simulators[m]=launch('simulator-'+m,['bash',str(V2/'serve_simulator.sh')],BASELINE_SIM_PORT=str(e['sim_port']),
                         OPENETA_WORKER_GPUS=str(e['sim_gpu']),OPENETA_WORKER_LOG_DIR=str(C/'logs'))
 save(C/'processes.json',{'supervisor':os.getpid(),'models':{m:p.pid for m,p in children.items()},'simulators':{m:p.pid for m,p in simulators.items()}})
 deadline=time.time()+1800
 while not all(healthy(e) for e in MODELS.values()):
  for m,p in children.items():
   if p.poll() is not None:raise RuntimeError(f'{m} model exited {p.returncode}; inspect model log')
  if time.time()>deadline:raise TimeoutError('Model readiness deadline exceeded')
  state('loading_models',ready=[m for m,e in MODELS.items() if healthy(e)])
  time.sleep(15)
 state('hardware_probe')
 if not (C/'hardware_probe_report.json').exists():run('hardware-probe',[PY,str(C/'hardware_probe.py')])
 assert json.loads((C/'hardware_probe_report.json').read_text())['passed']
 def canary(m):
  e=MODELS[m]
  for b in ('libero','metaworld','maniskill3','robocasa'):
   root=C/'canary'/b
   run('canary-'+m+'-'+b,[PY,str(C/'evaluate.py'),'--model',m,'--root',str(root),
       '--sim-url',f'http://127.0.0.1:{e["sim_port"]}/mcp','--benchmark',b,'--limit','1','--turns','2','--timeout','600'])
   results=list((root/m).glob('*/final.json'));assert len(results)==1
   result=json.loads(results[0].read_text());assert result['status']!='infrastructure_error',result.get('error')
   assert result['turns'],'No provider request was made'
   assert result['generation']['model']==e['model'],'Wrong served checkpoint'
   for turn in result['turns']:
    assert any(i.get('attached') for i in turn.get('details',{}).get('vision_attachments',[])),'No image sent'
  return m
 state('model_canaries')
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for done in concurrent.futures.as_completed([pool.submit(canary,m) for m in MODELS]):
   print('canary passed: '+done.result(),flush=True)
 for m,e in MODELS.items():
  evaluators[m]=launch('formal-'+m,[PY,str(C/'evaluate.py'),'--model',m,'--root',str(C/'formal'),'--sim-url',f'http://127.0.0.1:{e["sim_port"]}/mcp'])
 record=json.loads((C/'processes.json').read_text());record['evaluators']={m:p.pid for m,p in evaluators.items()};save(C/'processes.json',record)
 while True:
  running=[m for m,p in evaluators.items() if p.poll() is None]
  failed={m:p.returncode for m,p in evaluators.items() if p.poll() not in (None,0)}
  state('running' if running else ('needs_attention' if failed else 'complete'),running=running,failed=failed,
        summaries={m:summary(m) for m in MODELS})
  if not running:break
  time.sleep(30)
 if failed:raise RuntimeError('Evaluator stopped; preserve evidence and inspect before retrying persistent infrastructure errors')

if __name__=='__main__':
 try:main()
 except Exception as exc:
  state('needs_attention',error=f'{type(exc).__name__}: {exc}');raise
