"""Resume only the mixed 4B evaluator on the user-selected local 4090."""
import fcntl,json,os,socket,subprocess,sys,time,hashlib
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parents[3];sys.path.insert(0,str(C));sys.path.insert(0,str(ROOT));os.chdir(ROOT)
from supervise import healthy,listening,launch,PY,MODELS
from baseline.v2.runtime import verify_snapshot
M='qwen35_4b';D=C/'resume-4090';D.mkdir(exist_ok=True)
def state(stage,**kw):
 d={'stage':stage,'updated_at':time.time(),'host':socket.gethostname(),'model':M,'gpu':0,**kw};p=D/'status.tmp';p.write_text(json.dumps(d,indent=2)+'\n');p.replace(D/'status.json');print(json.dumps(d),flush=True)
def main():
 assert socket.gethostname()=='selfplay-emboid--52025e87e4d9-yx5sxlbb63'
 with (D/'supervisor.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  verify_snapshot();formal=C/'formal';e=MODELS[M]
  with (formal/f'.{M}.lock').open('a') as elock:
   try:fcntl.flock(elock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:state('running_existing_evaluator');return
   good=[];archived=[]
   for folder in (formal/M).iterdir():
    if not folder.is_dir():continue
    p=folder/'final.json'
    if p.exists() and json.loads(p.read_text())['status']!='infrastructure_error':good.append(folder.name);continue
    dest=D/'interrupted'/str(time.time_ns())/folder.name;dest.parent.mkdir(parents=True);folder.rename(dest);archived.append(str(dest))
   if len(good)==207:state('complete',completed=207);return
   (D/'migration.json').write_text(json.dumps({'host':socket.gethostname(),'preserved_episodes':good,'archived_interrupted':archived,'snapshot_sha256':verify_snapshot(),'remaining':207-len(good)},indent=2)+'\n')
  state('loading_model',completed=len(good),remaining=207-len(good))
  model=None
  if not healthy(e):
   assert not listening(e['port']),'Unexpected service on model port'
   model=launch('4090-model-4b',[PY,str(C/'serve_model.py'),M])
  for _ in range(240):
   if healthy(e):break
   if model and model.poll() is not None:raise RuntimeError('Model failed to start; see 4090-model-4b.log')
   time.sleep(5)
  else:raise TimeoutError('Model readiness timeout')
  if not listening(e['sim_port']):launch('4090-simulator-4b',['bash',str(C.parents[1]/'serve_simulator.sh')],BASELINE_SIM_PORT=str(e['sim_port']),OPENETA_WORKER_GPUS='0')
  for _ in range(60):
   if listening(e['sim_port']):break
   time.sleep(2)
  else:raise TimeoutError('Simulator readiness timeout')
  state('provider_probe')
  from provider_probe import probe
  print(probe(M),flush=True)
  state('hardware_probe')
  import asyncio,hardware_probe
  # Probe the assigned 4090 service, keeping new diagnostics separate.
  old=hardware_probe.streamablehttp_client
  hardware_probe.streamablehttp_client=lambda url:old('http://127.0.0.1:8230/mcp')
  hardware_probe.C=D
  asyncio.run(hardware_probe.main())
  p=launch('4090-formal-4b',[PY,str(C/'evaluate.py'),'--model',M,'--root',str(formal),'--sim-url','http://127.0.0.1:8230/mcp'])
  while p.poll() is None:
   summary=json.loads((formal/(M+'-summary.json')).read_text());state('running',evaluator_pid=p.pid,summary=summary);time.sleep(30)
  if p.returncode:raise RuntimeError(f'Evaluator exited {p.returncode}; records preserved')
  state('complete',summary=json.loads((formal/(M+'-summary.json')).read_text()))
if __name__=='__main__':
 try:main()
 except Exception as exc:state('needs_attention',error=str(exc));raise
