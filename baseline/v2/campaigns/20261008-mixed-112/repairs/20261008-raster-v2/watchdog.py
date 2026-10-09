"""Hourly recovery of interrupted evaluators; completed task failures are never retried."""
import fcntl,json,os,socket,subprocess,sys,time
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parents[3];V2=C.parents[1];sys.path.insert(0,str(ROOT))
os.environ.update(NO_PROXY='*',no_proxy='*',BASELINE_SNAPSHOT=str(C/'snapshot'))
from supervise import healthy,listening,launch,PY,MODELS
from baseline.v2.runtime import verify_snapshot

def model_check(m,e):
 formal=C/'formal';run=formal/(m+'-run.json')
 if not run.exists():return {'state':'waiting_for_initial_acceptance'}
 rows=json.loads((V2/'manifests/all.json').read_text())['episodes']
 paths=[formal/m/r['episode_id']/'final.json' for r in rows]
 good=[p for p in paths if p.exists() and json.loads(p.read_text())['status']!='infrastructure_error']
 result={'completed':len(good),'planned':len(rows),'actions':[]}
 if len(good)==len(rows):return {**result,'state':'complete'}
 with (formal/f'.{m}.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return {**result,'state':'running'}
  if not healthy(e):
   if listening(e['port']):return {**result,'state':'unhealthy_existing_api_requires_inspection'}
   child=launch('model-'+m,[PY,str(C/'serve_model.py'),m]);result['actions'].append({'start_model':child.pid})
   for _ in range(120):
    if healthy(e):break
    if child.poll() is not None:return {**result,'state':'model_start_failed'}
    time.sleep(5)
   else:return {**result,'state':'model_loading'}
  if not listening(e['sim_port']):
   child=launch('simulator-'+m,['bash',str(V2/'serve_simulator.sh')],BASELINE_SIM_PORT=str(e['sim_port']),OPENETA_WORKER_GPUS=str(e['sim_gpu']))
   for _ in range(30):
    if listening(e['sim_port']):break
    time.sleep(1)
   else:return {**result,'state':'simulator_start_failed'}
  for path in paths:
   if path.exists() and json.loads(path.read_text())['status']=='infrastructure_error':
    dest=C/'recovery'/m/str(time.time_ns());dest.mkdir(parents=True)
    path.parent.rename(dest/path.parent.name);result['actions'].append({'archived':str(dest)})
 child=launch('formal-'+m,[PY,str(C/'evaluate.py'),'--model',m,'--root',str(formal),'--sim-url',f'http://127.0.0.1:{e["sim_port"]}/mcp'])
 return {**result,'state':'restarted','pid':child.pid}

def main():
 assert socket.gethostname().startswith('embodied-selfplay-baseline--')
 os.chdir(ROOT)
 with (C/'watchdog.lock').open('a') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  verify_snapshot();report={'checked_at':time.time(),'models':{}}
  for m,e in MODELS.items():
   try:report['models'][m]=model_check(m,e)
   except Exception as exc:report['models'][m]={'state':'check_error','error':str(exc)}
  tmp=C/'watchdog-latest.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(C/'watchdog-latest.json')
  with (C/'watchdog-history.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
  print(json.dumps(report),flush=True)

if __name__=='__main__':main()
