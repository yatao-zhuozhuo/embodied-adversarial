"""Stop only a workload owned by a matching single-model submission receipt."""
import argparse
import json
import os
from pathlib import Path
import signal
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('submission',type=Path)
a=p.parse_args()
r=json.loads(a.submission.read_text());pid=r['pid']
proc=Path(f'/proc/{pid}')
if not proc.exists():raise SystemExit('Already stopped')
if r.get('hostname',os.uname().nodename)!=os.uname().nodename:
 raise SystemExit('Notebook ownership mismatch')
if 'start_ticks' in r and r['start_ticks']!=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]:
 raise SystemExit('PID was reused; refusing to signal')
environ=proc.joinpath('environ').read_bytes().split(b'\0')
expected=('RUN_ROOT='+str(a.submission.resolve().parent)).encode()
if expected not in environ:raise SystemExit('PID ownership mismatch; refusing to signal')
parents={}
for child in Path('/proc').iterdir():
 if child.name.isdigit():
  try:
   stat=child.joinpath('stat').read_text().rsplit(')',1)[1].split()
   parents[int(child.name)]=int(stat[1])
  except (OSError,ValueError):pass
owned={pid}
while True:
 new={child for child,parent in parents.items() if parent in owned}
 if new<=owned:break
 owned|=new
for target in sorted(owned,reverse=True):
 try:os.kill(target,signal.SIGTERM)
 except ProcessLookupError:pass
print(json.dumps({'stopped_submission':str(a.submission),'pids':sorted(owned)}))
