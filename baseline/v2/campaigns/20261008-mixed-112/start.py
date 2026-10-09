"""Detach the supervisor from the ephemeral Inspire notebook terminal."""
import subprocess,json,sys
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parents[3];(C/'logs').mkdir(exist_ok=True)
with (C/'logs/supervisor.log').open('a') as out:
 p=subprocess.Popen([str(ROOT/'baseline/.runtime/coordinator/bin/python'),str(C/'supervise.py')],cwd=ROOT,
       stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'supervisor_pid':p.pid,'campaign':str(C)}))
