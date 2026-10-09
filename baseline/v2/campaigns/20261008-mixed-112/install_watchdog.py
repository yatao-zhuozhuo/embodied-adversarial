import json,shlex,socket,subprocess
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parents[3]
assert socket.gethostname().startswith('embodied-selfplay-baseline--')
tag='# openeta-baseline-v2-mixed-112-watchdog'
current=subprocess.run(['crontab','-l'],text=True,capture_output=True)
assert current.returncode in (0,1),current.stderr
lines=[x for x in current.stdout.splitlines() if tag not in x]
command='cd '+shlex.quote(str(ROOT))+' && '+shlex.quote(str(ROOT/'baseline/.runtime/coordinator/bin/python'))+' '+shlex.quote(str(C/'watchdog.py'))+' >> '+shlex.quote(str(C/'logs/watchdog.log'))+' 2>&1'
lines.append('17 * * * * '+command+' '+tag)
subprocess.run(['crontab','-'],input='\n'.join(lines)+'\n',text=True,check=True)
print(json.dumps({'installed':True,'schedule':'17 * * * *','tag':tag}))
