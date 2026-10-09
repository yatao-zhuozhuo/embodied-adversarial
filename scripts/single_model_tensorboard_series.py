"""Launch an idempotent, CPU-only continuous TensorBoard checkpoint collector."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.training.tensorboard_series import ScalarSeries,active_state


def alive(receipt):
    if receipt.get('hostname')!=os.uname().nodename:return False
    path=Path('/proc')/str(receipt['pid'])/'stat'
    try:parts=path.read_text().rsplit(')',1)[1].split()
    except OSError:return False
    return parts[0]!='Z' and parts[19]==receipt['start_ticks']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path)
    p.add_argument('--interval',type=float,default=15.)
    p.add_argument('--start',action='store_true')
    p.add_argument('--once',action='store_true')
    a=p.parse_args();root=a.source_root.resolve()
    output=a.output.resolve() if a.output else root/'tensorboard/continuous'
    directory=output.parent;directory.mkdir(parents=True,exist_ok=True)
    if a.start:
        receipt_path=directory/'collector.submission.json'
        if receipt_path.exists() and alive(json.loads(receipt_path.read_text())):
            print(json.dumps({'collector':'already_running','output':str(output)}));return
        env=os.environ.copy();env['RUN_ROOT']=str(directory)
        env['CUDA_VISIBLE_DEVICES']='';env['OMP_NUM_THREADS']='1'
        with (directory/'collector.log').open('a') as stream:
            child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--source-root',str(root),
                '--output',str(output),'--interval',str(a.interval)],cwd=ROOT,env=env,
                stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        receipt={'kind':'tensorboard_collector','pid':child.pid,'hostname':os.uname().nodename,
            'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
            'output':str(output),'source_root':str(root)}
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return
    with (directory/'collector.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('Another collector owns this timeline')
        series=ScalarSeries(output)
        try:
            while True:
                try:
                    path,state=active_state(root)
                    if state is not None:
                        changed=series.publish(state)
                        if changed:print(json.dumps({'checkpoint':str(path),'global_step':state['global_step'],
                            'published_scalars':changed,'run':'continuous'}),flush=True)
                except (OSError,json.JSONDecodeError) as error:
                    print(json.dumps({'transient_read_error':str(error)}),flush=True)
                if a.once:break
                time.sleep(a.interval)
        finally:series.close()

if __name__=='__main__':main()
