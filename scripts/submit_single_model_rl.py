"""Detach one bounded workload on the current (explicitly selected) GPU notebook."""
import argparse
from pathlib import Path
import json
import os
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--kind',choices=['mcp','preflight','train'],required=True)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--steps',type=int,default=-1)
    p.add_argument('--turns',type=int,default=60)
    a=p.parse_args()
    a.run_root.mkdir(parents=True,exist_ok=True)
    receipt_path=a.run_root/(a.kind+'.submission.json')
    if receipt_path.exists():
        p.error('Submission receipt already exists; inspect it and use a new run directory')
    env=os.environ.copy();env['RUN_ROOT']=str(a.run_root.resolve())
    if a.kind=='mcp': command=['bash',str(ROOT/'scripts/launch_single_model_rl_mcp.sh')]
    else:
        if not a.dataset or not a.output: p.error('dataset and output required')
        env.update(DATASET_PATH=str(a.dataset),OUTPUT_DIR=str(a.output),TRAIN_STEPS=str(a.steps),MAX_TURNS=str(a.turns))
        # Token is read in the child from the protected shared file, never printed.
        prefix='source scripts/single_model_rl_env.sh\nexport OPENETA_HOST_STATE_TOKEN="$(cat "${OPENETA_HOST_TOKEN_FILE:-$RUN_ROOT/.host_token}")"\n'
        if a.kind=='preflight':
            env['PREFLIGHT_OUTPUT']=str(a.output)
            script=prefix+'/opt/openeta-swift/bin/python scripts/preflight_single_model_rl.py --dataset "$DATASET_PATH" --output "$PREFLIGHT_OUTPUT"\n'
        else: script=prefix+'bash scripts/run_single_model_grpo.sh\n'
        script+='status=$?\nprintf "%s\\n" "$status" > "$RUN_ROOT/'+a.kind+'.exit"\nexit "$status"\n'
        command=['bash','-c',script]
    log=a.run_root/(a.kind+'.log')
    with log.open('a') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'kind':a.kind,'pid':process.pid,'hostname':os.uname().nodename,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19],'log':str(log),'dataset':str(a.dataset),'output':str(a.output)}
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
