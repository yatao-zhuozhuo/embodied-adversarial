"""Run the frozen two-tool XML campaign across three independent MCP services."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
PYTHON=ROOT/'baseline/.runtime/coordinator/bin/python'
MODELS={'qwen35_4b':8200,'qwen38_27b':8201,'qwen35_9b':8202}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',required=True)
    args=parser.parse_args()
    if Path(args.campaign).name!=args.campaign or args.campaign in {'.','..'}:parser.error('invalid campaign name')
    root=ROOT/'baseline/results'/args.campaign;root.mkdir(exist_ok=False)
    logs=root/'logs';logs.mkdir();os.chdir(ROOT)
    def state(stage,**data):
        d={'stage':stage,'protocol':'openeta.two_tool_xml.v1','input_protocol':'openeta.two_tool_input.v1','updated_at':time.time(),**data}
        p=root/'status.tmp';p.write_text(json.dumps(d,indent=2)+'\n');p.replace(root/'status.json');print(json.dumps(d),flush=True)
    def run(log,script,*argv):
        with (logs/f'{log}.log').open('w') as f:
            subprocess.run([str(PYTHON),'baseline/scripts/'+script,*argv],stdout=f,stderr=subprocess.STDOUT,check=True)
    def canary(model,port):
        for bench in ['metaworld','libero','maniskill3','robocasa']:
            run(f'canary-{model}-{bench}','launch_model.py','--model',model,'--sim-url',f'http://127.0.0.1:{port}/mcp',
                '--root',str(root/'canary'/bench),'--benchmark',bench,'--limit','1','--turns','3','--timeout','300')
            run(f'check-{model}-{bench}','check_canary.py',str(root/'canary'/bench/model/f'baseline-{bench}-00/final.json'))
        return model
    try:
        # Require current input-profile acceptance before model canaries.
        for row in json.loads((ROOT/'baseline/manifests/all_40.json').read_text())['episodes']:
            evidence = ROOT/'baseline/results/input-v1-preflight'/f"{row['episode_id']}.json"
            proof = json.loads(evidence.read_text())
            assert proof['passed'], str(evidence)
            profile = json.loads((ROOT/'baseline/configs/input_profiles.json').read_text())[row['env_id']]
            assert proof['created']['input_capabilities'] == profile['expected_capabilities']
        # Environment lifecycle/repeat evidence is retained from this deployment.
        for phase in ['environment-preflight','environment-repeat']:
            for bench in ['libero','metaworld','maniskill3','robocasa']:
                for i in range(10):
                    assert json.loads((ROOT/f'baseline/results/{phase}/{bench}-{i}.json').read_text())['passed']
        state('xml_canary')
        with ThreadPoolExecutor(max_workers=3) as pool:
            for f in as_completed([pool.submit(canary,m,p) for m,p in MODELS.items()]):
                print('Canary passed: '+f.result(),flush=True)
        children={};handles=[]
        for model,port in MODELS.items():
            f=(logs/f'formal-{model}.log').open('w');handles.append(f)
            children[model]=subprocess.Popen([str(PYTHON),'baseline/scripts/launch_model.py','--model',model,
                '--sim-url',f'http://127.0.0.1:{port}/mcp','--root',str(root/'formal')],stdout=f,stderr=subprocess.STDOUT)
        (root/'processes.json').write_text(json.dumps({'supervisor':os.getpid(),'models':{m:p.pid for m,p in children.items()}},indent=2)+'\n')
        while True:
            running=[m for m,p in children.items() if p.poll() is None]
            failures={m:p.returncode for m,p in children.items() if p.poll() not in (None,0)}
            state('parallel_running' if running else ('blocked' if failures else 'complete'),running=running,failed=failures)
            if not running:break
            time.sleep(30)
        for f in handles:f.close()
    except Exception as exc:
        state('blocked',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
