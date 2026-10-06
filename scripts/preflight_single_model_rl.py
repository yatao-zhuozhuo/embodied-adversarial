"""Real reset/physics-fingerprint/profile/action canary for every native RL task."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.training.single_model_rollout import NativeTaskEnvironment, write_json
from agent.training.grpo_input import InputBuilder, execution_receipt
from agent.runtime.two_tool_xml import parse_decision, serialize_decision


def check(config, root):
    path=root/config['bench']/config['task_id']
    evidence=[]
    try:
        for repeat in range(2):
            env=NativeTaskEnvironment(config)
            try:
                evidence.append(env.fingerprint())
                if repeat and evidence[0]!=evidence[1]:
                    raise ValueError('Independent same-seed resets have different native state')
                builder=InputBuilder(env.profile,role='task solver',task=env.task,directory=path/'frames')
                builder.observe(env.observation,observation_id='initial',turn=0)
                before=builder.current
                action=[0.]*env.contract['action_dim']
                native=env.profile['native_control']
                if 'neutral_action' in native:
                    action=native['neutral_action']
                else:
                    action[3 if config['bench']=='metaworld' else 6]=native['gripper']['open_value']
                if config['bench']=='robocasa': action[-1]=-1.
                xml=serialize_decision('step_env',{'action':action,'num_steps':1})
                decision=parse_decision(xml,contract=env.contract,remaining_steps=400)['decision']
                observation,receipt,success=env.step(**decision['parameters'])
                builder.record(xml,execution_receipt(decision_id=1,command={'tool':'step_env',**decision['parameters']},executed_steps=receipt['executed_control_steps']))
                builder.observe(observation,observation_id='after',turn=1)
                write_json(path/f'reset-{repeat}.json',{'fingerprint':evidence[-1], 'profile':env.profile,
                    'before':before,'after':builder.current,'receipt':receipt,'success':success})
            finally: env.close()
        report={'status':'passed','task_id':config['task_id'],'bench':config['bench'],'fingerprints':evidence}
    except Exception as exc:
        report={'status':'failed','task_id':config['task_id'],'bench':config['bench'],'error':str(exc),'traceback':traceback.format_exc()}
    write_json(path/'preflight.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('traceback','fingerprints')}),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bench')
    a=p.parse_args()
    configs=[json.loads(line)['env_request'] for line in a.dataset.read_text().splitlines()]
    if a.bench: configs=[c for c in configs if c['bench']==a.bench]
    benches=sorted({c['bench'] for c in configs})
    def lane(bench): return [check(c,a.output) for c in configs if c['bench']==bench]
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports=[r for group in pool.map(lane,benches) for r in group]
    write_json(a.output/'summary.json',reports)
    raise SystemExit(0 if reports and all(r['status']=='passed' for r in reports) else 1)

if __name__=='__main__':main()
