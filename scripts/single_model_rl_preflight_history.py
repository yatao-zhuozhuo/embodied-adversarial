"""Reuse complete task checks only for identical data and unchanged environment code."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
ENVIRONMENT_PATHS=['sim','agent/runtime','agent/training/single_model_rollout.py',
    'agent/training/grpo_input.py','scripts/preflight_single_model_rl.py',
    'scripts/single_model_rl_env.sh','scripts/single_model_rl_yi_env.sh']


def validate_history(plan):
    history=Path(plan['preflight_history'])
    old=json.loads((history/'curriculum_plan.json').read_text())
    if old['config_sha256']!=plan['config_sha256']:
        raise ValueError('Cannot reuse preflight from a different curriculum')
    revision=plan['preflight_source_revision']
    subprocess.run(['git','-c',f'safe.directory={ROOT}','diff','--exit-code',revision,'--',*ENVIRONMENT_PATHS],
                   cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    canaries={}
    for stage in plan['stages']:
        previous=next(s for s in old['stages'] if s['name']==stage['name'])
        rows=[json.loads(line) for line in Path(stage['dataset']).read_text().splitlines()]
        old_rows=[json.loads(line) for line in Path(previous['dataset']).read_text().splitlines()]
        if rows!=old_rows:
            raise ValueError('Preflight dataset or seeds changed')
        expected={(r['env_request']['bench'],r['env_request']['task_id']) for r in rows}
        for rank in range(plan['batch']['nnodes']):
            reports=json.loads((history/f'nodes/{rank}'/stage['name']/'preflight/summary.json').read_text())
            passed={(r['bench'],r['task_id']) for r in reports if r['status']=='passed'}
            if len(reports)!=len(expected) or passed!=expected:
                raise ValueError('Preflight history is incomplete or contains failures')
        for row in rows:
            canaries.setdefault(row['env_request']['bench'],row)
    if len(canaries)!=4:
        raise ValueError('Expected all four native benchmarks')
    return list(canaries.values())
