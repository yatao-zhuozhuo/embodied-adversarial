"""Read-only rollout/trainer summary; zero-variance groups are not learning evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent.training.single_model_rollout import write_json
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('trainer',type=Path)
p.add_argument('--output',type=Path)
a=p.parse_args()
counts=Counter(); tasks={}; ranks=set(); turns=0; valid=0; control=0; success=0; episodes=0; infra=0
for path in (a.trainer/'rollouts/episodes').glob('*.json'):
 d=json.loads(path.read_text()); info=d['final_info']; episodes+=1
 key=d['config']['bench']+':'+d['config']['task_id']
 task=tasks.setdefault(key,{'episodes':0,'success':0,'infrastructure_errors':0})
 task['episodes']+=1;task['success']+=int(info.get('success',False))
 if info.get('termination_reason')=='infrastructure_error':task['infrastructure_errors']+=1;infra+=1
 counts[info.get('termination_reason','unknown')]+=1
 success+=int(info.get('success',False));control+=info.get('control_steps',0)
 for turn in d['steps']:
  turns+=1;valid+=int(bool(turn.get('validation',{})) and turn.get('invalid_reason') is None)
for path in a.trainer.rglob('conditional_input_validation.jsonl'):
 for line in path.read_text().splitlines():
  d=json.loads(line)
  if d.get('prompt_and_loss_verified'):ranks.add(d['rank'])
checkpoints=[str(path) for path in a.trainer.rglob('checkpoint-*') if path.is_dir() and (path/'trainer_state.json').exists()]
report={'episodes':episodes,'success':success,'decisions':turns,'valid_xml':valid,
 'executed_control_steps':control,'infrastructure_errors':infra,'termination_reasons':dict(counts),
 'tasks':tasks,'verified_training_ranks':sorted(ranks),'checkpoints':checkpoints,
 'learning_signal':'inspect actual group reward variance and policy gradients; a checkpoint alone is insufficient'}
if a.output:write_json(a.output,report)
print(json.dumps(report,ensure_ascii=False,indent=2))
