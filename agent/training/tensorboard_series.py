"""One durable scalar timeline from the active curriculum's checkpoints."""
import json
import math
from numbers import Real
from pathlib import Path


def scalar_tags(row):
    tags={}
    for key,value in row.items():
        if key in ('step','global_step') or isinstance(value,bool):continue
        if isinstance(value,Real) and math.isfinite(float(value)):
            name=key if key.startswith(('train/','eval/')) else 'train/'+key
            tags[name]=float(value)
    reward=tags.get('train/reward',tags.get('train/rewards/NativeSuccessReward/mean'))
    if reward is not None:
        tags.update({'train/reward':reward,'train/success_rate':reward,'train/success_percent':100*reward})
    return tags


def history_points(state):
    points={}
    for row in state.get('log_history',[]):
        step=row.get('step',row.get('global_step'))
        if step is None and 'global_step/max_steps' in row:
            step=int(row['global_step/max_steps'].split('/')[0])
        if step is not None and 0<=int(step)<=state['global_step']:
            points.setdefault(str(int(step)),{}).update(scalar_tags(row))
    return {step:tags for step,tags in points.items() if tags}


def active_state(source_root):
    """Follow the active lineage instead of mixing rolled-back runs."""
    root=Path(source_root)
    pointer=root/'current_run.json'
    run=Path(json.loads(pointer.read_text())['run_root']) if pointer.exists() else root
    candidates=[]
    for path in (run/'formal/trainer').glob('*/v*/checkpoint-*/trainer_state.json'):
        # Save-every-step writes state last; model/optimizer are required too.
        if not all((path.parent/name).is_file() for name in ('adapter_model.safetensors','optimizer.pt','scheduler.pt')):
            continue
        try:state=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        candidates.append((state['global_step'],path,state))
    if candidates:
        _,path,state=max(candidates,key=lambda item:item[0])
        return path,state
    plan=run/'curriculum_plan.json'
    if plan.exists():
        checkpoint=json.loads(plan.read_text()).get('initial_checkpoint')
        if checkpoint:
            path=Path(checkpoint)/'trainer_state.json'
            return path,json.loads(path.read_text())
    return None,None


class ScalarSeries:
    def __init__(self,directory):
        from torch.utils.tensorboard import SummaryWriter
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.ledger_path=self.directory/'published.json'
        self.points=json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else {}
        self.writer=SummaryWriter(str(self.directory),purge_step=None if self.points else 0)
        self.writer_class=SummaryWriter

    def publish(self,state):
        points=history_points(state)
        # Returning to an older checkpoint invalidates events above it.
        if any(int(step)>state['global_step'] for step in self.points):
            self.writer.close()
            self.writer=self.writer_class(str(self.directory),purge_step=state['global_step']+1)
            self.points={step:tags for step,tags in self.points.items() if int(step)<=state['global_step']}
        count=0
        for step,tags in sorted(points.items(),key=lambda item:int(item[0])):
            old=self.points.get(step,{})
            for name,value in tags.items():
                if old.get(name)!=value:
                    self.writer.add_scalar(name,value,int(step));count+=1
            self.points[step]={**old,**tags}
        self.writer.flush()
        temporary=self.ledger_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.points,sort_keys=True)+'\n')
        temporary.replace(self.ledger_path)
        return count

    def close(self):self.writer.close()
