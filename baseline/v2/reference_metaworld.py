"""Check official scripted policies against the selected MetaWorld seeds."""
import json
import sys
import time
from pathlib import Path
V2=Path(__file__).resolve().parent;ROOT=V2.parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from sim.env_registry import _make_metaworld_direct
import metaworld.policies as policies

def main():
    rows=json.loads((V2/'manifests/candidates.json').read_text())['episodes']
    out=V2/'reference/metaworld';out.mkdir(parents=True,exist_ok=True)
    for row in rows:
        if row['metadata']['benchmark']!='metaworld':continue
        path=out/(row['episode_id']+'.json')
        if path.exists():continue
        name=row['metadata']['native_task'];env=None;start=time.monotonic()
        record={'episode_id':row['episode_id'],'seed':row['seed'],'success':False,
                'reference':'official MetaWorld scripted policy with privileged state',
                'same_native_action_interface':True,'model_input_equivalence':False}
        try:
            env=_make_metaworld_direct(name,seed=row['seed'],image_width=256,image_height=256)
            obs,_=env.reset(seed=row['seed'])
            policy=getattr(policies,'Sawyer'+''.join(x.capitalize() for x in name.split('-')[:-1])+'V3Policy')()
            for step in range(500):
                action=np.clip(policy.get_action(obs),env.action_space.low,env.action_space.high)
                obs,reward,terminated,truncated,info=env.step(action)
                record['control_steps']=step+1
                if bool(info.get('success',False)):
                    record['success']=True;break
                if terminated or truncated:break
        except Exception as exc:record['error']=f'{type(exc).__name__}: {exc}'
        finally:
            if env:env.close()
        record['elapsed_s']=round(time.monotonic()-start,2)
        path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)

if __name__=='__main__':main()
