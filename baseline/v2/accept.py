"""Real MCP acceptance for every candidate and seed, without querying a model."""
import argparse
import asyncio
import copy
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

V2 = Path(__file__).resolve().parent
ROOT = V2.parents[1]
sys.path.insert(0,str(ROOT))
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from agent.runtime.two_tool_input import (project_observation, control_contract, build_input,
    render_messages, execution_record, robot_delta)
from agent.runtime.two_tool_xml import parse_decision, serialize_decision
import numpy as np
from PIL import Image
from baseline.v2.comparison import robot_equal


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


async def main():
    ap=argparse.ArgumentParser();ap.add_argument('--benchmark');ap.add_argument('--sim-url',default='http://127.0.0.1:8220/mcp')
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--task-id')
    args=ap.parse_args()
    rows=json.loads((V2/'manifests/candidates.json').read_text())['episodes']
    rows.sort(key=lambda e:(e['metadata']['benchmark'],e['metadata']['task_id'],e['seed']))
    templates=json.loads((ROOT/'baseline/configs/input_profiles.json').read_text())
    async with streamablehttp_client(args.sim_url) as (read,send,_):
      async with ClientSession(read,send) as session:
        await session.initialize()
        for row in rows:
          if args.benchmark and row['metadata']['benchmark']!=args.benchmark:continue
          if args.task_id and row['metadata']['task_id']!=args.task_id:continue
          if row['metadata']['seed_index'] % args.shards != args.shard:continue
          path=V2/'acceptance'/f"{row['episode_id']}.json"
          if path.exists():continue
          start=time.monotonic();handle=None;sid='v2-accept-'+uuid.uuid4().hex
          record={'episode':row,'passed':False,'checks':{}}
          async def call(name,**kwargs):
            response=await asyncio.wait_for(session.call_tool(name,{'session_id':sid,**kwargs}),300)
            if response.isError:raise RuntimeError(str(response.content))
            data=response.structuredContent or json.loads(next(c.text for c in response.content if c.type=='text'))
            if data.get('error'):raise RuntimeError(data['error'])
            return data
          async def create():
            return await call('create_env',env_id=row['env_id'],task=row['task'],seed=row['seed'],
                              image_width=256,image_height=256,include_objects=False)
          try:
            created=await create();handle=created['handle'];record['created']=created
            bench=row['metadata']['benchmark']
            profile=copy.deepcopy(templates.get(row['env_id']) or next(p for p in templates.values() if p['benchmark']==bench))
            profile.update(id=row['env_id']+'.v2.rgb256',action_dim=created['action_dim'],
                control_mode=created['input_capabilities']['control_mode'],expected_capabilities=created['input_capabilities'])
            raw=await call('reset_env',handle=handle,seed=row['seed'],settle=False)
            if bench=='maniskill3':
                primary='render' if row['env_id'] in ('openeta/maniskill_PickCube-v1-v0','openeta/maniskill_PickSingleYCB-v1-v0') else 'base_camera'
                profile['cameras']=[{'id':c['frame_id'],'role':'wrist' if 'hand' in c['frame_id'] or 'wrist' in c['frame_id'] else 'primary',
                                    'size':[c['width'],c['height']]} for c in raw['cameras'] if c['frame_id'] in (primary,'hand_camera','wrist')]
                profile['cameras'].sort(key=lambda c:c['role']!='primary')
                if primary=='render':
                    assert any(c['id']=='render' for c in profile['cameras']), 'required visible-goal camera missing'
                    profile['goal_observability']='Native human-render camera exposes task goal sphere hidden in sensor RGB; no privileged numerical goal state supplied.'
                profile['image_limit']=len(profile['cameras'])+sum(c['role']=='primary' for c in profile['cameras'])
            assert profile['cameras'] and profile['image_limit']<=5
            contract=control_contract(created,profile)
            before=project_observation(raw,V2/'acceptance/images'/row['episode_id'],0,profile)
            for camera in before['cameras']:
                pixels=np.asarray(Image.open(camera['image_path']))
                assert float(pixels.std())>1,'blank camera'
            assert np.isfinite(before['robot']['end_effector_pose']['xyz']).all(),'invalid TCP'
            language=raw.get('task','')
            if bench=='robocasa':
                assert language and 'Complete ' not in language,'missing official instruction'
            record.update(profile=profile,contract=contract,initial=before,reset_task=language)
            record['checks'].update(reset=True,images=True,control_contract=True)
            action=[0.0]*created['action_dim'];action[0]=0.1
            xml=serialize_decision('tool_call','step_env','Interface acceptance probe.',{'action':action,'num_steps':2},created)
            validation=parse_decision(xml,created);assert validation['valid']
            receipt=await call('step_env',handle=handle,action=action,num_steps=2)
            assert receipt['steps_executed']==2,receipt
            assert not receipt['terminated'] and not receipt['truncated'],'initial/near-initial terminal state'
            native=receipt.get('info',{}).get('native_success')
            assert isinstance(native,dict) and type(native['success']) is bool,'missing native success'
            assert not native['success'],'already solved at initialization'
            after=project_observation(receipt['observation'],V2/'acceptance/images'/row['episode_id'],1,profile)
            execution=execution_record(0,validation,receipt=receipt)
            package=build_input(task=language if bench=='robocasa' else row['task'],role='robot_controller',system='XML only',
                profile=profile,contract=contract,current=after,previous=before,
                history=[{'answer':xml,'execution':execution,'robot_delta':robot_delta(before,after,contract)}],
                budget={'remaining_turns':99,'remaining_control_steps':998})
            messages=render_messages(package)
            record.update(native_success=native,step_info=receipt['info'],actual_steps=receipt['steps_executed'],
                image_count=len(package['images']),tcp_delta=robot_delta(before,after,contract))
            record['checks'].update(step=True,native_success_negative=True,input_render=True,xml_roundtrip=True)
            # Recreate to match actual independent model runs, not merely reset a used instance.
            await call('close_env',handle=handle);handle=None
            again=await create();handle=again['handle'];assert again['input_capabilities']==created['input_capabilities']
            repeat=await call('reset_env',handle=handle,seed=row['seed'],settle=False)
            repeat_p=project_observation(repeat,V2/'acceptance/repeat_images'/row['episode_id'],0,profile)
            record['repeat_initial']=repeat_p
            assert robot_equal(before['robot'],repeat_p['robot']),'robot reset changed beyond 1e-7 absolute tolerance'
            if bench=='robocasa':assert repeat.get('task')==language,'instruction reset changed'
            errors=[];max_errors=[]
            for a,b in zip(before['cameras'],repeat_p['cameras']):
                x=np.asarray(Image.open(a['image_path'])).astype(float)
                y=np.asarray(Image.open(b['image_path'])).astype(float)
                errors.append(float(np.abs(x-y).mean()))
                max_errors.append(float(np.abs(x-y).max()))
            assert max(errors)<=(1.0 if bench=='metaworld' else 0.001),f'initial pixels changed: {errors}'
            if bench!='metaworld':assert max(max_errors)<=1.0,f'pixel delta exceeds quantization: {max_errors}'
            record['repeat_pixel_max_error']=max_errors
            record['repeat_pixel_mae']=errors;record['checks']['fresh_instance_reproducibility']=True
            record['passed']=True
          except Exception as exc:
            record['error']=f'{type(exc).__name__}: {exc}'
          finally:
            if handle:
              try:
                closed=await call('close_env',handle=handle);assert closed.get('ok')
                record['checks']['close']=True
              except Exception as exc:
                record['passed']=False;record['close_error']=str(exc)
            record['elapsed_s']=round(time.monotonic()-start,2)
            write(path,record)
            print(json.dumps({'episode':row['episode_id'],'passed':record['passed'],'elapsed_s':record['elapsed_s'],
                              'error':record.get('error','')[:600]},ensure_ascii=False),flush=True)


if __name__=='__main__':asyncio.run(main())
