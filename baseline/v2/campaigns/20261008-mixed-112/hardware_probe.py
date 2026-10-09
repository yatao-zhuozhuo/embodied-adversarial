"""Exercise frozen profiles and startup-state guards on the final v2 simulator."""
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
C=Path(__file__).resolve().parent;V2=C.parents[1];sys.path.insert(0,str(V2.parents[1]))
os.environ['NO_PROXY']='*';os.environ['no_proxy']='*'
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from baseline.v2.runtime import verify_snapshot
from initial_guard import verify_accepted_initial
from agent.runtime.two_tool_input import project_observation,control_contract,build_input,render_messages


async def main():
    lock=verify_snapshot()
    rows=json.loads((V2/'manifests/all.json').read_text())['episodes']
    profiles=json.loads((V2/'configs/input_profiles.json').read_text())
    chosen=[next(r for r in rows if r['metadata']['benchmark']==b) for b in ('libero','metaworld','maniskill3','robocasa')]
    chosen += [next(r for r in rows if r['env_id']==e) for e in ('openeta/maniskill_PickCube-v1-v0','openeta/maniskill_PickSingleYCB-v1-v0')]
    records=[]
    async with streamablehttp_client('http://127.0.0.1:8226/mcp') as (read,write,_):
      async with ClientSession(read,write) as session:
        await session.initialize()
        for row in chosen:
            sid='v2-final-'+uuid.uuid4().hex;handle=None
            async def call(name,**kw):
                response=await session.call_tool(name,{'session_id':sid,**kw})
                assert not response.isError,str(response.content)
                data=response.structuredContent or json.loads(next(c.text for c in response.content if c.type=='text'))
                assert not data.get('error'),data.get('error')
                return data
            try:
                created=await call('create_env',env_id=row['env_id'],task=row['task'],seed=row['seed'],image_width=256,image_height=256,include_objects=False)
                handle=created['handle'];profile=profiles[row['env_id']]
                contract=control_contract(created,profile)
                raw=await call('reset_env',handle=handle,seed=row['seed'],settle=False)
                projected=project_observation(raw,C/'hardware_probe_images'/row['episode_id'],0,profile)
                verify_accepted_initial(row,raw,projected)
                package=build_input(task=row['task'],role='robot_controller',system=(V2/'snapshot/system.txt').read_text(),
                    profile=profile,contract=contract,current=projected,previous=None,history=[],
                    budget={'remaining_turns':row['max_turns'],'remaining_control_steps':row['max_control_steps']})
                render_messages(package)
                records.append({'episode_id':row['episode_id'],'passed':True,'images':len(package['images'])})
                print(json.dumps(records[-1]),flush=True)
            finally:
                if handle:await call('close_env',handle=handle)
    (C/'hardware_probe_report.json').write_text(json.dumps({'passed':True,'snapshot_sha256':lock,'records':records},indent=2)+'\n')


if __name__=='__main__':asyncio.run(main())
