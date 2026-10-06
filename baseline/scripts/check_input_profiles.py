"""Live, no-policy profile acceptance; preserve raw audit separately from input."""
import asyncio, json, sys, uuid
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from agent.runtime.two_tool_input import project_observation,control_contract,build_input,execution_record,robot_delta,render_messages
from agent.runtime.two_tool_xml import parse_decision,serialize_decision

async def main():
 root=Path(__file__).resolve().parents[2]; output=root/'baseline/results/input-v1-preflight';output.mkdir(exist_ok=True)
 profiles=json.loads((root/'baseline/configs/input_profiles.json').read_text())
 rows=json.loads((root/'baseline/manifests/all_40.json').read_text())['episodes']
 async with streamablehttp_client('http://127.0.0.1:8210/mcp') as (read,write,_):
  async with ClientSession(read,write) as session:
   await session.initialize()
   for row in rows:
    path=output/(row['episode_id']+'.json')
    if path.exists() and json.loads(path.read_text()).get('passed'):continue
    sid='input-preflight-'+uuid.uuid4().hex;handle=None;record={'episode':row,'passed':False}
    async def call(name,**kwargs):
     r=await session.call_tool(name,{'session_id':sid,**kwargs})
     if r.isError:raise RuntimeError(str(r.content))
     d=r.structuredContent or json.loads(next(c.text for c in r.content if c.type=='text'))
     if d.get('error'):raise RuntimeError(d['error'])
     return d
    try:
     created=await call('create_env',env_id=row['env_id'],task=row['task'],seed=row['seed'],image_width=256,image_height=256,include_objects=False)
     handle=created['handle'];record['created']=created;profile=profiles[row['env_id']]
     contract=control_contract(created,profile);record['contract']=contract
     raw=await call('reset_env',handle=handle,seed=row['seed'],settle=False)
     before=project_observation(raw,output/row['episode_id'],0,profile)
     action=[0.0]*created['action_dim'];action[0]=0.1
     answer=serialize_decision('tool_call','step_env','Small probe.',{'action':action,'num_steps':2},created)
     validation=parse_decision(answer,created)
     receipt=await call('step_env',handle=handle,action=action,num_steps=2)
     after=project_observation(receipt['observation'],output/row['episode_id'],1,profile)
     execution=execution_record(0,validation,receipt=receipt)
     package=build_input(task=row['task'],role='robot_controller',system='XML only',profile=profile,contract=contract,current=after,previous=before,history=[{'answer':answer,'execution':execution,'robot_delta':robot_delta(before,after,contract)}],budget={'remaining_turns':99,'remaining_control_steps':998})
     rendered=render_messages(package)
     record.update(passed=True,input_package=package,attached_images=sum(p.get('type')=='image_url' for p in rendered[-1]['content']),actual_steps=receipt['steps_executed'])
    except Exception as exc:record['error']=f'{type(exc).__name__}: {exc}'
    finally:
     if handle:await call('close_env',handle=handle)
    path.write_text(json.dumps(record,indent=2));print(json.dumps({'episode':row['episode_id'],'passed':record['passed'],'error':record.get('error')}),flush=True)
    if not record['passed']:break

if __name__=='__main__':asyncio.run(main())
