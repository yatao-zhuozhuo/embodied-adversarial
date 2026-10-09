"""Real reset and negative regression checks for the campaign-only raster repair."""
import asyncio,copy,json,os,sys,uuid
from pathlib import Path
C=Path(__file__).resolve().parent;V2=C.parents[1];sys.path.insert(0,str(V2.parents[1]))
os.environ.update(NO_PROXY='*',no_proxy='*',BASELINE_SNAPSHOT=str(C/'snapshot'))
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from baseline.v2.runtime import verify_paired_initial
from initial_guard import verify_accepted_initial,quantization_measurements
from agent.runtime.two_tool_input import project_observation
import numpy as np
from PIL import Image
D=C/'repairs/20261008-raster-v2';records=[]
def rejects(fn):
 try:fn()
 except (RuntimeError,AssertionError):return True
 raise AssertionError('Negative test incorrectly accepted')
async def probe(model,port):
 rows=json.loads((V2/'manifests/all.json').read_text())['episodes'];profiles=json.loads((V2/'configs/input_profiles.json').read_text())
 chosen=[r for r in rows if r['episode_id'] in ('v2-libero-006-s1','v2-libero-006-s2','v2-libero-007-s0','v2-libero-019-s2')]
 chosen += [next(r for r in rows if r['metadata']['benchmark']==b) for b in ('metaworld','maniskill3','robocasa')]
 async with streamablehttp_client(f'http://127.0.0.1:{port}/mcp') as (read,write,_):
  async with ClientSession(read,write) as session:
   await session.initialize()
   for row in chosen:
    sid='raster-repair-'+uuid.uuid4().hex;handle=None
    async def call(name,**kw):
     response=await session.call_tool(name,{'session_id':sid,**kw});assert not response.isError,str(response.content)
     data=response.structuredContent or json.loads(next(c.text for c in response.content if c.type=='text'));assert not data.get('error'),data.get('error');return data
    try:
     created=await call('create_env',env_id=row['env_id'],task=row['task'],seed=row['seed'],image_width=256,image_height=256,include_objects=False);handle=created['handle']
     raw=await call('reset_env',handle=handle,seed=row['seed'],settle=False)
     projected=project_observation(raw,D/'images'/model/row['episode_id'],0,profiles[row['env_id']])
     verify_accepted_initial(row,raw,projected)
     verify_paired_initial(D/'paired',row,projected)
     negative={}
     if row['episode_id']=='v2-libero-006-s1':
      altered=copy.deepcopy(projected);altered['robot']['end_effector_pose']['xyz'][0]+=0.001
      negative['robot_1mm']=rejects(lambda:verify_accepted_initial(row,raw,altered))
      altered=copy.deepcopy(projected);altered['cameras'].pop()
      negative['missing_camera']=rejects(lambda:verify_accepted_initial(row,raw,altered))
      a=np.zeros((256,256,3),dtype=np.uint8);b=a.copy();b[0,0,0]=2
      quantization_measurements(a,b,'libero')
      b[0,0,0]=3;negative['one_channel_delta_3']=rejects(lambda:quantization_measurements(a,b,'libero'))
      b[:]=1;negative['global_one_level_shift']=rejects(lambda:quantization_measurements(a,b,'libero'))
      b[:]=0;b[10:20,10:20]=2;negative['localized_patch']=rejects(lambda:quantization_measurements(a,b,'libero'))
     if row['metadata']['benchmark']=='robocasa':
      altered=copy.deepcopy(raw);altered['task']='wrong instruction'
      negative['wrong_instruction']=rejects(lambda:verify_accepted_initial(row,altered,projected))
     rec={'model':model,'episode_id':row['episode_id'],'passed':True,'negative_checks':negative};records.append(rec);print(json.dumps(rec),flush=True)
    finally:
     if handle:await call('close_env',handle=handle)
async def main():
 await asyncio.gather(probe('qwen35_4b',8230),probe('qwen35_9b',8231),probe('qwen38_27b',8226))
 (D/'validation.json').write_text(json.dumps({'passed':True,'records':records},indent=2)+'\n')
if __name__=='__main__':asyncio.run(main())
