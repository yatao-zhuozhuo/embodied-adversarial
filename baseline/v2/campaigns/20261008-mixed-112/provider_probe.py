"""Exercise the real five-image processor, tokenizer and generation API."""
import concurrent.futures,json,sys,shutil
from pathlib import Path
C=Path(__file__).resolve().parent;ROOT=C.parents[3];sys.path.insert(0,str(ROOT))
from backend import build_backend
from baseline.v2.provider import count_prompt_tokens
from agent.runtime.two_tool_input import render_messages
from agent.backends.planner import PlannerBackendRequest
from agent.runtime.actions import PipelineStatus

def probe(model):
 package=json.loads((C/'provider-probe-input.json').read_text())
 backend=build_backend(model);tokens=count_prompt_tokens(backend,render_messages(package))
 response=backend.decide(PlannerBackendRequest(tool_context={'input_protocol':'openeta.two_tool_input.v1','input_package':package},system_prompt=package['system']))
 assert response.status==PipelineStatus.PLANNED,response.details
 assert len(response.details.get('vision_attachments',[]))==5
 assert all(x['attached'] for x in response.details['vision_attachments'])
 assert response.details['usage']['prompt_tokens']==tokens
 record={'passed':True,'model':backend.config.model,'prompt_tokens':tokens,'details':response.details,'answer':response.payload}
 (C/f'provider-probe-{model}.json').write_text(json.dumps(record,indent=2)+'\n')
 return {'model':model,'passed':True,'prompt_tokens':tokens}

if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for result in pool.map(probe,json.loads((C/'models.json').read_text())):print(json.dumps(result),flush=True)
