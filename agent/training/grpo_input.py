"""GRPO adapter for the shared input builder; no independent camera selection."""
from copy import deepcopy
from pathlib import Path
import hashlib
import io
import json
import os
import uuid
import numpy as np
from PIL import Image
from agent.runtime import two_tool_input as shared


def execution_receipt(*, decision_id, command, executed_steps, error=None,
                      outcome_unknown=False, terminated=False, truncated=False):
    return {'turn':decision_id, 'tool':(command or {}).get('tool'),
            'arguments':{k:deepcopy(v) for k,v in (command or {}).items() if k in ('action','num_steps')},
            'valid':error is None and not outcome_unknown,
            'validation':{'valid':command is not None},
            'tool_request':bool(command) and (outcome_unknown or error is None),
            'executed_control_steps':None if outcome_unknown else executed_steps,
            'error':error,'terminated':terminated,'truncated':truncated}


def swift_request(package, thinking):
    messages=[]; images=[]
    for message in package['messages']:
        content=message['content']
        if isinstance(content,list):
            parts=[]
            for block in content:
                if block['type']=='text':parts.append(block['text'])
                elif block['type']=='image_url':
                    parts.append('<image>');images.append(block['image_url']['url'])
                else:raise ValueError('unsupported shared message block')
            content='\n'.join(parts)
        messages.append({'role':message['role'],'content':content})
    if images != [item['path'] for item in package['images']]:
        raise ValueError('shared image order mismatch')
    return {'messages':messages,'images':images,'chat_template_kwargs':{'enable_thinking':thinking}}


class InputBuilder:
    def __init__(self, profile, *, role, task, directory):
        self.profile=deepcopy(profile); self.role=role; self.task=task
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.current=None;self.previous=None;self.history=[]

    def observe(self,payload,*,observation_id,turn,new_observation=True):
        frames=[]
        for camera in payload.get('cameras',[]):
            im=Image.fromarray(np.asarray(camera['rgb'],dtype=np.uint8))
            stream=io.BytesIO();im.save(stream,format='PNG');data=stream.getvalue()
            sha=hashlib.sha256(data).hexdigest()
            name=f"{sha}-{camera['frame_id']}.png"
            path=self.directory/name
            if not path.is_file() or path.stat().st_size == 0:
                temporary = path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
                try:
                    temporary.write_bytes(data)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            frames.append(name)
        after=self.history[-1]['turn'] if self.history else None
        self.current=shared.snapshot(payload,frames,self.directory,self.profile,observation_id,after)
        self.turn=turn+1

    def record(self,raw,receipt):
        event=deepcopy(receipt);event['model_message']={'content':raw}
        self.history.append(event);self.previous=deepcopy(self.current)

    def build(self,*,remaining_turns,remaining_control_steps,generation_budget,context_limit,encode,enable_thinking=False):
        package, request, ids = shared.build_bounded_input(
            task=self.task, role=self.role, profile=self.profile, current=self.current,
            previous=self.previous, history=self.history, turn=self.turn,
            max_turns=self.turn+remaining_turns-1, remaining_steps=remaining_control_steps,
            generation_budget=generation_budget, context_limit=context_limit,
            render_request=lambda package: swift_request(package,enable_thinking), encode_request=encode)
        return {'packet':package,'request':request,'prompt_token_ids':ids,
                'prompt_token_count':len(ids),'adapter_version':'openeta.two_tool_input.swift.v1'}
