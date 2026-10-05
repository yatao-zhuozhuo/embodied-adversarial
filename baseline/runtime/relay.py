"""Shared-storage DeepSeek transport for task containers without public egress.

Only fixed-protocol planner requests are accepted; the relay never runs tools.
Credentials remain in the existing user-provided file on the egress host.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from agent.backends.planner import PlannerBackend, PlannerBackendRequest, PlannerBackendResult
from agent.runtime.actions import PipelineStatus


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False) + '\n')
    tmp.replace(path)


class SharedStorageBackend(PlannerBackend):
    def __init__(self, config, root):
        self.config = config
        self.root = Path(root)

    def decide(self, request):
        ident = uuid.uuid4().hex
        reply = self.root / 'responses' / f'{ident}.json'
        atomic_json(self.root / 'requests' / f'{ident}.json', {
            'id': ident, 'created_at': time.time(), 'expires_at': time.time()+self.config.timeout_s,
            'request': asdict(request), 'timeout_s': self.config.timeout_s})
        deadline = time.monotonic()+self.config.timeout_s
        while time.monotonic() < deadline:
            if reply.exists():
                data = json.loads(reply.read_text())
                data['status'] = PipelineStatus(data['status'])
                return PlannerBackendResult(**data)
            time.sleep(0.2)
        raise TimeoutError('DeepSeek shared-storage relay timeout')

    def descriptor(self):
        return {'name': 'SharedStorageDeepSeek', 'implemented': True}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--credential-file',required=True)
    args=p.parse_args()
    for line in Path(args.credential_file).read_text().splitlines():
        if line.startswith('DEEPSEEK_API_KEY='):
            os.environ['DEEPSEEK_API_KEY']=line.split('=',1)[1].strip().strip('"\'')
    os.environ.pop('BASELINE_DEEPSEEK_RELAY_DIR',None)
    from baseline.runtime.provider import build_backend
    from baseline.scripts.inference_smoke import redact
    backend=build_backend('deepseek')
    root=Path(args.root);(root/'requests').mkdir(parents=True,exist_ok=True);(root/'responses').mkdir(exist_ok=True)
    with (root/'relay.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            atomic_json(root/'health.json',{'pid':os.getpid(),'time':time.time()})
            for path in sorted((root/'requests').glob('*.json')):
                output=root/'responses'/path.name
                if output.exists(): continue
                data=json.loads(path.read_text())
                if data.get('expires_at',0)<=time.time():
                    atomic_json(output,{'payload':'','status':'failed','details':{'error':'request expired before relay dispatch'}})
                    continue
                # No job can change the provider, key, model or decoding parameters.
                backend.config.timeout_s=max(1,min(180,float(data['expires_at'])-time.time()))
                try:
                    result=backend.decide(PlannerBackendRequest(**data['request']))
                    payload=asdict(result);payload['status']=result.status.value
                except Exception as exc:
                    payload={'payload':'','status':'failed','details':{'error':str(exc)}}
                atomic_json(output,redact(payload,[backend.config.api_key]))
            time.sleep(0.2)


if __name__=='__main__': main()
