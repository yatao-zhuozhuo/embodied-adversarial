"""Freeze all executable evaluation code, questions and generic guidance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
base = ROOT / 'baseline'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot', default='two-tool-input-v1')
args = parser.parse_args()
if Path(args.snapshot).name != args.snapshot or args.snapshot in {'.', '..'}:
    parser.error('snapshot must be a directory name')
snapshot = base / 'snapshots' / args.snapshot
files = set()
for name in subprocess.check_output(['git', 'ls-files', 'adapter', 'agent', 'sim'], cwd=ROOT, text=True).splitlines():
    p = ROOT / name
    if p.is_file() and p.suffix in {'.py', '.json', '.md', '.yaml'}:
        files.add(p)
for directory in ('runtime', str(snapshot.relative_to(base)), 'manifests', 'configs'):
    for p in (base / directory).rglob('*'):
        if p.is_file() and p.suffix in {'.py', '.json', '.md', '.txt', '.lock'} and p.name not in {'lock.json', 'reset_evidence.json'}:
            files.add(p)
files.add(ROOT / 'adapter/native_success.py')
files.add(ROOT / 'agent/runtime/two_tool_xml.py')
files.add(ROOT / 'agent/runtime/two_tool_input.py')
files.add(ROOT / 'sim/input_contract.py')
for name in ('run_job.sh', 'serve_vllm.sh', 'environment.sh', 'serve_simulator.sh', 'launch_model.py', 'check_canary.py', 'native_chain_smoke.py', 'run_inspire_campaign.py', 'run_xml_parallel_campaign.py', 'check_input_profiles.py', 'render_input_audit.py'):
    p = base / 'scripts' / name
    if p.exists(): files.add(p)
lock = {'schema': 'baseline.frozen_protocol.v1', 'protocol': 'openeta.two_tool_xml.v1', 'input_protocol':'openeta.two_tool_input.v1',
        'cross_episode_memory': False, 'skill_mutation': False, 'external_perception': False,
        'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
p = snapshot / 'lock.json'
p.write_text(json.dumps(lock, indent=2) + '\n')
print(json.dumps({'files': len(files), 'snapshot_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}))
