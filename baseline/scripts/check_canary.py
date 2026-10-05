"""Require a real model-generated native action and native checker evidence."""
import json
import sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['status'] != 'infrastructure_error', r.get('error')
assert r.get('native_success_observed'), 'model did not execute a valid native action'
assert r.get('closed'), 'simulator cleanup failed'
assert any(any(v.get('attached') for v in t.get('details',{}).get('vision_attachments',[])) for t in r['turns']), 'no RGB attachment'
print(json.dumps({'model': r['model'], 'canary_passed': True, 'note': 'pipeline acceptance, not task success'}))
