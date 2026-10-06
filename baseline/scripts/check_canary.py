"""Require a real model-generated native action and native checker evidence."""
import json
import sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['protocol'] == 'openeta.two_tool_xml.v1'
assert r['status'] != 'infrastructure_error', r.get('error')
assert r.get('native_success_observed'), 'model did not execute a valid native action'
assert r.get('closed'), 'simulator cleanup failed'
assert any(any(v.get('attached') for v in t.get('details',{}).get('vision_attachments',[])) for t in r['turns']), 'no RGB attachment'

executed = 0
for turn in r['turns']:
    validation = turn.get('validation', {})
    request = turn.get('mcp_request')
    if request:
        assert validation.get('valid') and not validation.get('repair_applied')
        decision = turn['parsed_decision']
        assert request['name'] == decision['name']
        if request['name'] == 'step_env':
            assert request['parameters']['action'] == decision['parameters']['action']
            assert request['parameters']['num_steps'] == decision['parameters']['num_steps']
            assert 0 <= turn['steps_executed'] <= decision['parameters']['num_steps']
            assert turn['observation_before']['vision_image_paths']
            assert turn['observation_after']['vision_image_paths']
            executed += turn['steps_executed']
    if not validation.get('valid'):
        assert not request and turn.get('steps_executed', 0) == 0
assert executed > 0, 'no real control steps executed'
assert executed == r['native_steps_executed']
print(json.dumps({'model': r['model'], 'canary_passed': True, 'note': 'pipeline acceptance, not task success'}))
