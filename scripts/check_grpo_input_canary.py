"""Audit real input-v1 artifacts and complete training checkpoints before launch."""
import json
from pathlib import Path
import sys
from inspect_embodied_checkpoint import inspect

root=Path(sys.argv[1]); report={}
for role in ('alice','bob'):
    checkpoint,complete=inspect(root/'round-0000'/f'{role}_train',8)
    if not complete: raise SystemExit(f'{role}: complete checkpoint missing')
    validations=list((root/'round-0000'/f'{role}_train').glob('*/conditional_input_validation.jsonl'))
    rows=[json.loads(line) for p in validations for line in p.read_text().splitlines()]
    assert {x['rank'] for x in rows if x['prompt_and_loss_verified']}==set(range(8)),f'{role}: rank audit missing'
    artifacts=list((root/'round-0000'/f'{role}_rollout'/'rollouts'/role).glob('*.json'))
    turns=valid=sliding=0
    for p in artifacts:
        x=json.loads(p.read_text())
        assert x['input_protocol']=='openeta.two_tool_input.v1'
        assert not x['final_info'].get('infrastructure_error'),x['final_info']
        for record in x['visible_inputs']:
            turns+=1
            assert record['prompt_verified']
            assert record['prompt_token_ids']==record['actual_prompt_token_ids']
            assert len(record['response_token_ids'])==len(record['rollout_logprobs'])==sum(record['response_loss_mask'])
            h=record['packet']['history_decision_ids'];assert len(h)<=4
            sliding+=bool(h and h[0]>1)
            assert 1<=len(record['request']['images'])<=2
            for im in record['packet']['images']:assert im['width']==im['height']==256
            audit=json.loads(Path(record['audit_path']).read_text())
            assert audit['response_token_ids']==record['response_token_ids']
        valid+=sum(s['valid_action'] for s in x['steps'])
    assert turns,f'{role}: no real decisions'
    report[role]={'checkpoint':str(checkpoint),'episodes':len(artifacts),'decisions':turns,
                  'valid_decisions':valid,'sliding_decisions':sliding,'all_ranks_condition_and_loss_verified':True}
assert sum(item['sliding_decisions'] for item in report.values()) > 0, 'real sliding-window trajectory missing'
assert (root/'round-0000/bob_holdout_eval.json').is_file(),'holdout missing'
holdout=json.loads((root/'round-0000/bob_holdout_eval.json').read_text())
assert holdout['episodes']==holdout['tasks'] and holdout['episodes']>0
assert all(not receipt.get('infrastructure_error') for receipt in holdout['receipts'])
eval_artifacts=list((root/'round-0000/bob_eval_rollout/rollouts/bob').glob('*.json'))
assert len(eval_artifacts)==holdout['episodes'],'holdout audit artifacts missing'
eval_turns=0
for path in eval_artifacts:
    artifact=json.loads(path.read_text())
    assert artifact['input_protocol']=='openeta.two_tool_input.v1'
    assert not artifact['final_info'].get('infrastructure_error')
    assert artifact['visible_inputs']
    for record in artifact['visible_inputs']:
        assert record['prompt_verified'] and record['prompt_token_ids']==record['actual_prompt_token_ids']
        assert len(record['response_token_ids'])==len(record['rollout_logprobs'])==sum(record['response_loss_mask'])
        assert len(record['packet']['history_decision_ids'])<=4
        assert 1<=len(record['request']['images'])<=2
        assert all(im['width']==im['height']==256 for im in record['packet']['images'])
        eval_turns+=1
report['holdout']={'episodes':holdout['episodes'],'decisions':eval_turns,'inputs_verified':True,
                   'successes':holdout['successes']}
assert (root/'round-0000/summary.json').is_file(),'round summary missing'
(root/'input_acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
