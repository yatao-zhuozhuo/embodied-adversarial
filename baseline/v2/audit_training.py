"""Audit task-level separation and canonical LIBERO task definitions."""
import hashlib
import json
import re
from pathlib import Path
from select_tasks import V2, ROOT, LIBERO, canonical, write


def sha(data):return hashlib.sha256(data).hexdigest()


def parse_bddl(text):
    tokens=re.findall(r'\(|\)|[^\s()]+',text.lower())
    stack=[];root=[];current=root
    for token in tokens:
        if token=='(':
            node=[];current.append(node);stack.append(current);current=node
        elif token==')':current=stack.pop()
        else:current.append(token)
    assert not stack
    definition=root[0]
    blocks={n[0]:n[1:] for n in definition if isinstance(n,list) and n and str(n[0]).startswith(':')}
    blocks.pop(':language',None)
    return blocks


def task_definition(env_id):
    match=re.fullmatch(r'openeta/libero_(libero_\w+)_task(\d+)-v0',env_id)
    if not match:return None
    suite,index=match[1],int(match[2]);name=LIBERO[suite][index]
    path=ROOT/'baseline/.runtime/sources/LIBERO/libero/libero/bddl_files'/suite/(name+'.bddl')
    blocks=parse_bddl(path.read_text())
    return {'source':str(path.relative_to(ROOT)), 'sha256':sha(path.read_bytes()),
            'definition_sha256':sha(json.dumps(blocks,sort_keys=True).encode()),
            'goal_sha256':sha(json.dumps(blocks.get(':goal'),sort_keys=True).encode()),
            'goal':blocks.get(':goal')}


def main():
    project=ROOT.parents[1]
    paths=[project/'SFT-train/mixed-data/train.jsonl',project/'SFT-train/data/train.jsonl']
    sources=[];training=set()
    for path in paths:
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids={r['metadata']['env_id'] for r in rows};training.update(ids)
        sources.append({'path':str(path),'sha256':sha(path.read_bytes()),'samples':len(rows),
                        'episodes':len({r['episode_id'] for r in rows}),'task_count':len(ids)})
    train_keys={canonical(e) for e in training}
    definitions={e:task_definition(e) for e in training}
    tasks=json.loads((V2/'candidate_pool.json').read_text())['tasks']
    overlap=[];same_goal=[];records=[]
    for task in tasks:
        e=task['env_id'];definition=task_definition(e)
        record={'env_id':e,'canonical_key':canonical(e),'definition':definition}
        if canonical(e) in train_keys:overlap.append(e)
        if definition:
            identical=[k for k,d in definitions.items() if d and d['definition_sha256']==definition['definition_sha256']]
            if identical:overlap.append(e)
            related=[k for k,d in definitions.items() if d and d['goal_sha256']==definition['goal_sha256']]
            if related:same_goal.append({'env_id':e,'training_tasks':related,'classification':'shared_goal_predicate_different_full_task_definition'})
        records.append(record)
    result={'status':'passed' if not overlap else 'failed','training_sources':sources,
            'training_task_union_count':len(training),'canonical_or_definition_overlaps':sorted(set(overlap)),
            'shared_goal_families':same_goal,'candidate_definitions':records,
            'scope':'task-level held-out; shared manipulation primitives allowed; full task definitions excluding language are checked for LIBERO',
            'limitations':['Unrecorded model training sources cannot be audited; bind each evaluated checkpoint to its actual data manifest.',
                           'No claim of disjoint skill families or base-model pretraining decontamination.']}
    write('overlap_audit.json',result)
    assert not overlap,overlap
    print(json.dumps({'training_task_union':len(training),'candidate_tasks':len(tasks),'overlap':len(overlap),'shared_goal_families':len(same_goal)}))


if __name__=='__main__':main()
