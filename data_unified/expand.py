"""Revision 5: expand LIBERO / MetaWorld while preserving existing case bindings."""
import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from data_unified.build import cache_inventory, canonical
from data_unified.reduce import cache_geometry, choose, read, write
import data_unified.reduce as geometry

HERE=Path(__file__).resolve().parent
POOL=HERE/'expansion_candidates/v5'
OLD=HERE/'revisions/v4'
ADDITIONS={
 'libero': [('libero_90_task8','open_and_place'),('libero_90_task10','cabinet_top'),('libero_90_task28','drawer_close'),('libero_90_task30','plate_placement'),('libero_90_task41','pan_placement'),('libero_90_task58','tray_placement'),('libero_90_task59','tray_placement'),('libero_90_task73','caddy_front'),('libero_90_task77','caddy_back'),('libero_object_task3','object_pick')],
 'metaworld': [('button-press-topdown-wall-v3','obstacle_button'),('door-open-v3','articulated_open'),('drawer-close-v3','drawer_close'),('handle-press-side-v3','side_handle_press'),('handle-pull-v3','handle_pull'),('pick-place-wall-v3','obstacle_pick_place'),('push-back-v3','push_back'),('plate-slide-side-v3','side_slide'),('peg-unplug-side-v3','unplug'),('stick-pull-v3','tool_pull')]
}
POLICIES=['SawyerButtonPressTopdownWallV3Policy','SawyerDoorOpenV3Policy','SawyerDrawerCloseV3Policy','SawyerHandlePressSideV3Policy','SawyerHandlePullV3Policy','SawyerPickPlaceWallV3Policy','SawyerPushBackV3Policy','SawyerPlateSlideSideV3Policy','SawyerPegUnplugSideV3Policy','SawyerStickPullV3Policy']
PHASES=['sft_train','rl_train','validation','eval_seen']

def seed(cid):
 return int(hashlib.sha256(('openeta-v5:'+cid).encode()).hexdigest()[:8],16)%2**31

def prepare():
 old=read(OLD/'task_catalog.json'); history=read(OLD/'catalogs/historical_evidence.json')
 holdout=read(HERE.parent/'SFT_data/configs/two_tool_test_holdout.v1.json')
 tasks=[]; cases=[]; inventories={}
 for bench,added in ADDITIONS.items():
  template=next(t for t in old['tasks'] if t['bench']==bench and t['role']=='shared')
  catalog={t['task_slug']:t for t in read(HERE/f'catalogs/{bench}.json')['tasks'] if bench!='metaworld' or t['suite']=='metaworld_50'}
  for index,(slug,family) in enumerate(added):
   row=catalog[slug];key=f'{bench}/{slug}';task=copy.deepcopy(template)
   task.update(task_key=key,task_id=slug,env_id=row['id'],task=row['task_description'],family=family,
               selection_reason='v5 expansion: installed official cache / scripted expert with complementary skill and scene coverage.',
               historical_exposure=history['training'].get(key,{}),legacy_holdout=False,horizon=row['max_episode_steps'])
   task['environment_config']['model_scope']='frozen_cached_episode_scene' if bench=='libero' else 'native_task_distribution'
   task['legacy_holdout']=slug in holdout.get('canonical_excluded_task_keys',{}).get(bench,[]) or task['task'] in holdout['tasks'][bench]
   if bench=='metaworld':task['expert_policy']=POLICIES[index]
   if bench=='libero':
    records=cache_inventory({**task,'catalog':row});inventories[key]=records
    if len(records)<14:raise ValueError('Insufficient cache: '+key)
    task.update(cache_states=len(records),cache_unique_states=len({r['state_fingerprint'] for r in records}))
    for r in records:cache_geometry(r)
   else:
    task.update(cache_states=0,cache_unique_states=0)
    for phase in PHASES:
     for i in range(7):
      cid=f'{key}/{phase}/{i:03d}'
      cases.append(dict(case_id=cid,task_key=key,bench=bench,task_id=slug,env_id=row['id'],phase=phase,seed=seed(cid),
                        seed_role='environment_seed',cached_episode_seed=None,task=task['task'],
                        initial_state={'kind':'native_seed','state_fingerprint':None},runtime_verified=False))
   tasks.append(task)
 write(POOL/'task_catalog.json',{'tasks':tasks})
 (POOL/'initial_states.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases))
 write(POOL/'inventories.json',inventories)
 write(POOL/'cache_geometry.json',geometry._geometry_cache)
 print('Prepared',len(tasks),'added tasks;',len(cases),'native candidates')

def build():
 config=read(HERE/'selection.v1.json'); active=config['active_benches'];old=read(OLD/'task_catalog.json')
 tasks=[t for t in old['tasks'] if t['bench'] in active];tasks+=read(POOL/'task_catalog.json')['tasks']
 cases=[c for c in map(json.loads,(OLD/'initial_states.jsonl').read_text().splitlines()) if c['bench'] in active]
 refs=[r for r in map(json.loads,(OLD/'source_refs.jsonl').read_text().splitlines()) if r['task_key'].split('/')[0] in active]
 inventories=read(POOL/'inventories.json');geometry._geometry_cache=read(POOL/'cache_geometry.json')
 candidates=list(map(json.loads,(POOL/'initial_states.jsonl').read_text().splitlines()))
 historical=read(OLD/'audit.json')['known_historical_demo_indices_protected_from_test']; selections=[]
 for t in read(POOL/'task_catalog.json')['tasks']:
  key=t['task_key'];picked=[]
  if t['bench']=='libero':
   records=inventories[key]; byfp={r['state_fingerprint']:r for r in records}
   protected={r['state_fingerprint'] for r in records if r['reference_demo_index'] in historical.get(key,[])}
   for phase in PHASES:
    eligible=[r for fp,r in byfp.items() if fp not in {c['initial_state']['state_fingerprint'] for c in picked}
              and (phase in ('sft_train','rl_train') or fp not in protected)]
    pool=[];features={c['case_id']:cache_geometry(c['_ref']) for c in picked}
    for r in eligible:
     cid=f"{key}/{phase}/cache-{r['reference_demo_index']:03d}"
     c=dict(case_id=cid,task_key=key,bench=t['bench'],task_id=t['task_id'],env_id=t['env_id'],phase=phase,seed=seed(cid),seed_role='pre_restore_seed',
            cached_episode_seed=None,task=t['task'],initial_state={'kind':'reference_demo','reference_demo_index':r['reference_demo_index'],
            'source_ref_id':cid,'state_fingerprint':r['state_fingerprint']},runtime_verified=False,_ref=r)
     pool.append(c);features[cid]=cache_geometry(r)
    chosen,method=choose(pool,config['states_per_task'][phase],features,picked)
    picked+=chosen;selections.append(dict(task_key=key,phase=phase,method=method,candidate_count=len(pool),case_ids=[c['case_id'] for c in chosen]))
   for c in picked:
    r=c.pop('_ref');refs.append(dict(source_ref_id=c['case_id'],task_key=key,phase=c['phase'],**r));c['diversity_selection_method']='task_geometry_maximin'
   write(HERE/'inventory/libero'/f"{t['task_id']}.json",dict(task_key=key,records=records))
  else:
   observed=read(HERE/'diversity_candidates'/key/'features.json')['records']
   valid={r['case_id']:r for r in observed if r.get('capture_success') and not r.get('initial_success')}
   features={cid:r['task_state'] for cid,r in valid.items()}
   for phase in PHASES:
    pool=[c for c in candidates if c['task_key']==key and c['phase']==phase and c['case_id'] in valid]
    if len(pool)<config['states_per_task'][phase]:raise ValueError('Insufficient measured states: '+key+'/'+phase)
    chosen,method=choose(pool,config['states_per_task'][phase],features,picked);picked+=chosen
    selections.append(dict(task_key=key,phase=phase,method=method,candidate_count=len(pool),case_ids=[c['case_id'] for c in chosen]))
   for c in picked:c['diversity_selection_method']='task_geometry_maximin'
  cases+=picked
 old.update(split_revision=5,tasks=tasks,task_count=len(tasks),shared_task_count=sum(t['role']=='shared' for t in tasks),
            generalization_task_count=sum(t['role']=='generalization' for t in tasks),active_benches=active,
            shared_tasks_per_bench=config['shared_tasks_per_bench'],generalization_tasks_per_bench=5)
 write(HERE/'task_catalog.json',old)
 for name,rows in [('initial_states.jsonl',cases),('source_refs.jsonl',refs)]:
  (HERE/name).write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in rows))
 for phase in config['states_per_task']:
  manifest=read(OLD/'manifests'/f'{phase}.json');manifest.update(split_revision=5,cases=[c for c in cases if c['phase']==phase],
   tasks=[t for t in tasks if t['role']==('generalization' if phase=='eval_generalization' else 'shared')])
  write(HERE/'manifests'/f'{phase}.json',manifest)
 audit=read(OLD/'audit.json');audit.update(task_counts={b:dict(Counter(t['role'] for t in tasks if t['bench']==b)) for b in active},
  case_counts=dict(Counter(c['phase'] for c in cases)),cached_case_count=len(refs),native_seed_case_count=len(cases)-len(refs),
  cache_only_task_count=sum(bool(t['cache_states']) for t in tasks if t['role']=='shared'),
  missing_sft_cache=[t['task_key'] for t in tasks if t['role']=='shared' and not t['cache_states']],
  legacy_holdout_now_shared=[t['task_key'] for t in tasks if t['role']=='shared' and t['legacy_holdout']],
  active_benches=active,inactive_benches=[b for b in ['libero','metaworld','maniskill','robocasa'] if b not in active])
 write(HERE/'audit.json',audit)
 write(HERE/'expansion-record.v5.json',dict(split_revision=5,active_benches=active,added_tasks=ADDITIONS,selections=selections,
  preserved_case_count=sum(c['bench'] in active for c in map(json.loads,(OLD/'initial_states.jsonl').read_text().splitlines())),
  previous_revision='revisions/v4',policy='Existing task and case bindings are preserved; new states use measured / cached task geometry.'))
 write(HERE/'diversity_features/cache_geometry.json',geometry._geometry_cache)
 paths=[HERE/p for p in read(OLD/'checksums.json')]
 paths += [HERE/'expand.py',HERE/'expansion-record.v5.json']
 paths += sorted(POOL.glob('*.json'))+[POOL/'initial_states.jsonl']
 paths += list((HERE/'inventory/libero').glob('*.json'))+list((HERE/'diversity_candidates/metaworld').glob('*/features.json'))
 write(HERE/'checksums.json',{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
 print(json.dumps(audit['case_counts']))

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
 prepare() if args.prepare else build()
