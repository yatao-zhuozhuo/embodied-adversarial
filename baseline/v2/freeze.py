"""Freeze only fully accepted tasks; every included task has three checked initial states."""
import collections
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from select_tasks import V2, ROOT, write


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (V2/'snapshot/lock.json').exists(), 'Already frozen; use a new version for changes'
    audit=json.loads((V2/'overlap_audit.json').read_text());assert audit['status']=='passed'
    for source in audit['training_sources']:
        assert sha(Path(source['path']))==source['sha256'],'Training data changed during selection'
    mani={}
    task_dir=next((ROOT/'sim/venvs/maniskill/lib').glob('python*/site-packages/mani_skill/envs/tasks/tabletop'))
    for p in task_dir.glob('*.py'):
        for name,horizon in re.findall(r'@register_env\(\s*"([^"]+)"[^)]*?max_episode_steps\s*=\s*(\d+)',p.read_text(),re.S):
            mani[name]={'horizon':int(horizon),'source':str(p.relative_to(ROOT)),'sha256':sha(p)}
    tasks=json.loads((V2/'candidate_pool.json').read_text())['tasks']
    candidate_rows=json.loads((V2/'manifests/candidates.json').read_text())['episodes']
    records={}
    for row in candidate_rows:
        path=V2/'acceptance'/(row['episode_id']+'.json')
        assert path.exists(), f'Acceptance not finished: {row["episode_id"]}'
        records[row['episode_id']]=json.loads(path.read_text())
    included=[];excluded=[];profiles={};initial={};references=[];rows=[]
    for task in tasks:
        episodes=[r for r in candidate_rows if r['metadata']['task_id']==task['task_id']]
        failures=[{'episode_id':r['episode_id'],'error':records[r['episode_id']].get('error','interrupted')} for r in episodes if not records[r['episode_id']]['passed']]
        if failures:
            excluded.append({'task_id':task['task_id'],'env_id':task['env_id'],'failures':failures});continue
        fingerprints=[]
        for row in episodes:
            record=records[row['episode_id']];b=row['metadata']['benchmark'];profile=record['profile']
            version_bench='maniskill' if b=='maniskill3' else b
            profile['source_versions']=json.loads((V2/f'configs/versions-{version_bench}.json').read_text())['packages']
            if row['env_id'] in profiles:assert profiles[row['env_id']]==profile
            profiles[row['env_id']]=profile
            images=record['initial']['cameras']
            fingerprints.append(tuple(c['pixel_sha256'] for c in images))
            initial[row['episode_id']]={'robot':record['initial']['robot'],'seed':row['seed'],
                'cameras':[{'frame_id':c['frame_id'],'pixel_sha256':c['pixel_sha256'],
                    'path':str(Path(c['image_path']).relative_to(V2)),'file_sha256':sha(Path(c['image_path']))} for c in images]}
            if b=='robocasa':
                row['task']=record['reset_task'];horizon=int(record['step_info']['horizon'])
                row['metadata']['instruction_source']='official reset task language, frozen separately per seed'
                row['max_turns']=max(100,math.ceil(horizon/5))
            elif b=='maniskill3':
                source=mani[row['metadata']['native_task']];horizon=source['horizon']
                row['metadata']['native_horizon_source']=source
            else:horizon=500 if b=='metaworld' else 1000
            row['max_control_steps']=horizon
            row['max_tool_calls']=2*row['max_turns']
            row['metadata']['native_horizon']=horizon
            row['metadata']['native_success_checker']=record['native_success']['source']
            row['metadata']['acceptance_sha256']=sha(V2/'acceptance'/(row['episode_id']+'.json'))
            row['metadata']['difficulty_calibration']='requirement-based; not a model-success-derived label'
            row['metadata']['budget_rule']='native control horizon; RoboCasa decision budget max(100,ceil(horizon/5)); others 100'
            if b=='metaworld':
                ref=V2/'reference/metaworld'/(row['episode_id']+'.json')
                assert ref.exists() and json.loads(ref.read_text())['success'],'Official MetaWorld policy acceptance missing'
                references.append({'episode_id':row['episode_id'],'sha256':sha(ref),**json.loads(ref.read_text())})
            rows.append(row)
        assert len(set(fingerprints))==3,f'Duplicate initial camera states: {task["task_id"]}'
        included.append(task)
    assert len(included)>=60,'Unexpectedly small accepted set; inspect failures'
    write('configs/input_profiles.json',profiles)
    write('manifests/initial_states.json',initial)
    write('manifests/all.json',{'schema_version':'openeta.baseline.v2.manifest.v1','episodes':rows})
    for bench in ('libero','metaworld','maniskill3','robocasa'):
        write('manifests/'+bench+'.json',{'episodes':[r for r in rows if r['metadata']['benchmark']==bench]})
    write('manifests/tasks.json',{'tasks':included})
    write('rejected_tasks.json',{'tasks':excluded,'selection_independent_of_model_outputs':True})
    report={'status':'ready_for_evaluation','task_count':len(included),'episode_count':len(rows),
            'added_task_count':sum(t['source']=='new' for t in included),
            'by_benchmark':dict(collections.Counter(t['benchmark'] for t in included)),
            'by_difficulty':dict(collections.Counter(t['difficulty'] for t in included)),
            'difficulty_by_benchmark':{b:dict(collections.Counter(t['difficulty'] for t in included if t['benchmark']==b)) for b in profiles_benches(included)},
            'excluded_task_count':len(excluded),'training_task_overlap':0,
            'acceptance':{'all_episodes_live_mcp':True,'reset_step_close':True,'xml_and_input_render':True,
                          'native_success_negative_check':True,'fresh_instance_reset_comparison':True,
                          'distinct_initial_image_states_per_task':3,'official_metaworld_reference_successes':len(references)},
            'reset_tolerances':{'robot_absolute':1e-7,'metaworld_pixel_mae_255':1.0,
                                'other_pixel_mae_255':0.001,'other_pixel_max_error_255':1.0},
            'goal_visibility_fix':['PickCube-v1 and PickSingleYCB-v1 use native render camera with goal sphere visible'],
            'limitations':['Difficulty labels are requirement-based; expert calibration across all four benchmarks is not complete.',
                           'Positive end-to-end scripted success was checked for MetaWorld; other benchmarks have live negative checks and native-checker/source audit, not a successful expert rollout for every task.',
                           'Reset comparison covers model-visible robot and RGB, not a byte-identical dump of all hidden simulator state.',
                           'No model inference performed during this acceptance; endpoint token/image smoke is required when starting a model.',
                           'Original native ManiSkill horizons remain unchanged and can be short; report horizon exhaustion separately.'],
            'official_reference':references}
    write('acceptance_report.json',report)
    # Lock shared evaluator dependencies and all v2 executable/configuration artifacts.
    files=set()
    for name in subprocess.check_output(['git','ls-files','adapter','agent','sim'],cwd=ROOT,text=True).splitlines():
        p=ROOT/name
        if p.is_file() and p.suffix in {'.py','.json','.md','.yaml'}:files.add(p)
    for p in V2.glob('*.py'):files.add(p)
    for p in V2.glob('*.sh'):files.add(p)
    for directory in ('configs','manifests','snapshot'):
        files.update(p for p in (V2/directory).rglob('*') if p.is_file() and p.name!='lock.json')
    for name in ('overlap_audit.json','training_exclusions.json','acceptance_report.json','rejected_tasks.json'):
        files.add(V2/name)
    files.add(ROOT/'baseline/scripts/inference_smoke.py')
    files.add(ROOT/'baseline/scripts/environment.sh')
    lock={'schema':'baseline.frozen_protocol.v1','protocol':'openeta.two_tool_xml.v1','input_protocol':'openeta.two_tool_input.v1',
          'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}}
    write('snapshot/lock.json',lock)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','task_count','episode_count','added_task_count','by_benchmark','by_difficulty')},ensure_ascii=False))


def profiles_benches(tasks):return sorted({t['benchmark'] for t in tasks})


if __name__=='__main__':main()
