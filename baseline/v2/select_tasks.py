"""Select v2 candidates without model results; keep all output under v2."""
import ast
import hashlib
import json
import re
import runpy
from pathlib import Path

V2 = Path(__file__).resolve().parent
ROOT = V2.parents[1]


def write(name, value):
    path = V2 / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def canonical(env_id):
    m = re.fullmatch(r'openeta/libero_(libero_\w+)_task(\d+)-v0', env_id)
    if m:
        return 'libero:' + LIBERO[m[1]][int(m[2])]
    return re.sub(r'^openeta/robocasa_(pretrain|target)_', 'robocasa:', env_id)


LIBERO = runpy.run_path(str(ROOT / 'baseline/.runtime/sources/LIBERO/libero/libero/benchmark/libero_suite_task_map.py'))['libero_task_map']


def main():
    legacy = json.loads((ROOT / 'baseline/manifests/all_40.json').read_text())['episodes']
    tasks = []
    # Labels describe relative requirements of our raw-action interface, not official tiers.
    tiers = {
        'libero': ['medium'] * 8 + ['hard'] * 2,
        'metaworld': ['easy','easy','medium','hard','hard','easy','medium','easy','hard','medium'],
        'maniskill3': ['medium','medium','medium','hard','hard','hard','easy','medium','easy','hard'],
        'robocasa': ['easy','medium','easy','easy','medium','hard','hard','hard','hard','hard'],
    }
    counts = {}
    for row in legacy:
        b = row['metadata']['benchmark']; i = counts.get(b, 0); counts[b] = i + 1
        tasks.append({'env_id':row['env_id'], 'task':row['task'], 'benchmark':b,
                      'difficulty':tiers[b][i], 'source':'legacy', 'metadata':row['metadata']})
    additions = [('libero_goal', i, d) for i,d in [(0,'easy'),(5,'easy'),(7,'easy'),(6,'medium')]]
    additions += [('libero_90', i, 'easy') for i in (0,6,22,33,39)]
    additions += [('libero_object',7,'medium')]
    for suite,i,d in additions:
        name = LIBERO[suite][i]
        task = re.sub(r'^[A-Z_]+SCENE\d+_', '', name).replace('_',' ')
        tasks.append({'env_id':f'openeta/libero_{suite}_task{i}-v0','task':task,
                      'benchmark':'libero','difficulty':d,'source':'new',
                      'metadata':{'suite':suite,'task_index':i}})
    descriptions = json.loads((ROOT/'sim/envs/metaworld/metaworld_config.json').read_text())['TASK_DESCRIPTIONS']
    for name,d in [('push','easy'),('drawer-close','easy'),('button-press','easy'),
                   ('button-press-topdown','easy'),('button-press-wall','medium'),
                   ('button-press-topdown-wall','medium'),('door-close','easy'),
                   ('handle-press','easy'),('handle-press-side','easy'),('handle-pull','medium')]:
        name += '-v3'
        tasks.append({'env_id':f'openeta/metaworld_50_{name}-v0','task':descriptions[name],
                      'benchmark':'metaworld','difficulty':d,'source':'new','metadata':{'native_task':name}})
    for name,desc,d in [('TurnFaucet-v1','Turn the faucet handle to open the faucet.','medium'),
                        ('PickSingleYCB-v1','Pick up the object and move it to the goal marked by the green sphere; keep the robot still at the end.','medium')]:
        tasks.append({'env_id':f'openeta/maniskill_{name}-v0','task':desc,
                      'benchmark':'maniskill3','difficulty':d,'source':'new','metadata':{'native_task':name}})
    registry = ROOT/'sim/venvs/robocasa/src/robocasa/robocasa/utils/dataset_registry.py'
    horizons = {}; target = {}
    for node in ast.parse(registry.read_text()).body:
        if not isinstance(node,ast.Assign) or not isinstance(node.value,ast.Call): continue
        name = getattr(node.targets[0],'id','')
        if name in ('ATOMIC_TASK_DATASETS','COMPOSITE_TASK_DATASETS'):
            for kw in node.value.keywords:
                horizons[kw.arg] = next(ast.literal_eval(v.value) for v in kw.value.keywords if v.arg=='horizon')
        if name=='TARGET_TASKS': target={k.arg:ast.literal_eval(k.value) for k in node.value.keywords}
    for name,d in [('CloseBlenderLid','medium'),('CloseFridge','easy'),('CloseToasterOvenDoor','easy'),
                   ('OpenStandMixerHead','easy'),('SlideDishwasherRack','easy'),('TurnOnElectricKettle','easy'),
                   ('TurnOnMicrowave','easy'),('TurnOnSinkFaucet','easy'),
                   ('PickPlaceCounterToStove','medium'),('PickPlaceDrawerToCounter','medium')]:
        assert name in target['atomic_seen']
        tasks.append({'env_id':f'openeta/robocasa_target_{name}-v0',
                      'task':f'Complete {name}; the official per-seed instruction must be frozen after reset.',
                      'benchmark':'robocasa','difficulty':d,'source':'new',
                      'metadata':{'native_task':name,'split':'target','task_family':'atomic','native_horizon':horizons[name]}})
    excluded = {canonical(t['env_id']) for t in json.loads((V2/'training_exclusions.json').read_text())['tasks']}
    assert len({t['env_id'] for t in tasks})==len(tasks)
    assert not excluded.intersection(canonical(t['env_id']) for t in tasks)
    episodes=[]
    for i,t in enumerate(tasks):
        t['task_id']=f"v2-{t['benchmark']}-{i:03d}"
        t['canonical_key']=canonical(t['env_id'])
        t['difficulty_source']='OpenETA requirement-based provisional v1'
        t['difficulty_reason']={'easy':'短流程、相对宽容的接触或移动操作',
            'medium':'抓取放置、空间定位、障碍或多阶段操作',
            'hard':'精密插入、工具、复杂堆叠或长流程组合'}[t['difficulty']]
        for j in range(3):
            seed=2026100800+i*10+j
            episodes.append({'episode_id':f"{t['task_id']}-s{j}",'env_id':t['env_id'],
                'task':t['task'],'seed':seed,'max_turns':100,'max_control_steps':1000,
                'max_tool_calls':200,'timeout_s':3600,'max_total_tokens':5000000,
                'metadata':{**t['metadata'],'benchmark':t['benchmark'],'task_id':t['task_id'],
                    'difficulty':t['difficulty'],'difficulty_source':t['difficulty_source'],
                    'selection_source':t['source'],'seed_index':j}})
    write('candidate_pool.json',{'status':'selected_pending_acceptance','tasks':tasks})
    write('manifests/candidates.json',{'episodes':episodes})
    print(json.dumps({'tasks':len(tasks),'episodes':len(episodes),'training_overlap':0}))


if __name__=='__main__':main()
