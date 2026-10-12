"""Revision 4: reduce immutable v3 pools with farthest-point geometry selection."""
import gzip
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def read(p):
    return json.loads(p.read_text())


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def mujoco_geometry(state, xml):
    """Extract non-robot qpos; omit clock, velocities and robot joint noise."""
    tree = ET.fromstring(xml)
    fields, offset = {}, 1
    for joint in tree.find('worldbody').iter():
        if joint.tag not in ('joint', 'freejoint'):
            continue
        kind = 'free' if joint.tag == 'freejoint' else joint.get('type', 'hinge')
        size = {'free':7, 'ball':4}.get(kind,1)
        name = joint.get('name', f'joint_{offset}')
        if not name.startswith(('robot', 'gripper')):
            fields['joint/'+name] = np.asarray(state[offset:offset+size]).tolist()
        offset += size
    if offset > len(state):
        raise ValueError('XML qpos layout exceeds saved state')
    return fields


_h5_handle = None


def _h5_file(path):
    global _h5_handle
    if _h5_handle is None or Path(_h5_handle.filename) != path:
        if _h5_handle is not None:
            _h5_handle.close()
        _h5_handle = h5py.File(path,'r')
    return _h5_handle


def _cache_geometry(ref):
    p = Path(ref['dataset_uri'])
    if not p.is_absolute():
        p = ROOT/p
    index = ref['reference_demo_index']
    if ref['kind'] == 'libero_demo_state':
        g = _h5_file(p)[f'data/demo_{index}']
        return mujoco_geometry(g.attrs['init_state'],g.attrs['model_file'])
    if ref['kind'] == 'robocasa_demo_state':
        extra = p/f'extras/episode_{index:06d}'
        with np.load(extra/'states.npz') as f:
            state = f['states'][0]
        with gzip.open(extra/'model.xml.gz','rt') as f:
            fields = mujoco_geometry(state,f.read())
        meta = read(extra/'ep_meta.json')
        fields['scene/layout'] = str(meta['layout_id'])
        fields['scene/style'] = str(meta['style_id'])
        return fields
    fields = {}
    def visit(name,obj):
        if isinstance(obj,h5py.Dataset) and name.startswith('actors/') and 'table' not in name:
            fields[name] = np.asarray(obj[0]).reshape(-1)[:7].tolist()
    _h5_file(p)[f'traj_{index}/env_states'].visititems(visit)
    return fields


_geometry_cache = None


def cache_geometry(ref):
    global _geometry_cache
    p = HERE/'diversity_features/cache_geometry.json'
    if _geometry_cache is None:
        _geometry_cache = read(p) if p.is_file() else {}
    key = ref['state_fingerprint'] + ':' + ref.get('model_xml_sha256',ref.get('metadata_sha256',''))
    if key not in _geometry_cache:
        _geometry_cache[key] = _cache_geometry(ref)
    return _geometry_cache[key]


def warm_cache():
    for p in sorted((HERE/'revisions/v3/inventory').glob('*/*.json')):
        for ref in read(p)['records']:
            cache_geometry(ref)
        print(p.stem,flush=True)
    write(HERE/'diversity_features/cache_geometry.json',_geometry_cache)


def distances(features):
    """Scale numeric coordinates by pool spread; categorical mismatch is 1.

    Constant and sub-micrometre coordinates do not add diversity. Rotation
    quaternions are canonicalized to avoid q/-q duplicate orientations.
    """
    rows = []
    for feature in features:
        flat = {}
        for key,value in feature.items():
            try:
                array = np.asarray(value,dtype=float).reshape(-1)
                if len(array) == 7 and ('actor' in key or 'joint' in key):
                    if array[3] < 0:
                        array[3:] *= -1
                for i,v in enumerate(array):
                    flat[f'{key}/{i}'] = float(v)
            except (ValueError,TypeError):
                flat[key] = str(value)
        rows.append(flat)
    keys = sorted(set.union(*(set(r) for r in rows)))
    columns = []
    for key in keys:
        values = [r.get(key) for r in rows]
        if all(isinstance(v,(float,int)) for v in values):
            col = np.asarray(values)
            spread = np.ptp(col)
            if spread > 1e-4:
                # Do not amplify tiny pose jitter into a full-scale difference.
                floor = 0.02
                if ('joint/' in key and sum(k.startswith(key.rsplit('/',1)[0]+'/') for k in rows[0]) == 1):
                    floor = 0.1
                columns.append((col-col.min())/max(spread,floor))
        elif len(set(map(str,values))) > 1:
            for value in sorted(set(map(str,values))):
                columns.append(np.array([str(v)==value for v in values],dtype=float)/np.sqrt(2))
    matrix = np.array(columns).T if columns else np.zeros((len(rows),1))
    return np.sqrt(np.mean((matrix[:,None,:]-matrix[None,:,:])**2,axis=-1))


def choose(pool, count, features, selected):
    measured = [c for c in pool if c['case_id'] in features]
    if len(measured) >= count:
        all_rows = selected + measured
        metric = distances([features[c['case_id']] for c in all_rows])
        chosen = list(range(len(selected)))
        remaining = list(range(len(selected),len(all_rows)))
        while len(chosen) < len(selected)+count:
            # Start at the most distant endpoint; thereafter maximize the
            # nearest selected-state distance across all phases of this task.
            scores = metric[remaining][:,chosen].min(axis=1) if chosen else metric[remaining].max(axis=1)
            take = remaining[int(np.argmax(scores))]
            chosen.append(take)
            remaining.remove(take)
        return [all_rows[i] for i in chosen[len(selected):]], 'task_geometry_maximin'
    # No assertion about geometric separation for uncaptured native tasks.
    # Spread ordinal candidates deterministically; flag this weaker fallback.
    ids = np.linspace(0,len(pool)-1,count,dtype=int)
    return [pool[i] for i in ids], 'unmeasured_candidate_spacing'


def build():
    old = HERE/'revisions/v3'
    config = read(HERE/'selection.v1.json')
    catalog = read(old/'task_catalog.json')
    old_cases = list(map(json.loads,(old/'initial_states.jsonl').read_text().splitlines()))
    old_refs = {r['source_ref_id']:r for r in map(json.loads,(old/'source_refs.jsonl').read_text().splitlines())}
    runtime = read(old/'runtime_checks/index.json')
    captures = dict(runtime['cases'])
    for p in sorted((HERE/'runtime_checks').glob('*/*/results.json')):
        for r in read(p)['records']:
            if r.get('capture_success'):
                captures[r['case_id']]=r
    for p in sorted((HERE/'diversity_candidates').glob('*/*/features.json')):
        for r in read(p)['records']:
            if r.get('capture_success') and not r.get('initial_success'):
                captures.setdefault(r['case_id'],r)
    features = {key:r['task_state'] for key,r in captures.items() if r.get('capture_success') and not r.get('initial_success')}
    grouped = defaultdict(lambda:defaultdict(list))
    for c in old_cases:
        grouped[c['task_key']][c['phase']].append(c)
    cases,refs,report = [],[],[]
    for task in catalog['tasks']:
        key = task['task_key']
        pools = grouped[key]
        phases = ['sft_train','rl_train','eval_seen','validation'] if task['role']=='shared' else ['eval_generalization']
        local_features = {c['case_id']:features[c['case_id']] for rows in pools.values() for c in rows if c['case_id'] in features}
        for rows in pools.values():
            for c in rows:
                if c['initial_state']['kind']=='reference_demo':
                    local_features[c['case_id']] = cache_geometry(old_refs[c['case_id']])
        if task['role']=='shared' and not task['cache_states']:
            c = dict(pools['validation'][0])
            c.update(case_id=key+'/validation/001',seed=int(hashlib.sha256(f"20261010:{key}:validation:1".encode()).hexdigest()[:8],16)%2**31)
            pools['validation'].append(c)
            if c['case_id'] in features:
                local_features[c['case_id']] = features[c['case_id']]
        picked=[]
        for phase in phases:
            pool = pools[phase]
            if phase=='validation' and task['cache_states']:
                # Reduced training/test pools leave independent, unexposed cache
                # states available. Never repurpose an historically exposed state.
                used = {c['initial_state']['state_fingerprint'] for c in picked}
                historic = set(read(old/'audit.json')['known_historical_demo_indices_protected_from_test'].get(key,[]))
                records = read(old/'inventory'/task['bench']/(task['task_id']+'.json'))['records']
                protected_fp = {r['state_fingerprint'] for r in records if r['reference_demo_index'] in historic}
                # v3 collectors are still active from their own frozen snapshots.
                # Even unselected v3 training states must remain training-only.
                protected_fp |= {c['initial_state']['state_fingerprint']
                                 for train_phase in ('sft_train','rl_train') for c in pools[train_phase]
                                 if c['initial_state']['kind']=='reference_demo'}
                pool=[]
                for r in records:
                    if r['state_fingerprint'] in used|protected_fp:
                        continue
                    i=r['reference_demo_index']; cid=f'{key}/validation/cache-{i:03d}'
                    c={**pools['validation'][0], 'case_id':cid,'seed':int(hashlib.sha256(cid.encode()).hexdigest()[:8],16)%2**31,
                       'seed_role':'pre_restore_seed','cached_episode_seed':r.get('episode_seed'),
                       'initial_state':{'kind':'reference_demo','reference_demo_index':i,'source_ref_id':cid,'state_fingerprint':r['state_fingerprint']}}
                    old_refs[cid]={'source_ref_id':cid,'task_key':key,'phase':phase,**r}
                    local_features[cid]=cache_geometry(r)
                    pool.append(c)
            # Failed/initial-success captures cannot be chosen as known valid states.
            pool=[c for c in pool if c['case_id'] not in captures or (captures[c['case_id']].get('capture_success') and not captures[c['case_id']].get('initial_success'))]
            selection,method=choose(pool,config['states_per_task'][phase],local_features,[c for c in picked if c['case_id'] in local_features])
            selection = [{**c,'diversity_selection_method':method} for c in selection]
            picked+=selection
            report.append({'task_key':key,'phase':phase,'candidate_count':len(pool),
                           'measured_candidate_count':sum(c['case_id'] in local_features for c in pool),
                           'method':method,'case_ids':[c['case_id'] for c in selection]})
        cases.extend(sorted(picked,key=lambda c:(list(config['states_per_task']).index(c['phase']),c['case_id'])))
        refs.extend(old_refs[c['case_id']] for c in picked if c['initial_state']['kind']=='reference_demo')
    catalog.update(split_revision=4,states_per_task=config['states_per_task'],status='partitioned_pending_runtime_verification')
    write(HERE/'task_catalog.json',catalog)
    for name,rows in [('initial_states.jsonl',cases),('source_refs.jsonl',refs)]:
        (HERE/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    for phase in config['states_per_task']:
        manifest=read(old/'manifests'/(phase+'.json'))
        manifest.update(split_revision=4,cases=[c for c in cases if c['phase']==phase])
        write(HERE/'manifests'/(phase+'.json'),manifest)
    audit=read(old/'audit.json')
    audit.update(case_counts=dict(Counter(c['phase'] for c in cases)),cached_case_count=len(refs),native_seed_case_count=len(cases)-len(refs))
    prior_training = defaultdict(set)
    for c in old_cases:
        if c['phase'] in ('sft_train','rl_train') and c['initial_state']['kind']=='reference_demo':
            prior_training[c['task_key']].add(c['initial_state']['reference_demo_index'])
    audit['v3_training_pool_protected_from_validation_test']={k:sorted(v) for k,v in prior_training.items()}
    write(HERE/'audit.json',audit)
    write(HERE/'diversity_selection.v4.json',{'split_revision':4,'algorithm':'task geometry normalized farthest-point selection, respecting phase pools and historical exposure',
        'scope':'Cached non-robot qpos/scene identity; captured native object and goal geometry. Unmeasured tasks explicitly flagged; seed magnitude is not a distance.',
        'selections':report})
    # Publish current selection overlay only; archived v3 evidence remains intact.
    current={c['case_id']:captures[c['case_id']] for c in cases if c['case_id'] in captures and captures[c['case_id']].get('seed')==c['seed'] and captures[c['case_id']].get('initial_state_selector')==c['initial_state']}
    runtime.update(cases=current,case_count=len(current),captured_case_count=sum(r.get('capture_success',False) for r in current.values()),split_revision=4)
    runtime['tasks']=[]  # v3 task-wide sample summaries do not describe v4 selection
    runtime['task_status_counts']={}
    runtime['scope']='Revision 4 selected-state captures; geometry-only captures are not reproducibility checks. Original v3 probes are archived.'
    write(HERE/'runtime_checks/index.json',runtime)
    paths=[HERE/p for p in read(old/'checksums.json')]
    write(HERE/'diversity_features/cache_geometry.json',_geometry_cache)
    paths += [HERE/'reduce.py',HERE/'collect_diversity.py',HERE/'probe_initial_states.py',HERE/'diversity_selection.v4.json',HERE/'diversity_features/cache_geometry.json']
    paths += sorted((HERE/'diversity_candidates').glob('*/*/features.json'))
    write(HERE/'checksums.json',{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    print(json.dumps(audit['case_counts']))


if __name__=='__main__':
    build()
