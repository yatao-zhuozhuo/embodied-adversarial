"""Shared, provider-independent two-tool input contract (baseline adapter first).

Full host observations and receipts never enter prompts. Images are selected by
an explicit profile, not by list order. Renderers encode, never select evidence.
"""
from __future__ import annotations
import base64
import copy
import hashlib
import io
import json
import math
from pathlib import Path
from PIL import Image

INPUT_PROTOCOL = 'openeta.two_tool_input.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def project_observation(raw, directory, turn, profile):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {c['frame_id']: c for c in raw.get('cameras', [])}
    cameras = []
    for spec in profile['cameras']:
        name = spec['id']
        if name not in by_id:
            raise ValueError(f'Observation missing required profile camera: {name}')
        camera = by_id[name]
        data = base64.b64decode(camera['rgb_base64'], validate=True)
        with Image.open(io.BytesIO(data)) as source:
            pixels = source.convert('RGB')
            if list(pixels.size) != spec['size']:
                raise ValueError(f'Camera {name}: expected native {spec["size"]}, got {pixels.size}')
            pixel_hash = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
            path = directory / f'{turn:03d}-{name}.png'
            pixels.save(path)
        cameras.append({'frame_id': name, 'camera_role': spec['role'], 'image_path': str(path),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'pixel_sha256': pixel_hash,
                        'width': pixels.width, 'height': pixels.height,
                        'intrinsics': copy.deepcopy(camera.get('intrinsics')) if profile.get('benchmark') != 'maniskill3' else None,
                        'extrinsics': copy.deepcopy(camera.get('extrinsics')) if profile.get('benchmark') != 'maniskill3' else None})
    # Explicit robot whitelist: never carry object poses, reward, handles or task truth.
    robot = raw.get('robot') or {}
    visible = {k: copy.deepcopy(robot.get(k)) for k in ('joint_positions', 'joint_velocities')}
    ee = robot.get('end_effector_pose') or {}
    visible['end_effector_pose'] = {k: copy.deepcopy(ee.get(k)) for k in ('xyz', 'quat_xyzw')}
    grip = robot.get('gripper_state') or {}
    visible['gripper_state'] = {k: copy.deepcopy(grip.get(k)) for k in ('openness', 'open')}
    base = robot.get('base_pose') or {}
    visible['base_pose'] = {k: copy.deepcopy(base.get(k)) for k in ('xyz', 'quat_xyzw')}
    return {'observation_id': f'obs-{turn}', 'turn': turn, 'robot': visible, 'cameras': cameras,
            'vision_image_paths': [c['image_path'] for c in cameras]}


def robot_delta(before, after, contract):
    result = {'tcp_displacement_m': None, 'tcp_distance_m': None, 'gripper_openness_change': None,
              'frame': contract['tcp']['frame'], 'reason': None}
    if before is None:
        result['reason'] = 'initial observation'
        return result
    a = before['robot']['end_effector_pose'].get('xyz')
    b = after['robot']['end_effector_pose'].get('xyz')
    if contract['tcp']['frame'] == 'unknown' or not a or not b:
        result['reason'] = 'missing TCP measurement or unverified reference frame'
    else:
        delta = [float(y)-float(x) for x, y in zip(a, b)]
        if len(delta) == 3 and all(math.isfinite(x) for x in delta):
            result.update(tcp_displacement_m=delta, tcp_distance_m=math.sqrt(sum(x*x for x in delta)))
    a = before['robot']['gripper_state'].get('openness')
    b = after['robot']['gripper_state'].get('openness')
    if type(a) in (int, float) and type(b) in (int, float):
        result['gripper_openness_change'] = b-a
    return result


def execution_record(turn, validation, *, receipt=None, error=None, sent=False):
    """Unknown transport outcome is distinct from a rejected, unexecuted action."""
    decision = validation.get('decision') or {}
    params = decision.get('parameters') or {}
    if error:
        state, actual = ('error', None) if sent else ('rejected', 0)
    elif not validation.get('valid'):
        state, actual = 'rejected', 0
        error = '; '.join(validation.get('errors', []))
    else:
        state = 'executed'
        actual = (receipt or {}).get('steps_executed', 0 if decision.get('name') != 'step_env' else None)
    return {'decision_id': f'decision-{turn}', 'tool': decision.get('name'),
            'parameters': copy.deepcopy(params) if validation.get('valid') else None,
            'requested_steps': params.get('num_steps') if decision.get('name') == 'step_env' else 0,
            'steps_executed': actual, 'status': state, 'result_known': state != 'error',
            'error': {'message': error} if error else None,
            'terminated': (receipt or {}).get('terminated', False),
            'truncated': (receipt or {}).get('truncated', False)}


def answer_only(raw):
    # Preserve invalid answer text, but never replay hidden thinking.
    text = str(raw)
    if text.lstrip().startswith('<think>'):
        return text.split('</think>', 1)[1].strip() if '</think>' in text else '[Unclosed thinking; no final answer]'
    return text


def build_input(*, task, role, system, profile, contract, current, previous, history, budget):
    if not task.strip() or len(contract['slots']) != contract['action_dim']:
        raise ValueError('Missing task or inconsistent control contract')
    selected = copy.deepcopy(history[-4:])
    latest = selected[-1]['execution'] if selected else None
    evidence, by_hash = [], {}
    sources = []
    if previous:
        sources.extend((previous, c, 'historical') for c in previous['cameras'] if c['camera_role'] == 'primary')
    sources.extend((current, c, 'current') for c in current['cameras'])
    for obs, camera, timing in sources:
        ref = {'camera_id': camera['frame_id'], 'camera_role': camera['camera_role'],
               'observation_id': obs['observation_id'], 'turn': obs['turn'], 'timing': timing,
               'after_decision_id': f'decision-{obs["turn"]-1}' if obs['turn'] else None}
        key = camera['pixel_sha256']
        if key in by_hash:
            by_hash[key]['references'].append(ref)
        else:
            item = {'image_id': f'image-{len(evidence)}', 'path': camera['image_path'],
                    'pixel_sha256': key, 'sha256': camera['sha256'],
                    'width': camera['width'], 'height': camera['height'], 'references': [ref]}
            evidence.append(item)
            by_hash[key] = item
    if len(evidence) > profile['image_limit']:
        raise ValueError('Profile image limit exceeded')
    payload = {'input_protocol': INPUT_PROTOCOL, 'turn': current['turn'], 'observation_id': current['observation_id'],
               'budget': budget, 'last_action': {k:latest[k] for k in ('decision_id','tool','parameters','requested_steps','steps_executed') if k in latest} if latest else None,
               'execution': {k:latest[k] for k in ('status','result_known','error','terminated','truncated') if k in latest} if latest else {'status': 'initial', 'result_known': True},
               'robot': current['robot'], 'robot_delta': robot_delta(previous, current, contract),
               'observation': {'new_observation': latest is None or latest['status'] == 'executed',
                               'cameras': [{k:v for k,v in c.items() if k in ('frame_id','camera_role','width','height','intrinsics','extrinsics')} for c in current['cameras']]},
               'terminated': bool(latest and latest['terminated']), 'truncated': bool(latest and latest['truncated'])}
    return {'input_protocol': INPUT_PROTOCOL, 'profile_id': profile['id'], 'profile_hash': digest(profile),
            'control_contract_version': contract['version'], 'control_contract_hash': digest(contract),
            'adapter_version': 'openeta.two_tool_chat.v1', 'system': system + '\n' + json.dumps({'task': task, 'role':role, 'control_contract':contract}, ensure_ascii=False),
            'current': payload, 'history': selected, 'images': evidence, 'trimming': [],
            'history_decision_ids': [h['execution']['decision_id'] for h in selected]}


def render_messages(package):
    """One deterministic renderer; latest receipt occurs only in current user."""
    messages = [{'role': 'system', 'content': package['system']}]
    for i, group in enumerate(package['history']):
        user = {'decision_id':group['execution']['decision_id'],
                'note':'Historical decision; its original images are not attached in this message.'}
        if i:
            prior = package['history'][i-1]
            user.update(previous_execution=prior['execution'], previous_robot_delta=prior.get('robot_delta'))
        messages.append({'role':'user', 'content':json.dumps(user, ensure_ascii=False)})
        messages.append({'role':'assistant', 'content':answer_only(group['answer'])})
    current = copy.deepcopy(package['current'])
    current['observation']['image_evidence'] = [{k:v for k,v in e.items() if k != 'path'} for e in package['images']]
    parts = [{'type':'text', 'text':json.dumps(current, ensure_ascii=False)}]
    for evidence in package['images']:
        label = json.dumps({'image_id': evidence['image_id'], 'references': evidence['references'],
                            'identical_image_at_multiple_times': len(evidence['references']) > 1}, ensure_ascii=False)
        parts.append({'type':'text', 'text':label})
        data = Path(evidence['path']).read_bytes()
        if hashlib.sha256(data).hexdigest() != evidence['sha256']:
            raise ValueError('Image changed after input snapshot')
        parts.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(data).decode()}})
    messages.append({'role':'user','content':parts})
    return messages


def fit_budget(package, count_tokens, context_tokens, generation_tokens):
    package = copy.deepcopy(package)
    while True:
        count = count_tokens(render_messages(package))
        if type(count) is not int or count < 0:
            raise ValueError('Provider did not return exact prompt token count')
        if count + generation_tokens <= context_tokens:
            package['prompt_tokens'] = count
            package['generation_token_reserve'] = generation_tokens
            return package
        if package['history']:
            old = package['history'].pop(0)
            package['trimming'].append({'reason':'context_budget','decision_id':old['execution']['decision_id']})
            package['history_decision_ids'] = [h['execution']['decision_id'] for h in package['history']]
        else:
            candidate = next((e for e in package['images'] if all(r['timing']=='historical' for r in e['references'])), None)
            if candidate is None:
                raise ValueError('Minimum required input exceeds context budget')
            package['images'].remove(candidate)
            package['trimming'].append({'reason':'context_budget','image_id':candidate['image_id']})


def control_contract(created, profile):
    """Use the actual bounds and dimension; unknown semantics stay explicit."""
    dim = created['action_dim']
    if dim != profile['action_dim'] or len(created['action_low']) != dim or len(created['action_high']) != dim:
        raise ValueError('Live action space differs from frozen profile')
    b = profile['benchmark']
    caps = created.get('input_capabilities') or {}
    if 'expected_capabilities' in profile and caps != profile['expected_capabilities']:
        raise ValueError('Live controller changed from versioned profile')
    if caps.get('error') or caps.get('control_mode') != profile['control_mode']:
        raise ValueError(f'Unverified live controller capability: {caps}')
    if b == 'maniskill3' and bool(caps.get('has_gripper')) != (dim == 7):
        raise ValueError('Live gripper capability differs from action profile')
    slots = ['dx', 'dy', 'dz']
    if b != 'metaworld':
        slots += ['axis_angle_x', 'axis_angle_y', 'axis_angle_z']
    if dim != 6:
        slots += ['gripper']
    if b == 'robocasa':
        slots += ['base_x_velocity', 'base_y_velocity', 'base_yaw_velocity', 'torso_position', 'goal_update_mode']
    if len(slots) != dim:
        raise ValueError('No audited slot layout for action dimension')
    frames = {'metaworld':'world', 'libero':'world', 'maniskill3':'robot_root', 'robocasa':'robot_base'}
    scales = {'metaworld':0.01, 'libero':0.05, 'maniskill3':0.1, 'robocasa':0.05}
    contract = {'version':'openeta.two_tool_control.v1','controller':profile['control_mode'],
                'control_mode':profile['control_mode'], 'action_dim':dim,
                'action_low':created['action_low'], 'action_high':created['action_high'],
                'slots':[{'index':i,'semantic':name,'normalized':True} for i,name in enumerate(slots)],
                'translation':{'frame':frames[b], 'scale_m':caps.get('position_scale_m', scales[b]), 'semantics':'command delta, not guaranteed measured displacement'},
                'rotation': {'representation':'axis_angle_delta' if b!='metaworld' else 'not_controlled',
                             'frame':frames[b], 'scale_rad':0.1 if b=='maniskill3' else 0.5 if b!='metaworld' else None},
                'gripper':{'supported':dim!=6,'open':1 if b=='maniskill3' else -1,'close':-1 if b=='maniskill3' else 1,
                           'semantics':'command; measured state is reported separately'} if dim!=6 else {'supported':False},
                'tcp':{'frame':'world','position_unit':'meter','quaternion_order':'xyzw'},
                'num_steps':{'min':1,'max':10,'meaning':'repeat environment control steps; actual count may be smaller on termination'},
                'observe_env':'Only observes. Does not move cameras, change viewpoints or increase resolution.',
                'source_versions':profile['source_versions']}
    if b == 'maniskill3':
        contract['translation']['frame'] = caps['translation_frame']
        contract['translation']['scale_m'] = caps['position_upper']
        contract['rotation']['frame'] = caps['rotation_frame']
    if b == 'robocasa':
        contract['verified_parts'] = caps['parts']
        contract['verified_action_slices'] = caps['action_slices']
        contract['mobile']={'mode_positive':'update arm goal from desired goal (base tracking)',
                            'mode_nonpositive':'update arm goal from achieved pose',
                            'unused_dimensions':'zero', 'base_velocity_units':'unknown', 'torso_scale':'unknown',
                            'base_reference_frame':'unknown'}
    if b == 'libero':
        live = created.get('control_spec', {})
        if live.get('controller',{}).get('configured_name') != 'OSC_POSE':
            raise ValueError('LIBERO controller differs from audited OSC_POSE profile')
    return contract
