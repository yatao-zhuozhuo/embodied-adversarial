"""Host-only reset fingerprints; never included in model observations."""
import hashlib
import json
import numpy as np


def _plain(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    raise TypeError(f'Unsupported state component: {type(value).__name__}')


def native_fingerprint(env):
    backend = env._backend
    inner = env._unwrap()
    if backend == 'maniskill':
        state = {'state_dict': inner.get_state_dict(), 'elapsed_steps': inner._elapsed_steps,
                 'evaluation': inner.evaluate()}
        scope = 'ManiSkill state_dict, elapsed steps and task evaluation'
    else:
        # Direct wrappers for robosuite/LIBERO retain their simulator under _env/env.
        obj = inner
        for _ in range(8):
            if hasattr(obj, 'sim') or (hasattr(obj, 'model') and hasattr(obj, 'data')):
                break
            nxt = getattr(obj, '_env', None)
            if nxt is None: nxt = getattr(obj, 'env', None)
            if nxt is None: raise ValueError(f'No native simulator for {backend}')
            obj = nxt
        sim = getattr(obj, 'sim', obj)
        model, data = sim.model, sim.data
        state = {'data': {}, 'model': {}}
        for name in ('qpos', 'qvel', 'act', 'ctrl', 'mocap_pos', 'mocap_quat', 'qacc_warmstart', 'time'):
            if hasattr(data, name): state['data'][name] = getattr(data, name)
        for name in ('body_pos', 'body_quat', 'site_pos', 'site_quat', 'geom_size', 'geom_pos', 'geom_quat'):
            if hasattr(model, name): state['model'][name] = getattr(model, name)
        if hasattr(obj, '_target_pos'): state['target_pos'] = obj._target_pos
        scope = 'MuJoCo dynamics, actuator/mocap state, body/site/geometry configuration and target'
    encoded = json.dumps(_plain(state), sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
    return {'state_sha256': hashlib.sha256(encoded).hexdigest(), 'backend': backend,
            'schema': 'openeta.native_reset_fingerprint.v1', 'scope': scope}
