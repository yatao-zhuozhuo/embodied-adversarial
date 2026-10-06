"""Native-task MCP rollouts shared by single-policy GRPO and its preflight."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

import numpy as np
from PIL import Image

from agent.runtime import two_tool_input as shared
from agent.runtime.two_tool_xml import action_contract
from agent.tools.sim_mcp import SseSimulatorMcpTransport


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{uuid4().hex}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def official_success(payload, bench):
    """Only the native checker, never dense reward, termination or model prose."""
    info = payload.get('info')
    if not isinstance(info, dict) or 'success' not in info:
        raise ValueError('Missing official success receipt')
    success = info['success']
    if bench == 'maniskill' and isinstance(success, list) and len(success) == 1:
        success = success[0]
    if type(success) not in (bool, int, float) or success not in (0, 1):
        raise ValueError('Invalid official success receipt')
    return bool(success)


def decode_observation(payload):
    observation = deepcopy(payload.get('observation', payload))
    for camera in observation.get('cameras', []):
        if camera.get('rgb_base64'):
            with Image.open(io.BytesIO(base64.b64decode(camera.pop('rgb_base64')))) as image:
                camera['rgb'] = np.asarray(image.convert('RGB')).copy()
    if not observation.get('cameras'):
        raise ValueError('Observation has no cameras')
    return observation


def audit_initial_state(directory, key, evidence):
    """Compare host physics fingerprints across ranks, before accepting a group."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (hashlib.sha256(key.encode()).hexdigest() + '.json')
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            expected = json.loads(path.read_text())
            if expected != evidence:
                raise ValueError(f'Initial-state mismatch for {key}; refusing unequal GRPO conditions')
        else:
            write_json(path, evidence)
    return str(path)


class NativeTaskEnvironment:
    def __init__(self, config, *, transport=None):
        self.config = deepcopy(config)
        self.base = os.environ.get('OPENETA_EMBODIED_MCP_URL', '').removesuffix('/sse').rstrip('/')
        self.transport = transport or SseSimulatorMcpTransport(self.base + '/sse')
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='single-policy-mcp')
        self.session = 'single-rl-' + uuid4().hex
        self.handle = None
        try:
            self.created = self.call('create_env', env_id=config['env_id'], task=config['task'],
                seed=config['seed'], render_mode='rgb_array', image_width=256, image_height=256,
                include_objects=False)
            self.handle = self.created['handle']
            self.session = self.created.get('session_id', self.session)
            self.contract = action_contract(self.created)
            if self.contract['action_dim'] != config['expected_action_dim']:
                raise ValueError('Live action dimension differs from manifest')
            self.profile = shared.make_profile(config['bench'], config['task_id'], self.created, self.contract)
            self.reset_receipt = self.call('reset_env', seed=config['seed'])
            self.task = shared.task_description(config['task_id'], self.reset_receipt.get('task') or config['task'])
            self.observation = decode_observation(self.call('observe_env'))
        except Exception:
            self.close()
            raise

    def call(self, name, **arguments):
        arguments = {'session_id': self.session, **arguments}
        if self.handle:
            arguments['handle'] = self.handle
        result = self.pool.submit(self.transport.call_tool, name, arguments, timeout_s=300).result()
        if not isinstance(result, dict) or result.get('error'):
            raise RuntimeError(f'{name}: {result.get("error") if isinstance(result, dict) else "invalid receipt"}')
        return result

    def fingerprint(self):
        token = os.environ.get('OPENETA_HOST_STATE_TOKEN', '')
        if not token:
            raise ValueError('Host state authentication required')
        req = Request(f'{self.base}/host-state/{self.session}/{self.handle}',
            json.dumps({'operation': 'fingerprint'}).encode(),
            {'Content-Type': 'application/json', 'X-OpenETA-Host-Token': token}, method='POST')
        with urlopen(req, timeout=300) as response:
            result = json.load(response)
        if result.get('error') or not result.get('state_sha256'):
            raise ValueError('Missing native state fingerprint')
        return result

    def observe(self):
        self.observation = decode_observation(self.call('observe_env'))
        return self.observation

    def step(self, action, num_steps):
        payload = self.call('step_env', action=action, num_steps=num_steps, strict=True)
        executed = payload.get('executed_control_steps')
        if type(executed) is not int or not 0 <= executed <= num_steps:
            raise ValueError('Missing or invalid executed control-step receipt')
        success = official_success(payload, self.config['bench'])
        self.observation = decode_observation(payload)
        # Keep scoring evidence apart from model-visible feedback, without duplicating RGB blobs.
        receipt = {k: v for k, v in payload.items() if k not in ('observation', 'cameras')}
        return self.observation, receipt, success

    def close(self):
        try:
            if self.handle:
                self.call('close_env')
        finally:
            self.handle = None
            self.pool.shutdown(wait=True)
