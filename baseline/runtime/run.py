"""RGB + native-action evaluation, with host-owned scoring and durable records."""
from __future__ import annotations
import argparse
import asyncio
import base64
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from adapter.native_success import scalar_success
from agent.backends.planner import PlannerBackendRequest
from agent.runtime.actions import PipelineStatus
from agent.runtime.planner import _parse_backend_payload
from baseline.runtime.provider import BASELINE, build_backend
from baseline.scripts.inference_smoke import redact

SNAPSHOT = BASELINE / 'snapshots/native-actions-v1'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def verify_snapshot():
    lock = json.loads((SNAPSHOT / 'lock.json').read_text())
    for rel, expected in lock['files'].items():
        if hashlib.sha256((BASELINE.parent / rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen input changed: {rel}')
    return hashlib.sha256((SNAPSHOT / 'lock.json').read_bytes()).hexdigest()


def validate_action(parameters, created):
    action = parameters.get('action')
    if isinstance(action, str):
        try:
            action = json.loads(action)
        except ValueError:
            pass
    dim = created['action_dim']
    if not isinstance(action, list) or len(action) != dim:
        raise ValueError(f'action must contain exactly {dim} numbers')
    for i, x in enumerate(action):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError('action must contain finite numbers')
        if not created['action_low'][i] <= x <= created['action_high'][i]:
            raise ValueError(f'action[{i}] is outside native bounds')
    steps = parameters.get('num_steps', 1)
    if type(steps) is not int or not 1 <= steps <= 10:
        raise ValueError('num_steps must be an integer from 1 to 10')
    if set(parameters) - {'action', 'num_steps'}:
        raise ValueError('unknown step_env parameter')
    return action, steps


def project_observation(observation, directory, turn):
    """Never expose privileged objects, reward shaping, goal positions or depth."""
    cameras = observation.get('cameras') or []
    primary = [c for c in cameras if not any(s in c.get('frame_id', '').lower() for s in ('wrist', 'hand'))]
    wrist = [c for c in cameras if any(s in c.get('frame_id', '').lower() for s in ('wrist', 'hand'))]
    # Meta-World's adapter 'view' slot is a zero placeholder; the worker's
    # real MuJoCo RGB frame is named 'render'. Keep native cameras elsewhere.
    primary.sort(key=lambda c: (c.get('frame_id') == 'view', c.get('frame_id') == 'render'))
    selected = primary[:1] + wrist[:1]
    if not primary:
        raise RuntimeError('No primary RGB camera')
    paths, descriptors = [], []
    for i, camera in enumerate(selected):
        data = base64.b64decode(camera['rgb_base64'])
        pixels = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
        if i == 0 and pixels.shape != (512, 512, 3):
            raise RuntimeError(f'Unexpected primary image size: {pixels.shape}')
        if not np.isfinite(pixels).all() or pixels.std() <= 1:
            raise RuntimeError('Invalid or blank RGB image')
        path = directory / f'{turn:03d}-{i}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        paths.append(str(path))
        descriptors.append({k: camera[k] for k in ('frame_id', 'width', 'height', 'intrinsics', 'extrinsics') if k in camera})
    return {'robot': observation.get('robot', {}), 'cameras': descriptors, 'vision_image_paths': paths}


def verify_paired_initial(root, row, projection):
    directory = root / '_initial_states'
    directory.mkdir(parents=True, exist_ok=True)
    name = row['episode_id']
    current = np.array(Image.open(projection['vision_image_paths'][0]))
    with (directory / f'{name}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        reference = directory / f'{name}.png'
        state = directory / f'{name}.json'
        if reference.exists():
            previous = np.array(Image.open(reference))
            mae = float(np.abs(current.astype(float) - previous).mean())
            tolerance = 1.0 if row['metadata']['benchmark'] == 'metaworld' else 0.0
            if mae > tolerance or json.loads(state.read_text())['robot'] != projection['robot']:
                raise RuntimeError(f'Paired initial state mismatch: {name}, pixel MAE={mae}')
        else:
            Image.fromarray(current).save(reference)
            write_json(state, {'robot': projection['robot'], 'seed': row['seed']})
    return hashlib.sha256(current.tobytes()).hexdigest()


async def run_episode(args, row, attempt):
    backend = build_backend(args.model)
    sid = 'baseline-' + uuid.uuid4().hex
    directory = Path(args.root) / args.model / row['episode_id'] / f'attempt-{attempt}'
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + (args.timeout or row['timeout_s'])
    record = {'model': args.model, 'episode': row, 'attempt': attempt, 'session_id': sid,
              'protocol': 'rgb_native_actions_v1', 'snapshot_sha256': verify_snapshot(),
              'generation': backend.config.redacted(), 'status': 'infrastructure_error',
              'success': False, 'native_success_observed': False, 'turns': [], 'total_tokens': 0,
              'tool_calls': 0, 'native_steps_requested': 0}
    handle = None
    try:
        async with streamablehttp_client(args.sim_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                async def call(name, **values):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('episode wallclock budget exhausted')
                    response = await asyncio.wait_for(session.call_tool(name, {**values, 'session_id': sid}), min(300, remaining))
                    if response.isError:
                        raise RuntimeError(f'{name}: {response.content}')
                    data = response.structuredContent
                    if data is None:
                        data = json.loads(next(c.text for c in response.content if c.type == 'text'))
                    if data.get('error'):
                        raise RuntimeError(f'{name}: {data["error"]}')
                    return data
                try:
                    created = await call('create_env', env_id=row['env_id'], seed=row['seed'], task=row['task'], image_width=512, image_height=512, include_objects=False)
                    handle = created['handle']
                    record['controller'] = created
                    if any(len(created.get(k, [])) != created['action_dim'] for k in ('action_low', 'action_high')):
                        raise RuntimeError('Native action bounds missing or incomplete')
                    observation = await call('reset_env', handle=handle, seed=row['seed'], settle=False)
                    projected = project_observation(observation, directory / 'images', 0)
                    record['initial_frame_sha256'] = verify_paired_initial(Path(args.root), row, projected)
                    system = (SNAPSHOT / 'system.txt').read_text() + '\n' + (SNAPSHOT / 'skills/closed_loop.md').read_text()
                    control = json.loads((SNAPSHOT / 'controls.json').read_text())[row['metadata']['benchmark']]
                    history = []
                    feedback = ''
                    max_turns = args.turns or row['max_turns']
                    for turn in range(max_turns):
                        if time.monotonic() >= deadline:
                            record['status'] = 'timeout'; break
                        if record['tool_calls'] >= row['max_tool_calls'] or record['total_tokens'] >= row['max_total_tokens']:
                            record['status'] = 'budget_exhausted'; break
                        context = {'task': row['task'], **projected, 'control': control,
                            'action_dim': created['action_dim'], 'action_low': created['action_low'], 'action_high': created['action_high'],
                            'available_tools': [
                                {'name': 'observe_env', 'parameters': {'type': 'object', 'properties': {}}},
                                {'name': 'step_env', 'parameters': {'type': 'object', 'properties': {'action': {'type': 'array', 'items': {'type': 'number'}, 'minItems': created['action_dim'], 'maxItems': created['action_dim']}, 'num_steps': {'type': 'integer', 'minimum': 1, 'maximum': 10}}, 'required': ['action']}}],
                            'remaining_turns': max_turns-turn, 'feedback': feedback}
                        while len(json.dumps(history, ensure_ascii=False)) > 16000:
                            history = history[2:]
                        backend.config.timeout_s = max(1, min(180, deadline-time.monotonic()))
                        response = await asyncio.to_thread(backend.decide, PlannerBackendRequest(tool_context=context, system_prompt=system, conversation_messages=history))
                        result = {'turn': turn, 'content': response.payload, 'details': response.details,
                                  'exchange': response.rollout_exchange}
                        record['total_tokens'] += int((response.details.get('usage') or {}).get('total_tokens', 0))
                        record['turns'].append(result)
                        if response.status != PipelineStatus.PLANNED:
                            raise RuntimeError('Provider request failed; see recorded response details')
                        if not any(x.get('attached') for x in response.details.get('vision_attachments', [])):
                            raise RuntimeError('RGB attachment was not sent to provider')
                        payload, errors = _parse_backend_payload(response.payload)
                        feedback = ''
                        if response.details.get('finish_reason') != 'stop':
                            errors.append('Incomplete model output')
                        try:
                            if errors:
                                raise ValueError('; '.join(errors))
                            if payload.get('kind') == 'response' and payload.get('name') == 'task_complete':
                                record['status'] = 'model_stopped'; break
                            if payload.get('kind') != 'tool_call':
                                raise ValueError('Only step_env, observe_env or task_complete are allowed')
                            if payload.get('name') == 'step_env':
                                action, steps = validate_action(payload.get('parameters') or {}, created)
                                record['tool_calls'] += 1
                                response_step = await call('step_env', handle=handle, action=action, num_steps=steps)
                                native = (response_step.get('info') or {}).get('native_success')
                                if not isinstance(native, dict) or native.get('source') != 'environment.info.success' or scalar_success(native.get('success')) is None:
                                    raise RuntimeError('Missing native success evidence; refusing reward-based scoring')
                                result['native_success'] = native
                                result['native_reward'] = response_step.get('reward')
                                result['terminated'] = response_step.get('terminated')
                                result['truncated'] = response_step.get('truncated')
                                record['native_success_observed'] = True
                                record['native_steps_requested'] += steps
                                observation = response_step['observation']
                                if native.get('ever_success') is True:
                                    record['success'] = True
                                    record['status'] = 'success'; break
                                if response_step.get('terminated') or response_step.get('truncated'):
                                    record['status'] = 'environment_horizon'; break
                            elif payload.get('name') == 'observe_env':
                                if payload.get('parameters'):
                                    raise ValueError('observe_env takes no parameters')
                                record['tool_calls'] += 1
                                observation = await call('observe_env', handle=handle)
                            else:
                                raise ValueError('Unknown tool')
                        except ValueError as exc:
                            feedback = str(exc)
                            result['validation_error'] = feedback
                        projected = project_observation(observation, directory / 'images', turn+1)
                        history.extend([{'role': 'user', 'content': 'Execute one bounded native action for the task.'},
                                        {'role': 'assistant', 'content': str(response.payload)[:3000]}])
                        history = history[-24:]
                        write_json(directory / 'progress.json', redact(record, [backend.config.api_key]))
                    else:
                        record['status'] = 'turn_limit'
                    record['final_observation'] = project_observation(observation, directory / 'final-images', 0)
                finally:
                    if handle:
                        try:
                            response = await asyncio.wait_for(session.call_tool('close_env', {'handle': handle, 'session_id': sid}), 30)
                            if response.isError:
                                raise RuntimeError('close_env returned an error')
                            record['closed'] = True
                        except Exception as exc:
                            record['cleanup_error'] = str(exc)
                            record['status'] = 'infrastructure_error'
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
        record['status'] = 'timeout' if time.monotonic() >= deadline else 'infrastructure_error'
    record['elapsed_s'] = round(time.monotonic()-started, 3)
    record = redact(record, [backend.config.api_key])
    write_json(directory / 'result.json', record)
    return record


async def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['deepseek', 'qwen38_27b', 'qwen35_9b', 'qwen35_4b'])
    p.add_argument('--root', required=True)
    p.add_argument('--sim-url', default='http://127.0.0.1:8200/mcp')
    p.add_argument('--benchmark', choices=['libero', 'metaworld', 'maniskill3', 'robocasa'])
    p.add_argument('--limit', type=int)
    p.add_argument('--turns', type=int)
    p.add_argument('--timeout', type=int)
    args = p.parse_args()
    snapshot_hash = verify_snapshot()
    rows = json.loads((BASELINE / 'manifests/all_40.json').read_text())['episodes']
    if args.benchmark:
        rows = [r for r in rows if r['metadata']['benchmark'] == args.benchmark]
    if args.limit:
        rows = rows[:args.limit]
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / f'.{args.model}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = root / f'{args.model}-run.json'
        config = {'snapshot_sha256': snapshot_hash, 'model': args.model,
                  'episode_ids': [r['episode_id'] for r in rows], 'turns': args.turns,
                  'timeout': args.timeout, 'protocol': 'rgb_native_actions_v1'}
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise RuntimeError('Resume configuration differs from original run')
        write_json(config_path, config)
        finals = []
        for row in rows:
            final_path = root / args.model / row['episode_id'] / 'final.json'
            if final_path.exists():
                final = json.loads(final_path.read_text())
            else:
                for attempt in (1, 2):
                    attempt_path = final_path.parent / f'attempt-{attempt}' / 'result.json'
                    if attempt_path.exists():
                        final = json.loads(attempt_path.read_text())
                    elif attempt_path.parent.exists():
                        final = {'status': 'infrastructure_error', 'success': False,
                                 'error': 'interrupted attempt', 'episode': row, 'attempt': attempt}
                        write_json(attempt_path, final)
                    else:
                        final = await run_episode(args, row, attempt)
                    if final['status'] != 'infrastructure_error':
                        break
                write_json(final_path, final)
            finals.append(final)
            summary = {'model': args.model, 'protocol': config['protocol'], 'snapshot_sha256': snapshot_hash,
                       'completed': len(finals), 'planned': len(rows),
                       'success_count': sum(x.get('success', False) and x['status'] != 'infrastructure_error' for x in finals),
                       'infrastructure_errors': sum(x['status'] == 'infrastructure_error' for x in finals),
                       'by_benchmark': {}}
            for bench in ('libero', 'metaworld', 'maniskill3', 'robocasa'):
                subset = [x for x in finals if x['episode']['metadata']['benchmark'] == bench]
                n = sum(x.get('success', False) and x['status'] != 'infrastructure_error' for x in subset)
                summary['by_benchmark'][bench] = {'completed': len(subset), 'success_count': n,
                    'success_rate': n/len(subset) if subset else None}
            write_json(root / f'{args.model}-summary.json', summary)
            print(json.dumps({'model': args.model, 'episode': row['episode_id'], 'status': final['status'], 'success': final.get('success', False)}), flush=True)
            if final['status'] == 'infrastructure_error':
                raise RuntimeError('Two infrastructure failures; stopped instead of producing invalid benchmark scores')


if __name__ == '__main__':
    asyncio.run(main())
