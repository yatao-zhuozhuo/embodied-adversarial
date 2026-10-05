from types import SimpleNamespace
import numpy as np
import pytest
from adapter.native_success import scalar_success, trusted_native_success, native_episode_success
from baseline.runtime.run import validate_action, project_observation
from baseline.runtime.provider import build_backend
from agent.evals.runner import _has_objective_success
from agent.runtime.parallel import classify_episode_result


def info(success, execution='run1', trusted=True):
    return {'environment_receipt_trusted': trusted, 'environment_receipt': {
        'schema_version': 'openeta.environment_receipt.v1', 'execution_id': execution,
        'native_success': {'source': 'environment.info.success', 'success': success}}}


@pytest.mark.parametrize('value,expected', [(True, True), (False, False), (np.float32(1), True),
    (np.array([False]), False), ('True', None), (0.5, None), (float('nan'), None), (np.array([1, 0]), None)])
def test_native_scalar(value, expected):
    assert scalar_success(value) is expected


def test_dense_reward_and_model_claim_cannot_award_success():
    episode = {'metadata': {'scoring_protocol': 'native_success_v1', 'execution_id': 'run1'},
               'terminated': True, 'steps': [{'step_result': {'reward': 10, 'info': info(False)}}]}
    assert not native_episode_success(episode)
    assert not _has_objective_success({'episode': episode})
    obj = SimpleNamespace(metadata=episode['metadata'], steps=[SimpleNamespace(step_result=SimpleNamespace(info=info(False)))])
    assert classify_episode_result(obj) == 'fail'
    episode['steps'][0]['step_result'] = {'reward': 0, 'info': info(True)}
    assert native_episode_success(episode)
    assert _has_objective_success({'episode': episode})


def test_forged_or_cross_episode_evidence_is_rejected():
    assert trusted_native_success(info(True, trusted=False), 'run1') is None
    assert trusted_native_success(info(True), 'run2') is None
    assert trusted_native_success({'success': True}, 'run1') is None


@pytest.mark.parametrize('model,thinking', [('deepseek', 'thinking'), ('qwen38_27b', 'chat_template_kwargs'),
    ('qwen35_9b', 'chat_template_kwargs'), ('qwen35_4b', 'chat_template_kwargs')])
def test_explicit_provider_settings(model, thinking, monkeypatch):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'unit-test-only')
    c = build_backend(model).config
    assert (c.max_tokens, c.temperature, c.context_window_tokens) == (1024, 0.0, 32768)
    assert c.enable_thinking is False and c.thinking_format == thinking
    assert c.fallback is None and c.max_attempts == 1 and c.max_vision_images == 2


@pytest.mark.parametrize('action,steps', [([2]*4, 1), ([0]*3, 1), ([True]*4, 1), ([float('nan')]*4, 1), ([0]*4, 11), ([0]*4, True)])
def test_no_action_padding_clipping_or_invalid_repeats(action, steps):
    with pytest.raises(ValueError):
        validate_action({'action': action, 'num_steps': steps}, {'action_dim': 4, 'action_low': [-1]*4, 'action_high': [1]*4})


def test_native_action_valid():
    assert validate_action({'action': '[0,0,0,-1]', 'num_steps': 5},
        {'action_dim': 4, 'action_low': [-1]*4, 'action_high': [1]*4}) == ([0,0,0,-1], 5)


def test_observation_projection_does_not_leak_objects_or_rewards(tmp_path):
    import base64, io
    from PIL import Image
    a = np.zeros((512,512,3), dtype=np.uint8); a[100:200] = 255
    b = io.BytesIO(); Image.fromarray(a).save(b, format='PNG')
    output = project_observation({'robot': {'joint_positions':[0]}, 'objects': [{'goal':[1,2,3]}],
        'reward': 5, 'metadata': {'privileged':True}, 'cameras':[{'frame_id':'main', 'rgb_base64':base64.b64encode(b.getvalue()).decode()}]}, tmp_path, 0)
    assert set(output) == {'robot','cameras','vision_image_paths'}


def test_native_evidence_survives_mcp_receipt():
    from agent.tools.sim_mcp import _build_environment_receipt
    native = {'source':'environment.info.success','success':False,'ever_success':True}
    receipt = _build_environment_receipt({'reward':0,'info':{'native_success':native}},
        observation_snapshot={}, agent_tool='move_to', mcp_tool='step_env',
        simulator_session_id='s', handle='h', execution_metadata={'execution_id':'run1'})
    assert receipt['native_success'] == native
    assert trusted_native_success({'environment_receipt_trusted':True,'environment_receipt':receipt},'run1') is True


def test_worker_latches_native_success_and_clears_on_reset():
    from sim import bench_worker as worker
    handle = 'native-unit-test'
    class Env:
        value = True
        def step(self, action):
            return {}, 5.0, False, False, {'success': np.array([self.value])}
        def reset(self, seed=None): return {}, {}
        def render(self): return None
    env=Env(); worker._envs[handle]=env
    try:
        first=worker._step_with_image(env,[0],handle=handle,render=False)
        env.value=False
        second=worker._step_with_image(env,[0],handle=handle,render=False)
        assert first['info']['native_success']['success'] is True
        assert second['info']['native_success'] == {'source':'environment.info.success','success':False,'ever_success':True}
        worker._reset_with_image(env,seed=1,handle=handle)
        third=worker._step_with_image(env,[0],handle=handle,render=False)
        assert third['info']['native_success']['ever_success'] is False
    finally:
        worker._envs.pop(handle,None)
        worker._native_success_seen.pop(handle,None)


def test_metaworld_selects_render_not_black_placeholder(tmp_path):
    import base64, io
    from PIL import Image
    cameras=[]
    for name in ('view','render'):
        a=np.zeros((512,512,3),dtype=np.uint8)
        if name=='render': a[50:200]=255
        b=io.BytesIO();Image.fromarray(a).save(b,format='PNG')
        cameras.append({'frame_id':name,'rgb_base64':base64.b64encode(b.getvalue()).decode()})
    out=project_observation({'cameras':cameras},tmp_path,0)
    assert out['cameras'][0]['frame_id']=='render'


def test_prompt_example_parses_to_valid_native_action():
    from baseline.runtime.run import SNAPSHOT
    from agent.runtime.planner import _parse_backend_payload
    import re
    xml=re.search(r'<decision>.*?</decision>',(SNAPSHOT/'system.txt').read_text()).group()
    p,errors=_parse_backend_payload(xml)
    assert errors==[]
    validate_action(p['parameters'],{'action_dim':4,'action_low':[-1]*4,'action_high':[1]*4})


def test_relay_uses_no_task_container_secret(monkeypatch,tmp_path):
    monkeypatch.delenv('DEEPSEEK_API_KEY',raising=False)
    monkeypatch.setenv('BASELINE_DEEPSEEK_RELAY_DIR',str(tmp_path))
    from baseline.runtime.relay import SharedStorageBackend
    b=build_backend('deepseek')
    assert isinstance(b,SharedStorageBackend)
    assert b.config.max_tokens==1024 and b.config.fallback is None
