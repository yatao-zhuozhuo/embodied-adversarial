"""Single trainable policy, native benchmark tasks, terminal binary rewards."""
from __future__ import annotations

import asyncio
import time
import faulthandler
import signal
faulthandler.register(signal.SIGUSR2, all_threads=True)

from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path
import re

from swift.rewards import ORM, orms
from swift.rollout.multi_turn import multi_turns
from swift.trainers.trainer_factory import TrainerFactory

# Reuse the tested Swift token/logprob hooks and memory patches, not Alice/Bob task logic.
from plugins.embodied_swift_grpo import _EmbodiedScheduler, _request_id, _choice_content
from agent.runtime.two_tool_xml import parse_decision, PROTOCOL, INVALID_POLICY
from agent.runtime.two_tool_input import INPUT_PROTOCOL
from agent.training.grpo_input import InputBuilder, execution_receipt
from agent.training.conditional_turns import persist_input_record
from agent.training.single_model_rollout import NativeTaskEnvironment, audit_initial_state, write_json


@dataclass
class NativeEpisode:
    request_id: str
    env: NativeTaskEnvironment
    config: dict
    input_builder: InputBuilder
    initial_evidence: dict
    max_steps: int
    max_control_steps: int
    visible_inputs: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    control_steps: int = 0
    invalid_streak: int = 0
    observe_streak: int = 0
    pending_input: dict | None = None
    pending_prompt: str | None = None
    success: bool = False


class SingleModelScheduler(_EmbodiedScheduler):
    role = 'task solver'
    two_tool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if os.environ.get('OPENETA_INPUT_PROTOCOL') != INPUT_PROTOCOL:
            raise ValueError('Single-policy GRPO requires unified input v1')
        self.per_turn_tokens = int(os.environ.get('OPENETA_RL_TURN_TOKENS', '768'))
        self.episode_tokens = int(os.environ.get('OPENETA_RL_EPISODE_TOKENS', '46080'))
        if self.per_turn_tokens < 16 or self.episode_tokens < self.per_turn_tokens:
            raise ValueError('Invalid generation budgets')

    def prepare_request_config(self, request_config):
        bounded = deepcopy(request_config)
        bounded.max_tokens = self.per_turn_tokens
        bounded.structured_outputs_regex = None
        bounded.return_details = True
        bounded.logprobs = True
        return bounded

    def _build(self, state, request, turn, new_observation):
        if self.input_encoder is None:
            raise ValueError('Exact multimodal tokenizer required')
        builder = state.input_builder
        obs_id = f'observation-{turn}' if new_observation else builder.current['observation_id']
        builder.observe(state.env.observation, observation_id=obs_id, turn=turn,
                        new_observation=new_observation)
        built = builder.build(remaining_turns=state.max_steps-turn,
            remaining_control_steps=state.max_control_steps-state.control_steps,
            generation_budget=self.per_turn_tokens,
            context_limit=int(os.environ.get('VLLM_MAX_MODEL_LEN', '16384')),
            encode=self.input_encoder, enable_thinking=self._thinking_enabled)
        safe_id = re.sub(r'[^A-Za-z0-9_.-]', '-', state.request_id)
        built['audit_path'] = str(self._artifact_root/'inputs'/f'{safe_id}.turn-{turn+1:04d}.json')
        persist_input_record(built)
        state.visible_inputs.append(built)
        self.input_records[state.request_id] = state.visible_inputs
        if turn == 0:
            request.messages = deepcopy(built['request']['messages'])
            request.images = list(built['request']['images'])
            request.chat_template_kwargs = deepcopy(built['request']['chat_template_kwargs'])
        else:
            state.pending_input = built['request']

    async def on_trajectory_start(self, requests):
        try:
            for request in requests:
                rid = _request_id(request)
                await self._close(rid)
                self._finished.discard(rid)
                config = deepcopy((request.data_dict or {})['env_request'])
                env = NativeTaskEnvironment(config)
                # Register immediately so all failures close the environment.
                state = NativeEpisode(rid, env, config,
                    InputBuilder(env.profile, role=self.role, task=env.task,
                                 directory=self._artifact_root/'input_frames'),
                    {}, min(int(config['max_turns']), int(self.max_turns or config['max_turns'])),
                    int(config['max_control_steps']))
                self._episodes[rid] = state
                evidence = {'physics': env.fingerprint(), 'task': env.task, 'profile_hash': env.profile['hash']}
                state.initial_evidence = evidence
                audit_initial_state(self._artifact_root/'initial_states', config['initial_key'], evidence)
                self._build(state, request, 0, True)
        except Exception:
            for rid in list(self._episodes):
                await self._close(rid)
            raise

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        rid = _request_id(infer_request)
        state = self._episodes[rid]
        try:
            return await self._turn(state, infer_request, response_choice, current_turn)
        except Exception as exc:
            self._save(state, {'done': True, 'success': False, 'reward': None,
                              'termination_reason': 'infrastructure_error', 'error': str(exc)})
            for key in list(self._episodes):
                await self._close(key)
            raise  # fail the whole generation batch, never turn transport failures into R=0

    async def _turn(self, state, request, choice, turn):
        raw = _choice_content(choice)
        ids = list(choice.token_ids or [])
        logprobs = self._extract_logprobs_from_choice(choice)
        if not ids or len(ids) != len(logprobs):
            raise ValueError('Missing generated IDs/logprobs')
        record = state.visible_inputs[-1]
        record.update(answer=raw, response_token_ids=ids, response_loss_mask=[1]*len(ids),
                      rollout_logprobs=logprobs, finish_reason=choice.finish_reason,
                      episode_id=state.request_id, turn=turn)
        persist_input_record(record)
        command = None
        validation = None
        error = None
        try:
            validation = parse_decision(raw, contract=state.env.contract,
                remaining_steps=state.max_control_steps-state.control_steps,
                finish_reason=choice.finish_reason)
            decision = validation['decision']
            command = ({'response': 'finish'} if decision['kind'] == 'response' else
                       {'tool': decision['name'], **decision['parameters']})
            if command.get('tool') == 'observe_env' and state.observe_streak >= 3:
                raise ValueError('Too many consecutive observations without movement')
        except ValueError as exc:
            error = str(exc)
        executed = 0
        environment_seconds = 0.
        native_receipt = {}
        terminated = truncated = False
        new_observation = False
        reason = None
        if error:
            state.invalid_streak += 1
            if choice.finish_reason == 'length':
                reason = 'model_output_limit'
        else:
            state.invalid_streak = 0
            if command.get('tool') == 'step_env':
                state.observe_streak = 0
                started = time.perf_counter()
                _, native_receipt, state.success = await asyncio.to_thread(
                    state.env.step, command['action'], command['num_steps'])
                environment_seconds = time.perf_counter() - started
                executed = native_receipt['executed_control_steps']
                state.control_steps += executed
                terminated = bool(native_receipt.get('terminated'))
                truncated = bool(native_receipt.get('truncated'))
                new_observation = True
            elif command.get('tool') == 'observe_env':
                state.observe_streak += 1
                started = time.perf_counter()
                await asyncio.to_thread(state.env.observe)
                environment_seconds = time.perf_counter() - started
                new_observation = True
            else:
                reason = 'model_finish'
        feedback = execution_receipt(decision_id=turn, command=command if not error else None,
            executed_steps=executed, error=error, terminated=terminated, truncated=truncated)
        state.input_builder.record(raw, feedback)
        state.steps.append({'turn': turn, 'completion': raw, 'validation': validation,
            'invalid_reason': error, 'command': command, 'input_execution': feedback,
            'native_receipt': native_receipt, 'official_success': state.success,
            'environment_call_seconds': environment_seconds})
        used_tokens = sum(len(r.get('response_token_ids', [])) for r in state.visible_inputs)
        if state.success: reason = 'official_success'
        elif terminated or truncated: reason = 'environment_ended'
        elif state.invalid_streak >= 3: reason = 'invalid_xml_limit'
        elif state.control_steps >= state.max_control_steps: reason = 'control_step_limit'
        elif turn >= state.max_steps: reason = 'turn_limit'
        elif used_tokens + self.per_turn_tokens > self.episode_tokens: reason = 'episode_token_limit'
        if reason:
            # Save final evidence even when it will never become another model input.
            state.input_builder.observe(state.env.observation, observation_id=f'final-{turn}', turn=turn)
            info = {'role': self.role, 'done': True, 'success': state.success,
                    'reward': float(state.success), 'request_id': state.request_id,
                    'task_id': state.config['task_id'], 'bench': state.config['bench'],
                    'termination_reason': reason, 'num_turns': len(state.steps),
                    'control_steps': state.control_steps, 'generated_tokens': used_tokens}
            self._save(state, info)
            await self._close(state.request_id, mark_finished=True)
            return {'done': True, 'rollout_infos': info}
        started = time.perf_counter()
        self._build(state, request, turn, new_observation)
        state.steps[-1]['input_build_seconds'] = time.perf_counter() - started
        return {'done': False, 'rollout_infos': {'role': self.role, 'done': False}}

    def _save(self, state, info):
        safe_id = re.sub(r'[^A-Za-z0-9_.-]', '-', state.request_id)
        write_json(self._artifact_root/'episodes'/f'{safe_id}.json', {
            'schema_version': 'openeta.single_model_rl.v1', 'input_protocol': INPUT_PROTOCOL,
            'model_protocol': PROTOCOL, 'invalid_policy': INVALID_POLICY,
            'config': state.config, 'initial_evidence': state.initial_evidence,
            'input_profile': state.env.profile, 'steps': state.steps,
            'visible_inputs': state.visible_inputs, 'final_observation': state.input_builder.current,
            'final_info': info})


class NativeSuccessReward(ORM):
    def __call__(self, completions, **kwargs):
        infos = kwargs.get('rollout_infos')
        if not isinstance(infos, list) or len(infos) != len(completions):
            raise ValueError('Missing per-episode trusted rewards')
        result = []
        for info in infos:
            if not isinstance(info, dict) or not info.get('done') or info.get('role') != 'task solver':
                raise ValueError('Incomplete native-task rollout')
            reward = info.get('reward')
            if reward not in (0., 1.) or reward != float(info['success']):
                raise ValueError('Native-task reward must be binary official success')
            result.append(float(reward))
        return result


multi_turns['single_model_native_scheduler'] = SingleModelScheduler
orms['single_model_success'] = NativeSuccessReward
TrainerFactory.TRAINER_MAPPING['grpo'] = 'agent.training.input_grpo_trainer.InputGRPOTrainer'
