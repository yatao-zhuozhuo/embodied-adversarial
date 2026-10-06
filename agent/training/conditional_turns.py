"""Swift 4.4.2 per-decision conditioning for bounded-history GRPO.

Roll out and finalize rewards at episode granularity, then encode each decision
under its immutable sampled context. Swift's request-aware advantage path
uniquifies request_id before group normalization (rewards are not summed).
"""
from copy import deepcopy, copy
import os
from pathlib import Path
import json


def enabled():
    return os.environ.get('OPENETA_INPUT_PROTOCOL') == 'openeta.two_tool_input.v1'


def make_input_encoder(template):
    # vLLM-mode Swift emits ONE image placeholder; vLLM later expands it.
    # Transformers mode uses the same visual processor and performs the full
    # expansion here. Compare these IDs to vLLM's actual returned prompt IDs.
    encoder = copy(template)
    encoder.set_mode('transformers')
    return lambda data: encoder.encode(data)['input_ids']


def persist_input_record(record):
    path = record.get('audit_path')
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,ensure_ascii=False)+'\n')
        temporary.replace(target)


def split_turn_samples(samples, records):
    result = []
    for sample in samples:
        turns = records.pop(sample.request_id)
        if not turns:
            raise ValueError('episode has no audited decisions')
        for turn in turns:
            if not turn.get('prompt_verified'):
                raise ValueError('sampling prompt tokens were not verified')
            s = deepcopy(sample)
            s.messages = deepcopy(turn['request']['messages']) + [{'role':'assistant','content':turn['answer']}]
            s.images = list(turn['request']['images'])
            s.response_token_ids = [turn['response_token_ids']]
            s.response_loss_mask = [turn['response_loss_mask']]
            s.rollout_logprobs = [turn['rollout_logprobs']]
            s.finish_reason = turn['finish_reason']
            s.extra['chat_template_kwargs'] = deepcopy(turn['request']['chat_template_kwargs'])
            s.extra['input_audit'] = {k:deepcopy(turn[k]) for k in ('prompt_token_ids','turn','episode_id')}
            result.append(s)
    return result


class ConditionalTurnsMixin:
    def _infer_single_or_multi_turn(self, samples, request_config, is_global_inputs=False):
        if not enabled() or not self.multi_turn_scheduler or self.enable_server_multi_turn:
            return super()._infer_single_or_multi_turn(samples,request_config,is_global_inputs)
        from swift.rollout import invoke_async_hook
        requests = self.samples2requests(samples)
        invoke_async_hook(self.multi_turn_scheduler.on_trajectory_start(requests))
        # Swift 4.4.2 copies only messages from the initial hook. The shared
        # builder also replaces the dataset's images with live observations.
        for request,sample in zip(requests,samples):
            sample.messages = deepcopy(request.messages)
            sample.images = list(request.images)
            sample.extra['chat_template_kwargs'] = deepcopy(request.chat_template_kwargs)
        outputs = self._rollout(samples,request_config,is_global_inputs)
        return self._colocate_multi_turn_infer(samples,outputs,request_config,requests)

    def _generate_completions(self, samples):
        if not enabled():
            return super()._generate_completions(samples)
        if self.args.vllm_mode != 'colocate' or self.args.steps_per_generation != 1:
            raise ValueError('input v1 requires colocate and steps_per_generation=1')
        if 'last_round' not in self.template._loss_scale.split('+'):
            raise ValueError('input v1 requires --loss_scale last_round')
        encode = make_input_encoder(self.engine.template)
        for scheduler in (self.multi_turn_scheduler, getattr(self,'bob_scheduler',None)):
            if scheduler is not None:
                scheduler.input_encoder = encode
        self.dynamic_num_samples = False
        episodes = super()._generate_completions(samples)
        if getattr(self, "bob_scheduler", None) is not None:
            self.bob_scheduler.input_records.clear()
        turns = split_turn_samples(episodes, self.multi_turn_scheduler.input_records)
        self.dynamic_num_samples = True
        return turns

    def _verify_prompts(self, requests, outputs):
        if not enabled():
            return outputs
        schedulers = [self.multi_turn_scheduler, getattr(self,'bob_scheduler',None)]
        for req,out in zip(requests,outputs):
            rid = str(req.uuid)
            for scheduler in schedulers:
                if scheduler is not None and rid in scheduler._episodes:
                    record = scheduler._episodes[rid].visible_inputs[-1]
                    actual = out.response.prompt_token_ids
                    if actual is None or list(actual) != record['prompt_token_ids']:
                        record['actual_prompt_token_ids'] = list(actual) if actual is not None else None
                        record['prompt_verified'] = False
                        persist_input_record(record)
                        raise ValueError('actual generation prompt differs from input snapshot')
                    record['prompt_verified'] = True
                    record['actual_prompt_token_ids'] = list(actual)
                    record['provider_image_sizes'] = out.response.images_size
                    persist_input_record(record)
                    break
            else:
                raise ValueError('no input snapshot for generated request')
        return outputs

    def _engine_infer(self, infer_requests, request_config=None, **kwargs):
        if enabled():
            request_config = self.multi_turn_scheduler.prepare_request_config(
                request_config if request_config is not None else self._get_request_config())
        return self._verify_prompts(infer_requests, super()._engine_infer(infer_requests,request_config,**kwargs))

    def _prepare_batch_inputs(self, samples):
        if enabled():
            from swift.rlhf_trainers.utils import encode_sample
            with self._template_context(self.template):
                for s in samples:
                    encoded = encode_sample(deepcopy(s),self.template)
                    ids, labels = encoded['input_ids'], encoded['labels']
                    indices = [i for i,v in enumerate(labels) if v != -100]
                    if not indices:
                        raise ValueError('no supervised tokens')
                    prompt = ids[:indices[0]]
                    targets = [ids[i] for i in indices]
                    if prompt != s.extra['input_audit']['prompt_token_ids']:
                        raise ValueError('training condition differs from sampled prompt')
                    if targets != s.response_token_ids[0]:
                        raise ValueError('loss mask does not select exactly the generated tokens')
                    if len(targets) != len(s.rollout_logprobs[0]):
                        raise ValueError('rollout logprob count mismatch')
            path=Path(self.args.output_dir)/'conditional_input_validation.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps({'step':self.state.global_step,'rank':self.accelerator.process_index,
                    'turns':len(samples),'prompt_and_loss_verified':True})+'\n')
        return super()._prepare_batch_inputs(samples)

    def _postprocess_batch(self, samples, batch_encoded_inputs):
        if not enabled():
            return super()._postprocess_batch(samples,batch_encoded_inputs)
        import torch
        from accelerate.utils import gather_object
        from swift.rl_core.advantage import expand_advantage_to_per_token
        # Swift marks plain GRPO as teacher-capable (dynamic self distillation)
        # even without a teacher or teacher prompts. Reject actual teacher use.
        if self._has_teacher_explicit() or self.template.sequence_parallel_size != 1:
            raise ValueError('input v1 supports no teacher and sequence_parallel_size=1')
        if any(encoded['grpo_batch'].teacher_per_token_logps is not None for encoded in batch_encoded_inputs):
            raise ValueError('input v1 does not support teacher-conditioned advantages')
        advantages = self._compute_advantages(samples,self._rewards_per_func,batch_encoded_inputs)
        counts = gather_object([len(samples)])
        rank = self.accelerator.process_index
        start = sum(counts[:rank])
        local = advantages[start:start+counts[rank]]
        if len(local) != len(samples):
            raise ValueError('variable-turn advantage routing mismatch')
        for sample,advantage in zip(samples,local):
            sample.advantages = advantage
        self._logs['advantages'].extend(advantages.tolist())
        chunks = self.split_by_mini_batches(samples)
        if len(chunks) != len(batch_encoded_inputs):
            raise ValueError('microbatch alignment mismatch')
        for chunk,encoded in zip(chunks,batch_encoded_inputs):
            batch=encoded['grpo_batch']
            batch.advantages=expand_advantage_to_per_token(
                torch.stack([s.advantages.to(self.accelerator.device) for s in chunk]),batch.completion_mask)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not enabled():
            return super().training_step(model, inputs, num_items_in_batch)
        # A generation batch contains episodes; expanding it into decisions can
        # produce hundreds of rows. Liger's fused dispatch bypasses our chunk
        # hook, and accumulating chunk graphs also retains all activations.
        # Backpropagate each immutable decision before computing the next one.
        import time
        import torch
        from contextlib import nullcontext
        from accelerate.utils import gather_object
        if self.args.async_generate or self.args.loss_type != 'grpo':
            raise ValueError('conditional backward requires synchronous GRPO')
        if self.accelerator.distributed_type.value not in ('NO', 'MULTI_GPU'):
            raise ValueError('conditional backward supports plain/DDP training only')
        started = time.perf_counter()
        model.train()
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()
        prepared = self._prepare_inputs(inputs)
        if isinstance(prepared, list):
            if len(prepared) != 1:
                raise ValueError('conditional backward requires one generation microbatch')
            prepared = prepared[0]
        samples = prepared['_origin_data']
        weights = episode_token_weights(samples)
        denominator = sum(weights)
        max_chunks = max(gather_object([len(samples)]))
        total = torch.zeros((), device=self.accelerator.device)
        metrics = []
        chunk_inputs = chunk_batch = None
        for i in range(max_chunks):
            real = i < len(samples)
            if real:
                chunk_inputs, chunk_batch = self.get_chunked_inputs(
                    prepared['model_inputs'], prepared['grpo_batch'], i, i+1, samples)
            # Every rank runs the same number of forwards (Swift metrics use
            # collectives). Dummy decisions contribute zero loss. Synchronize
            # accumulated DDP gradients only on the final common backward.
            context = self.accelerator.no_sync(model) if i+1 < max_chunks else nullcontext()
            with context, self.compute_loss_context_manager():
                loss, metric = self._compute_loss_and_metrics(model, chunk_inputs, chunk_batch)
                weight = weights[i] if real else 0.
                weighted = loss * (weight / denominator / self.current_gradient_accumulation_steps)
                self.accelerator.backward(weighted)
            total += weighted.detach()
            if real:
                metrics.append((metric, weight))
            del loss, weighted
        aligned = align_adam_moments(self.optimizer)
        if aligned:
            path = Path(self.args.output_dir) / 'optimizer_resume_alignment.jsonl'
            with path.open('a') as stream:
                stream.write(json.dumps({'step': self.state.global_step,
                    'rank': self.accelerator.process_index, 'aligned_moments': aligned})+'\n')
        self._aggregate_and_update_metrics(metrics, 'train')
        self._step += 1
        self._current_train_step_time += time.perf_counter() - started
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics['train']['step_time'].append(self._current_train_step_time)
            self._current_train_step_time = 0.
        return total

    def _compute_loss_chunked(self, model, model_inputs, grpo_batch, origin_data=None):
        if not enabled():
            return super()._compute_loss_chunked(model,model_inputs,grpo_batch,origin_data)
        # Preserve GRPO's mean-token loss per episode, instead of giving a
        # short decision the same weight as an entire long decision.
        import torch
        from accelerate.utils import gather_object
        if origin_data is None or self.args.loss_type != 'grpo':
            raise ValueError('conditional turn weighting requires GRPO and original turn samples')
        weights = episode_token_weights(origin_data)
        max_chunks = max(gather_object([len(origin_data)]))
        losses, metrics = [], []
        chunk_inputs = chunk_batch = None
        for i in range(max_chunks):
            real = i < len(origin_data)
            if real:
                chunk_inputs,chunk_batch = self.get_chunked_inputs(model_inputs,grpo_batch,i,i+1,origin_data)
            loss, metric = self._compute_loss_and_metrics(model,chunk_inputs,chunk_batch)
            weight = weights[i] if real else 0.
            losses.append(loss * weight)
            if real:
                metrics.append((metric,weight))
        self._aggregate_and_update_metrics(metrics,'train' if self.model.training else 'eval')
        return torch.stack(losses).sum() / sum(weights)


def episode_token_weights(samples):
    totals = {}
    counts = [sum(sum(mask) for mask in sample.response_loss_mask) for sample in samples]
    for sample,count in zip(samples,counts):
        totals[sample.request_id] = totals.get(sample.request_id,0)+count
    if any(value <= 0 for value in totals.values()):
        raise ValueError('episode has no generated target tokens')
    return [count/totals[sample.request_id] for sample,count in zip(samples,counts)]


def align_adam_moments(optimizer):
    """Match restored Adam moments to parameters after colocate dtype changes.

    Adam's scalar step counters retain their original dtype/device. Moment
    values and optimizer progress are preserved; only dtype/device are aligned.
    """
    changed = 0
    for group in optimizer.param_groups:
        for parameter in group['params']:
            state = optimizer.state.get(parameter, {})
            for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
                moment = state.get(key)
                if moment is None:
                    continue
                if moment.shape != parameter.shape:
                    raise ValueError('restored Adam moment shape differs from parameter')
                if moment.dtype != parameter.dtype or moment.device != parameter.device:
                    state[key] = moment.to(device=parameter.device, dtype=parameter.dtype)
                    changed += 1
    return changed
