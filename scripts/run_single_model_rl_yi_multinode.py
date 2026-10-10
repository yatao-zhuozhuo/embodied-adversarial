"""Coordinate one curriculum over two job nodes, with node-local simulators."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_single_model_curriculum import write, latest_checkpoint, successful_tasks
from agent.training.curriculum import summarize_groups, gate_result
from scripts.single_model_rl_preflight_history import validate_history


def remaining_stages(plan):
    checkpoint = plan.get('initial_checkpoint')
    step = json.loads((Path(checkpoint)/'trainer_state.json').read_text())['global_step'] if checkpoint else 0
    if not 0 <= step < plan['total_updates']:
        raise ValueError('Resume checkpoint is outside the unfinished curriculum')
    for stage in plan['stages']:
        if step < stage['target_step']:
            yield stage, step > stage['start_step']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--original-acceptance', type=Path, required=True)
    args = p.parse_args()
    root = args.run_root.resolve()
    rank = int(os.environ['NODE_RANK'])
    world = int(os.environ['NNODES'])
    sync = root / 'coordination'
    sync.mkdir(parents=True, exist_ok=True)
    local = root / 'nodes' / str(rank)
    local.mkdir(parents=True, exist_ok=True)

    def check_peers():
        failures = list(sync.glob('failed-*.json'))
        if failures:
            raise RuntimeError('Peer failed: ' + failures[0].read_text())

    def wait(path, timeout=7200):
        deadline = time.monotonic() + timeout
        while not path.exists():
            check_peers()
            if time.monotonic() > deadline:
                raise TimeoutError(str(path))
            time.sleep(2)
        return json.loads(path.read_text())

    def barrier(name):
        write(sync / f'{name}-{rank}.json', {'node_rank': rank})
        for peer in range(world):
            wait(sync / f'{name}-{peer}.json')

    def command(argv, env, logpath):
        with logpath.open('a') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while child.poll() is None:
                    check_peers()
                    time.sleep(2)
                if child.returncode:
                    raise RuntimeError(f'Exit {child.returncode}: {logpath}')
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=60)

    try:
        ips = subprocess.check_output(['hostname', '-I'], text=True).split()
        ip = next(ip for ip in ips if ':' not in ip and not ip.startswith('127.'))
        write(sync / f'node-{rank}.json', {'ip': ip, 'host': os.uname().nodename})
        nodes = [wait(sync / f'node-{peer}.json') for peer in range(world)]
        if len({node['ip'] for node in nodes}) != world:
            raise RuntimeError('Expected distinct hosts for the two 8-GPU jobs')
        os.environ['MASTER_ADDR'] = nodes[0]['ip']
        command([sys.executable, '-m', 'torch.distributed.run', '--nnodes', str(world),
                 '--node_rank', str(rank), '--nproc_per_node', '8', '--master_addr', nodes[0]['ip'],
                 '--master_port', os.environ['MASTER_PORT'],
                 'scripts/probe_single_model_rl_yi_distributed.py'], os.environ.copy(), local/'distributed_probe.log')
        barrier('distributed-probe')
        if rank == 0:
            command([sys.executable, '-m', 'pytest', '-q',
                     'tests/test_single_model_yi_multinode.py', 'tests/test_single_model_yi_config.py',
                     'tests/test_single_model_curriculum.py', 'tests/test_worker_gpu_assignment.py',
                     'tests/test_simulator_mcp_sse_order.py', 'tests/test_curriculum_resume_position.py'],
                    os.environ.copy(), local/'contract_tests.log')
        barrier('contract-tests')
        plan = json.loads((root / 'curriculum_plan.json').read_text())
        batch = plan['batch']
        assert world == batch['nnodes'] == 2
        assert batch['num_generations'] * batch['tasks_per_batch'] == batch['generation_batch_size'] == 64
        reuse=bool(plan.get('preflight_history'))
        if reuse:
            canaries=validate_history(plan)
            dataset=local/'native_canaries.jsonl'
            dataset.write_text(''.join(json.dumps(row)+'\n' for row in canaries))
            if rank == 0:
                write(root/'curriculum_status.json', {'status':'preflight_canary','history':plan['preflight_history']})
            command([sys.executable,'scripts/preflight_single_model_rl.py','--dataset',str(dataset),
                     '--output',str(local/'native_canaries')],os.environ.copy(),local/'canary.log')
            write(local/'preflight_reuse.json', {'status':'passed','history':plan['preflight_history'],
                'source_revision':plan['preflight_source_revision'],'canary_benchmarks':4})
            barrier('native-canaries')
        # Validate both difficulty stages before spending time on any updates.
        for stage in ([] if reuse else plan['stages']):
            node_stage = local / stage['name']
            node_stage.mkdir(parents=True, exist_ok=True)
            if rank == 0:
                write(root/'curriculum_status.json', {'status':'preflight','stage':stage['name']})
            command([sys.executable, 'scripts/preflight_single_model_rl.py', '--dataset',
                str(Path(stage['directory'])/'preflight_tasks.jsonl'), '--output', str(node_stage/'preflight')],
                os.environ.copy(), node_stage/'preflight.log')
            barrier(stage['name']+'-preflight')
        # New training starts only after the original configuration saved a real update.
        wait(args.original_acceptance)
        checkpoint = Path(plan['initial_checkpoint']) if plan.get('initial_checkpoint') else None
        for stage, same_stage_resume in remaining_stages(plan):
            directory = Path(stage['directory'])
            node_stage = local / stage['name']
            node_stage.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(RUN_ROOT=str(directory), OPENETA_RL_CONFIG=str(directory/'config.json'),
                MODEL_PATH=plan['model'], DATASET_PATH=stage['dataset'], OUTPUT_DIR=stage['output'],
                TRAIN_STEPS=str(stage['target_step']), TRAIN_EPOCHS=str(stage['epochs']),
                MAX_TURNS=str(plan['max_turns']), OPENETA_RL_TURN_TOKENS=str(plan['per_turn_tokens']),
                OPENETA_RL_EPISODE_TOKENS=str(plan['episode_tokens']), VLLM_MAX_MODEL_LEN=str(plan['context_length']),
                NUM_GENERATIONS=str(batch['num_generations']), PER_DEVICE_TRAIN_BATCH_SIZE=str(batch['per_device_train_batch_size']),
                GRADIENT_ACCUMULATION_STEPS='1', DATASET_SHUFFLE='false', IGNORE_DATA_SKIP='false' if same_stage_resume or not checkpoint else 'true',
                OPENETA_CURRICULUM_ZERO_LIMIT=str(plan['zero_limit']),
                OPENETA_CURRICULUM_START_STEP=str(stage['start_step']))
            env.pop('TRAIN_ADAPTER_PATH', None)
            env.pop('RESUME_FROM_CHECKPOINT', None)
            if checkpoint:
                env['RESUME_FROM_CHECKPOINT'] = str(checkpoint)
            if rank == 0:
                write(root/'curriculum_status.json', {'status':'training','stage':stage['name'],
                    'start_step':stage['start_step'],'target_step':stage['target_step'],
                    'resume':str(checkpoint) if checkpoint else None})
            command(['bash','scripts/run_single_model_grpo.sh'], env,
                    directory/'train.log' if rank == 0 else node_stage/'train.log')
            barrier(stage['name']+'-trained')
            result_path = directory/'stage_result.json'
            if rank == 0:
                step, checkpoint, state = latest_checkpoint(stage['output'])
                summary = summarize_groups(state['log_history'],stage['start_step'])
                report = {**summary,'stage':stage['name'],'step':step,'checkpoint':str(checkpoint),
                          'successful_tasks':successful_tasks(stage['output'], plan.get('history_outputs',{}).get(stage['name'],[]))}
                report['promoted'] = step == stage['target_step'] and (not stage['promotion'] or gate_result(summary, report['successful_tasks'], stage['promotion']))
                write(result_path, report)
            report = wait(result_path)
            checkpoint = Path(report['checkpoint'])
            if not report['promoted']:
                if rank == 0:
                    write(root/'curriculum_status.json', {'status':'needs_reward_signal','detail':report})
                return
        if rank == 0:
            write(root/'curriculum_status.json', {'status':'complete','checkpoint':str(checkpoint),'step':plan['total_updates']})
    except BaseException as exc:
        write(sync/f'failed-{rank}.json', {'node_rank':rank,'error':str(exc)})
        if rank == 0:
            write(root/'curriculum_status.json', {'status':'failed','error':str(exc)})
        raise


if __name__ == '__main__':
    main()
