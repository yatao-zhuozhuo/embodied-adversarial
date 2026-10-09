"""Prepare and supervise ordered native-task GRPO stages with reward gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.create_single_model_rl_dataset import build_rows
from agent.training.curriculum import summarize_groups, gate_result


def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n')
    temporary.replace(path)


def prepare(config_path, root):
    cfg=json.loads(config_path.read_text())
    holdout=json.loads((ROOT/cfg['holdout']).read_text())
    baseline=json.loads((ROOT/cfg['baseline_exclusion']).read_text())
    stages=[]; cumulative=0
    for index,stage in enumerate(cfg['curriculum']):
        directory=root/'stages'/stage['name']
        manifest={'schema_version':'openeta.single_model_rl_tasks.v2','tasks':stage['tasks']}
        manifest_path=directory/'manifest.json'
        write(manifest_path,manifest)
        rows=[]
        for seed_index in range(stage['seeds_per_task']):
            rows.extend(build_rows(manifest,holdout,baseline=baseline,
                seed=cfg['seed_base']+index*100000+seed_index*10000,
                max_turns=cfg['max_turns'],max_control_steps=cfg['max_control_steps']))
        dataset=directory/'tasks.jsonl'
        dataset.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        unique=rows[:len(stage['tasks'])]
        (directory/'preflight_tasks.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in unique))
        phase_cfg={k:v for k,v in cfg.items() if k not in ('curriculum','consecutive_zero_group_limit')}
        phase_cfg.update(task_manifest=str(manifest_path),epochs=stage['epochs'])
        write(directory/'config.json',phase_cfg)
        updates=len(rows)*stage['epochs']
        stages.append({**stage,'directory':str(directory),'dataset':str(dataset),
            'output':str(root/'formal/trainer'/stage['name']),
            'start_step':cumulative,'target_step':cumulative+updates,'rows':len(rows)})
        cumulative+=updates
    plan={'source_config':str(config_path),'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'model':cfg['model'],'max_turns':cfg['max_turns'],'per_turn_tokens':cfg['per_turn_tokens'],
        'episode_tokens':cfg['episode_tokens'],'context_length':cfg['context_length'],
        'zero_limit':cfg['consecutive_zero_group_limit'],'total_updates':cumulative,'stages':stages}
    write(root/'curriculum_plan.json',plan)
    return plan


def history_sources(plan):
    inherited=plan.get('history_outputs',{})
    return {stage['name']:list(dict.fromkeys([stage['output'],*inherited.get(stage['name'],[])]))
        for stage in plan['stages']}


def latest_checkpoint(output):
    candidates=[]
    for path in Path(output).glob('v*/checkpoint-*/trainer_state.json'):
        state=json.loads(path.read_text())
        candidates.append((state['global_step'],path.parent,state))
    if not candidates:
        raise RuntimeError('Training did not save a checkpoint')
    return max(candidates,key=lambda item:item[0])


def successful_tasks(output, history_outputs=()):
    tasks=set()
    paths=[path for directory in [output,*history_outputs] for path in (Path(directory)/'rollouts/episodes').glob('*.json')]
    for path in paths:
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-8192))
            tail=stream.read().decode()
        pos=tail.rfind('"final_info":')
        info,_=json.JSONDecoder().raw_decode(tail[pos+len('"final_info":'):].lstrip())
        if info['success']:
            tasks.add((info['bench'],info['task_id']))
    return len(tasks)


def run(plan,root):
    checkpoint=Path(plan['initial_checkpoint']) if plan.get('initial_checkpoint') else None
    initial_step=json.loads((checkpoint/'trainer_state.json').read_text())['global_step'] if checkpoint else 0
    first_resumed_stage=True
    prefix='source scripts/single_model_rl_env.sh\nexport OPENETA_HOST_STATE_TOKEN="$(cat "$OPENETA_HOST_TOKEN_FILE")"\n'
    for stage in plan['stages']:
        if checkpoint and initial_step >= stage['target_step']:
            continue
        same_stage_resume=bool(checkpoint and first_resumed_stage and initial_step>stage['start_step'])
        directory=Path(stage['directory'])
        env=os.environ.copy()
        env.update(RUN_ROOT=str(directory),OPENETA_RL_CONFIG=str(directory/'config.json'),
            MODEL_PATH=plan['model'],DATASET_PATH=stage['dataset'],OUTPUT_DIR=stage['output'],
            TRAIN_STEPS=str(stage['target_step']),TRAIN_EPOCHS='5',MAX_TURNS=str(plan['max_turns']),
            OPENETA_RL_TURN_TOKENS=str(plan['per_turn_tokens']),
            OPENETA_RL_EPISODE_TOKENS=str(plan['episode_tokens']),VLLM_MAX_MODEL_LEN=str(plan['context_length']),
            DATASET_SHUFFLE='false',IGNORE_DATA_SKIP='false' if same_stage_resume or not checkpoint else 'true',
            OPENETA_CURRICULUM_ZERO_LIMIT=str(plan['zero_limit']))
        env.pop('TRAIN_ADAPTER_PATH',None)
        env.pop('RESUME_FROM_CHECKPOINT',None)
        if checkpoint:
            env['RESUME_FROM_CHECKPOINT']=str(checkpoint)
        write(root/'curriculum_status.json',{'status':'preflight','stage':stage['name']})
        preflight=prefix+('/opt/openeta-swift/bin/python scripts/preflight_single_model_rl.py '
            '--dataset "$RUN_ROOT/preflight_tasks.jsonl" --output "$RUN_ROOT/preflight"\n')
        with (directory/'preflight.log').open('a') as log:
            result=subprocess.run(['bash','-c',preflight],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Preflight failed: {stage['name']}; inspect preflight.log")
        write(root/'curriculum_status.json',{'status':'training','stage':stage['name'],
            'start_step':stage['start_step'],'target_step':stage['target_step'],'resume':str(checkpoint) if checkpoint else None})
        script=prefix+'bash scripts/run_single_model_grpo.sh\nstatus=$?\nprintf "%s\\n" "$status" > "$RUN_ROOT/train.exit"\nexit "$status"\n'
        with (directory/'train.log').open('a') as log:
            child=subprocess.Popen(['bash','-c',script],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            write(directory/'train.submission.json',{'kind':'train','pid':child.pid,'hostname':os.uname().nodename,
                'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                'log':str(directory/'train.log'),'dataset':stage['dataset'],'output':stage['output']})
            code=child.wait()
        if code:
            raise RuntimeError(f"Training failed: {stage['name']}; inspect train.log")
        step,checkpoint,state=latest_checkpoint(stage['output'])
        summary=summarize_groups(state['log_history'],stage['start_step'])
        report={**summary,'stage':stage['name'],'step':step,'checkpoint':str(checkpoint),
            'successful_tasks':successful_tasks(stage['output'],plan.get('history_outputs',{}).get(stage['name'],[]))
                if plan.get('history_outputs',{}).get(stage['name']) else successful_tasks(stage['output'])}
        first_resumed_stage=False
        write(directory/'stage_result.json',report)
        if step!=stage['target_step'] or (stage['promotion'] and not gate_result(summary,report['successful_tasks'],stage['promotion'])):
            write(root/'curriculum_status.json',{'status':'needs_reward_signal','detail':report,
                'reason':'Stage has not earned promotion; harder stages were not started'})
            return
    write(root/'curriculum_status.json',{'status':'complete','checkpoint':str(checkpoint),'step':plan['total_updates']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--resume-checkpoint',type=Path)
    parser.add_argument('--history-plan',type=Path)
    args=parser.parse_args();root=args.run_root.resolve()
    if args.worker:
        try:
            run(json.loads((root/'curriculum_plan.json').read_text()),root)
            (root/'curriculum.exit').write_text('0\n')
        except Exception as error:
            write(root/'curriculum_status.json',{'status':'failed','error':str(error)})
            (root/'curriculum.exit').write_text('1\n')
            raise
        return
    if (root/'curriculum.submission.json').exists():
        parser.error('Existing submission; inspect its state and use a fresh directory')
    plan=prepare(args.config.resolve(),root)
    if args.resume_checkpoint:
        checkpoint=args.resume_checkpoint.resolve()
        state=json.loads((checkpoint/'trainer_state.json').read_text())
        for filename in ['adapter_model.safetensors','optimizer.pt','scheduler.pt']:
            if not (checkpoint/filename).is_file():parser.error('Incomplete resume checkpoint')
        step=state['global_step']
        if not 0 < step < plan['total_updates']:parser.error('Checkpoint step is outside curriculum')
        plan['initial_checkpoint']=str(checkpoint)
        if args.history_plan:
            old=json.loads(args.history_plan.read_text())
            if old['config_sha256']!=plan['config_sha256']:parser.error('Curriculum config differs from saved run')
            plan['history_outputs']=history_sources(old)
        write(root/'curriculum_plan.json',plan)
    if args.prepare_only:
        print(json.dumps({'plan':str(root/'curriculum_plan.json'),'updates':plan['total_updates']}));return
    env=os.environ.copy();env['RUN_ROOT']=str(root)
    with (root/'curriculum.log').open('a') as stream:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve()),
            '--run-root',str(root),'--worker'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    receipt={'kind':'curriculum','pid':child.pid,'hostname':os.uname().nodename,
        'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
        'log':str(root/'curriculum.log'),'plan':str(root/'curriculum_plan.json')}
    write(root/'curriculum.submission.json',receipt)
    series_root=os.environ.get('OPENETA_TENSORBOARD_SERIES_ROOT')
    if series_root:
        checkpoint=plan.get('initial_checkpoint')
        step=json.loads((Path(checkpoint)/'trainer_state.json').read_text())['global_step'] if checkpoint else 0
        write(Path(series_root)/'current_run.json',{'run_root':str(root),
            'resume_checkpoint':checkpoint,'resumed_global_step':step,
            'tensorboard':'openeta-rl-curriculum-continuous-20261009'})
    print(json.dumps(receipt))

if __name__=='__main__':main()
