"""Native-reward gates for a staged, single-policy curriculum."""
import json
from pathlib import Path


def summarize_groups(rows, start_step=0):
    # HF checkpoints carry the previous stages' log history. Count this stage
    # once per global step, using only actual native reward log entries.
    groups = {}
    for row in rows:
        if 'reward' not in row:
            continue
        step = row.get('step', row.get('global_step'))
        if step is None and 'global_step/max_steps' in row:
            step = int(row['global_step/max_steps'].split('/')[0])
        if step is not None and int(step) > start_step:
            groups[int(step)] = float(row['reward'])
    rewards = list(groups.values())
    return {'groups': len(rewards),
        'success_rate': sum(rewards)/len(rewards) if rewards else 0.,
        'informative_groups': sum(0. < r < 1. for r in rewards),
        'all_zero': bool(rewards) and not any(rewards)}


def gate_result(summary, successful_tasks, gate):
    return (summary['success_rate'] >= gate['min_success_rate']
        and successful_tasks >= gate['min_successful_tasks']
        and (summary['informative_groups'] > 0 or summary['success_rate'] >= .8))


def install_zero_reward_stop(trainer):
    import os
    limit = int(os.environ.get('OPENETA_CURRICULUM_ZERO_LIMIT', '0'))
    if not limit:
        return
    from transformers import TrainerCallback
    class ZeroRewardStop(TrainerCallback):
        def __init__(self):
            self.zeros = 0
            self.last_step = -1
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs or 'reward' not in logs or state.global_step == self.last_step:
                return control
            self.last_step = state.global_step
            self.zeros = self.zeros+1 if float(logs['reward']) == 0. else 0
            if self.zeros >= limit:
                control.should_training_stop = True
                control.should_save = True
                if state.is_world_process_zero:
                    Path(args.output_dir, 'curriculum_zero_reward_stop.json').write_text(json.dumps({
                        'step':state.global_step, 'consecutive_zero_groups':self.zeros,
                        'reason':'No native success reward; do not promote to harder tasks'})+'\n')
            return control
    trainer.add_callback(ZeroRewardStop())
