from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from agent.training.conditional_turns import ConditionalTurnsMixin
from agent.training.rollout_cpu_sync import CpuRolloutControlMixin, create_rollout_cpu_group

class InputGRPOTrainer(CpuRolloutControlMixin, ConditionalTurnsMixin, GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._openeta_rollout_cpu_group = create_rollout_cpu_group()
        from agent.training.curriculum import install_zero_reward_stop
        install_zero_reward_stop(self)
