from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from agent.training.conditional_turns import ConditionalTurnsMixin

class InputGRPOTrainer(ConditionalTurnsMixin, GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from agent.training.curriculum import install_zero_reward_stop
        install_zero_reward_stop(self)
