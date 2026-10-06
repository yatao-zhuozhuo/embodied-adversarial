from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
from agent.training.conditional_turns import ConditionalTurnsMixin

class InputGRPOTrainer(ConditionalTurnsMixin, GRPOTrainer):
    pass
