from types import SimpleNamespace
import pytest
from agent.training.conditional_turns import ConditionalTurnsMixin


@pytest.mark.parametrize('start,step,size,expected', [
    (0,6,4,(1,2)), (12,12,8,(1,0)), (12,25,8,(2,5)),
    (12,28,8,(3,0)), (12,35,8,(3,7)),
])
def test_resume_dataset_position_uses_stage_updates(monkeypatch,start,step,size,expected):
    monkeypatch.setenv('OPENETA_INPUT_PROTOCOL','openeta.two_tool_input.v1')
    monkeypatch.setenv('OPENETA_CURRICULUM_START_STEP',str(start))
    class Parent:
        def _init_training_state(self,*args):
            self.state=SimpleNamespace(global_step=step)
            return step//size, step%size
    class Trainer(ConditionalTurnsMixin,Parent):
        args=SimpleNamespace(ignore_data_skip=False,gradient_accumulation_steps=1)
    trainer=Trainer()
    assert trainer._init_training_state(36,size,5,'checkpoint',None)==expected
    assert trainer.state.global_step==step
    # First remaining task batch is computed from completed stage updates.
    assert expected[1]==(step-start)%size
