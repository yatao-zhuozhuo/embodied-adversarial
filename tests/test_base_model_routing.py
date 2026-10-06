from types import SimpleNamespace
import pytest
from agent.training.base_model_routing import base_model_inference


def test_base_bob_never_falls_back_to_online_alice_and_restores_on_error():
    alice = object()
    engine = SimpleNamespace(enable_lora=True, default_adapter_request=alice)
    with pytest.raises(RuntimeError):
        with base_model_inference(engine, True):
            assert engine.enable_lora is False
            assert engine.default_adapter_request is None
            raise RuntimeError("inference failed")
    assert engine.enable_lora is True
    assert engine.default_adapter_request is alice


def test_checkpoint_bob_keeps_lora_enabled():
    engine = SimpleNamespace(enable_lora=True, default_adapter_request=None)
    with base_model_inference(engine, False):
        assert engine.enable_lora is True
