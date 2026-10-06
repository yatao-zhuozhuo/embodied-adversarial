"""Check base-Bob routing against the installed Swift engine without loading weights."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent.training.base_model_routing import base_model_inference
from swift.infer_engine.grpo_vllm_engine import GRPOVllmEngine, VLLM_LORA_INT_ID
from swift.infer_engine.vllm_engine import VllmEngine

engine = object.__new__(GRPOVllmEngine)
engine.enable_lora = True
engine.default_adapter_request = None
engine.engine = SimpleNamespace(list_loras=lambda: {VLLM_LORA_INT_ID})
with patch.object(VllmEngine, "infer", return_value=[]) as infer:
    engine.infer([], adapter_request=None)
    assert infer.call_args.kwargs["adapter_request"] is not None
    with base_model_inference(engine, True):
        engine.infer([], adapter_request=None)
        assert infer.call_args.kwargs["adapter_request"] is None
    assert engine.enable_lora is True
print(json.dumps({"base_bob_bypasses_alice_lora": True, "engine_settings_restored": True}))
