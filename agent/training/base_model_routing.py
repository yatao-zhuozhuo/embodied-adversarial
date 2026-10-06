"""Explicit base routing: Swift otherwise selects the online Alice LoRA for None."""
from contextlib import contextmanager


@contextmanager
def base_model_inference(engine, enabled):
    if not enabled:
        yield
        return
    lora = engine.enable_lora
    default = engine.default_adapter_request
    engine.enable_lora = False
    engine.default_adapter_request = None
    try:
        yield
    finally:
        engine.enable_lora = lora
        engine.default_adapter_request = default
