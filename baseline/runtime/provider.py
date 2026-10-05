"""Explicit inference configuration shared by smoke and formal evaluation."""
import json
import os
from pathlib import Path
from agent.backends.planner import OpenAICompatiblePlannerBackend, OpenAICompatiblePlannerBackendConfig

BASELINE = Path(__file__).resolve().parents[1]


def build_backend(name, *, timeout_s=180):
    entry = json.loads((BASELINE / 'configs/models.json').read_text())[name]
    key = os.environ.get(entry['api_key_env']) or ('local-no-auth' if entry.get('local') else '')
    relay = os.environ.get('BASELINE_DEEPSEEK_RELAY_DIR') if name == 'deepseek' else None
    if relay:
        key = 'relay-credentials-kept-on-egress-host'
    if not key:
        raise ValueError(f"Missing {entry['api_key_env']}")
    config = OpenAICompatiblePlannerBackendConfig(
        model=entry['model'], api_base=entry['api_base'], api_key=key,
        temperature=0.0, max_tokens=1024, context_window_tokens=32768,
        enable_thinking=False, thinking_format=entry.get('thinking_format', 'chat_template_kwargs'),
        enable_vision=True, max_vision_images=2, use_json_response_format=False,
        timeout_s=timeout_s, max_attempts=1, fallback=None,
    )
    if relay:
        from baseline.runtime.relay import SharedStorageBackend
        return SharedStorageBackend(config, relay)
    return OpenAICompatiblePlannerBackend(config)
