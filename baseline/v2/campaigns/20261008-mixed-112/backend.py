"""Mixed checkpoint configuration; preserves the frozen v2 decoding settings."""
import json
from pathlib import Path
from agent.backends.planner import OpenAICompatiblePlannerBackend, OpenAICompatiblePlannerBackendConfig
C=Path(__file__).resolve().parent

def build_backend(name, *, timeout_s=180):
    entry=json.loads((C/'models.json').read_text())[name]
    config=OpenAICompatiblePlannerBackendConfig(model=entry['model'],api_base=entry['api_base'],api_key='local-no-auth',
        temperature=0.0,max_tokens=1024,context_window_tokens=32768,enable_thinking=False,
        thinking_format='chat_template_kwargs',enable_vision=True,max_vision_images=5,use_json_response_format=False,
        timeout_s=timeout_s,max_attempts=1,fallback=None)
    return OpenAICompatiblePlannerBackend(config)
