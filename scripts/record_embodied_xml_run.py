"""Record protocol/config/source identity and reject legacy-directory reuse."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.runtime.two_tool_xml import PROTOCOL_VERSION, PROMPT_VERSION, INVALID_POLICY


def main():
    root = Path(sys.argv[1]).resolve()
    target = root / "run_manifest.json"
    if target.exists():
        if json.loads(target.read_text()).get("protocol_version") != PROTOCOL_VERSION:
            raise SystemExit("Refusing to mix XML with an existing protocol directory")
        if json.loads(target.read_text()).get('input_protocol') != os.environ.get('OPENETA_INPUT_PROTOCOL'):
            raise SystemExit('Refusing to mix input protocols in an existing run')
        return
    if list(root.glob("round-*")):
        raise SystemExit("Existing rounds have no XML manifest; use a new run directory")
    files = ["agent/runtime/two_tool_xml.py", "agent/runtime/planner_xml.py",
             "agent/runtime/embodied_two_tool.py", "plugins/embodied_swift_grpo.py",
             "agent/training/embodied_staged_colocate.py", "agent/training/base_model_routing.py",
             "scripts/run_embodied_swift_staged_formal5.sh", "scripts/run_embodied_swift_formal10.sh",
             "scripts/run_embodied_swift_staged_colocate.sh"]
    keys = ["OPENETA_INPUT_PROTOCOL", "OPENETA_CAMERA_RESOLUTION", "OPENETA_COMPLETION_SCOPE", "MODEL_PATH", "TRAIN_GPU", "NUM_GENERATIONS", "GENERATION_BATCH_SIZE",
            "LORA_RANK", "LORA_ALPHA", "OPENETA_BOB_EVALUATIONS",
            "OPENETA_INITIALIZE_FROM_BASE", "OPENETA_ENABLE_THINKING", "OPENETA_THINKING_MAX_TOKENS", "MAX_TURNS",
            "OPENETA_MAX_CONTROL_STEPS", "MAX_LENGTH", "MAX_COMPLETION_LENGTH",
            "VLLM_MAX_MODEL_LEN", "MAX_PIXELS", "ROUNDS", "TASKS_PER_ROUND",
            "TRAIN_EPOCHS", "TRAIN_STEPS", "LR", "TEMPERATURE", "TOP_P",
            "INITIAL_ALICE_ADAPTER", "INITIAL_BOB_ADAPTER", "OPENETA_INITIAL_POLICY_ORIGIN"]
    payload = {"schema_version": "openeta.xml_training_run.v1",
               "protocol_version": PROTOCOL_VERSION, "prompt_version": PROMPT_VERSION,
               "invalid_policy": INVALID_POLICY, "initial_observation": "host",
               "supervision": "all_generated_thinking_and_answer_tokens",
               "config": {key: os.environ.get(key) for key in keys},
               "code_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}}
    if os.environ.get('OPENETA_INPUT_PROTOCOL') == 'openeta.two_tool_input.v1':
        payload['input_protocol'] = 'openeta.two_tool_input.v1'
        payload['runtime_versions'] = {name: importlib.metadata.version(name) for name in ('ms-swift','transformers','vllm')}
        payload['input_profile_source'] = 'live create_env contract; profile/hash persisted in each decision input'
        for name in ('agent/runtime/two_tool_input.py','agent/training/conditional_turns.py','agent/training/grpo_input.py','agent/training/input_grpo_trainer.py','sim/two_tool_contract.py'):
            payload['source_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
