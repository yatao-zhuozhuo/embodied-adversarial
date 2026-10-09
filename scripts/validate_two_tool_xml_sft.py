#!/usr/bin/env python3
"""Encode an XML SFT sample with installed Swift; audit images and loss masks."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.runtime.two_tool_xml import serialize_decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "SFT_data/reports/qwen_xml_v1/swift_encoding.json")
    args = parser.parse_args()
    from PIL import Image
    from swift.model import get_model_processor
    from swift.template import get_template

    profile = json.loads((REPO_ROOT / "SFT_data/configs/two_tool_xml_sft.v1.json").read_text())
    version = importlib.metadata.version("ms-swift")
    if version != profile["swift_version"]:
        raise ValueError(f"Expected Swift {profile['swift_version']}, found {version}")
    with tempfile.TemporaryDirectory(prefix="xml-encoding-") as directory:
        if args.dataset:
            sample = json.loads(next(line for line in args.dataset.read_text().splitlines() if line.strip()))
        else:
            image_path = Path(directory) / "fixture.png"
            Image.new("RGB", (256, 256), (40, 80, 120)).save(image_path)
            sample = {"messages": [
                {"role": "system", "content": "Return a single XML decision."},
                {"role": "user", "content": "INITIAL_OBSERVATION_ONLY <image>"},
                {"role": "assistant", "content": serialize_decision("observe_env", reasoning="HISTORY_ONLY"), "loss": False},
                {"role": "user", "content": "LATEST_OBSERVATION_ONLY <image>"},
                {"role": "assistant", "content": serialize_decision("step_env", {"action": [0] * 7, "num_steps": 1}, "TARGET_ONLY"), "loss": True},
            ], "images": [str(image_path), str(image_path)]}
        _, processor = get_model_processor(profile["validation_model"], load_model=False)
        template = get_template(processor, template_type=profile["template"], loss_scale=profile["loss_scale"],
                                enable_thinking=profile["enable_thinking"], truncation_strategy="raise")
        template.set_mode("train")
        encoded = template.encode(sample)
        tokenizer = processor.tokenizer
        input_ids, labels = encoded["input_ids"], encoded["labels"]
        supervised = tokenizer.decode([token for token in labels if token != -100])
        target = sample["messages"][-1]["content"]
        assert target in supervised, "Target XML is not supervised verbatim"
        assert "<think>" not in supervised and "</think>" not in supervised, "Template-only thinking must not be supervised"
        if not args.dataset:
            assert not any(marker in supervised for marker in ("HISTORY_ONLY", "INITIAL_OBSERVATION_ONLY", "LATEST_OBSERVATION_ONLY"))
        assert "tool_calls" not in sample and "tools" not in sample
        assert sum(message["content"].count("<image>") for message in sample["messages"]) == len(sample["images"])
        assert len(encoded["image_grid_thw"]) == len(sample["images"])
        assert all(message.get("loss") is False for message in sample["messages"][:-1] if message["role"] == "assistant")
        report = {"swift_version": version, "profile": profile, "effective_agent_template": template._agent_template,
                  "source": str(args.dataset) if args.dataset else "synthetic_encoding_fixture_not_training_data",
                  "image_count": len(sample["images"]), "input_tokens": len(input_ids),
                  "supervised_tokens": sum(token != -100 for token in labels),
                  "input_ids": input_ids, "labels": labels,
                  "rendered_prompt": tokenizer.decode(input_ids), "supervised_text": supervised,
                  "processor_config_sha256": hashlib.sha256((Path(profile["validation_model"]) / "tokenizer_config.json").read_bytes()).hexdigest()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: report[key] for key in ("swift_version", "source", "image_count", "input_tokens", "supervised_tokens")}))
        print(f"Encoding audit: {args.output}")


if __name__ == "__main__":
    main()
