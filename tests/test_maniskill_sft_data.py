from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.collect_maniskill_sft import _build_wave
from scripts.export_openeta_sft import export_dataset
from scripts.generate_maniskill_sft_manifest import _load_catalog, build_manifest


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_collection_wave_does_not_starve_untried_tasks(tmp_path) -> None:
    tasks = [
        {
            "task_slug": slug,
            "env_id": f"example/{slug}",
            "instruction": f"Complete {slug}.",
            "tier": "A",
            "quota": 10,
        }
        for slug in ("pick_cube", "stack_cube", "peg_insertion_side")
    ]
    used = {"pick_cube": set(range(50))}
    wave = _build_wave(
        tasks,
        teacher_id="glm",
        raw_root=tmp_path,
        accepted=Counter(),
        used=used,
        batch_size=2,
    )
    assert [row["metadata"]["task_slug"] for row in wave] == [
        "stack_cube",
        "peg_insertion_side",
    ]
    assert [row["seed"] for row in wave] == [100_000, 200_000]


def test_generated_manifests_are_all_train_and_have_five_hundred_episodes(tmp_path) -> None:
    catalog_path = Path("SFT_data/configs/task_catalog.v1.json")
    tasks = _load_catalog(catalog_path)

    for teacher_id in ("glm", "qwen"):
        payload = build_manifest(
            tasks,
            teacher_id=teacher_id,
            output_root=tmp_path,
            seed_offset=0,
        )
        episodes = payload["episodes"]
        assert len(episodes) == 500
        assert len({row["episode_id"] for row in episodes}) == 500
        assert {row["metadata"]["dataset_split"] for row in episodes} == {"train"}
        assert all(row["metadata"]["require_official_reward"] is True for row in episodes)
        assert {row["metadata"]["include_objects"] for row in episodes} == {True}
        assert {row["metadata"]["perception_mode"] for row in episodes} == {
            "structured_state"
            if teacher_id == "glm"
            else "rgb_plus_structured_state"
        }
        assert all("every move_to must use the returned ik_receipt_id" in row["task"] for row in episodes)
        assert all("cuRobo endpoint collision checking is unavailable" in row["task"] for row in episodes)
        if teacher_id == "glm":
            assert all("objects list is the authoritative" in row["task"] for row in episodes)


def test_exporter_keeps_only_successful_main_planner_calls(tmp_path) -> None:
    bundle = tmp_path / "raw" / "glm" / "sessions" / "session-1" / "rollout"
    bundle.mkdir(parents=True)
    _write_json(
        bundle / "manifest.json",
        {
            "schema_version": "openeta.rollout.v1",
            "session_id": "session-1",
            "metadata": {
                "episode_id": "glm-pick-cube-seed-0",
                "env_id": "openeta/maniskill_PickCube-v1-v0",
                "seed": 0,
                "task_slug": "pick_cube",
                "teacher_model": "GLM-5.3-w8a8c8",
            },
            "files": {
                "model_calls": "model_calls.jsonl",
                "tool_calls": "tool_calls.jsonl",
                "transitions": "transitions.jsonl",
                "episodes": "episodes.jsonl",
                "artifacts": "artifacts.jsonl",
                "artifact_root": "artifacts",
            },
        },
    )
    _write_jsonl(
        bundle / "model_calls.jsonl",
        [
            {
                "schema_version": "openeta.rollout.model_call.v1",
                "seq": 1,
                "semantic_request": {
                    "system_prompt": "You are the OpenETA closed-loop embodied planner.",
                    "metadata": {},
                },
                "provider_exchange": {
                    "attempts": [
                        {
                            "response": {"choices": []},
                            "request_body": {
                                "model": "GLM-5.3-w8a8c8",
                                "messages": [
                                    {"role": "system", "content": "planner"},
                                    {"role": "user", "content": "move toward the cube"},
                                ],
                            },
                        }
                    ]
                },
                "result": {"model": "GLM-5.3-w8a8c8"},
                "parsed_decision": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "reasoning": "Approach the cube.",
                    "parameters": {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
                },
                "validation": {"accepted": True, "errors": []},
            }
        ],
    )
    _write_jsonl(
        bundle / "tool_calls.jsonl",
        [{"schema_version": "openeta.rollout.tool_event.v1", "seq": 1, "event": {}}],
    )
    _write_jsonl(
        bundle / "transitions.jsonl",
        [
            {
                "schema_version": "openeta.rollout.transition.v1",
                "seq": 1,
                "info": {
                    "environment_receipt_trusted": True,
                    "environment_success": True,
                    "environment_receipt": {"task_success": True},
                },
            }
        ],
    )
    _write_jsonl(
        bundle / "episodes.jsonl",
        [
            {"schema_version": "openeta.rollout.episode_event.v1", "seq": 1, "event": "start"},
            {
                "schema_version": "openeta.rollout.episode_event.v1",
                "seq": 2,
                "event": "result",
                "result": {"metadata": {"assistance": {"assisted": False}}},
            },
        ],
    )
    (bundle / "artifacts.jsonl").write_text("", encoding="utf-8")

    output = tmp_path / "exported" / "train.jsonl"
    report = tmp_path / "reports" / "export.json"
    accepted = tmp_path / "accepted" / "episodes.jsonl"
    payload = export_dataset(
        input_root=tmp_path / "raw" / "glm",
        output=output,
        report=report,
        accepted_index=accepted,
        teacher_id="glm",
        teacher_model="GLM-5.3-w8a8c8",
    )

    assert payload["accepted_episode_count"] == 1
    assert payload["sample_count"] == 1
    sample = json.loads(output.read_text(encoding="utf-8"))
    assert sample["split"] == "train"
    assert sample["metadata"]["episode_success"] is True
    assert sample["messages"][-1]["content"].startswith("<decision>")
    assert "<name>move_to</name>" in sample["messages"][-1]["content"]
