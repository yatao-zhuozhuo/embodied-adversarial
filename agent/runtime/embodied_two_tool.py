"""Strict model boundary for the PickCube native-action self-play protocol."""

from __future__ import annotations

import json
import math
from typing import Any

from agent.runtime.two_tool_xml import PROTOCOL_VERSION, parse_decision, protocol_prompt

LEGACY_PROTOCOL_VERSION = "openeta.embodied_two_tool.v1"
MODEL_TOOLS = ("observe_env", "step_env")
ALICE_TASK = "Move the red cube to a new reachable position. The original target is hidden."
BOB_TASK = "Move the red cube to the visible green target created by Alice."


def parse_legacy_command(text: str, *, action_low: list[float], action_high: list[float]) -> dict[str, Any]:
    """Parse one answer-channel JSON command without coercion or clipping."""
    answer = str(text).strip()
    if "<think>" in answer:
        if "</think>" not in answer:
            raise ValueError("unclosed thinking block")
        answer = answer.rsplit("</think>", 1)[-1].strip()
    try:
        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate key: {key}")
                result[key] = value
            return result
        command = json.loads(answer,
                             parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                             object_pairs_hook=unique_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON command: {exc}") from exc
    if not isinstance(command, dict):
        raise ValueError("command must be an object")
    if command == {"response": "finish"}:
        return command
    if command == {"tool": "observe_env"}:
        return command
    if set(command) != {"tool", "action", "num_steps"} or command.get("tool") != "step_env":
        raise ValueError("only observe_env, step_env and finish are allowed")
    if len(action_low) != 7 or len(action_high) != 7:
        raise ValueError("expected seven-dimensional action space")
    action = command["action"]
    if not isinstance(action, list) or len(action) != 7:
        raise ValueError("action must contain exactly seven numbers")
    for index, (value, low, high) in enumerate(zip(action, action_low, action_high)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"action[{index}] must be a finite number")
        if not (low <= value <= high):
            raise ValueError(f"action[{index}] outside [{low}, {high}]")
    steps = command["num_steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 10:
        raise ValueError("num_steps must be an integer in [1, 10]")
    return {"tool": "step_env", "action": [float(value) for value in action], "num_steps": steps}


def parse_command(text: str, *, action_low: list[float], action_high: list[float],
                  remaining_control_steps: int | None = None) -> dict[str, Any]:
    result = parse_decision(text, action_low=action_low, action_high=action_high,
                            remaining_control_steps=remaining_control_steps)
    decision = result["decision"]
    if decision["kind"] == "response":
        return {"response": "finish"}
    return {"tool": decision["name"], **decision["parameters"]}


def project_observation(payload: dict[str, Any]) -> dict[str, Any]:
    """Allowlist model-visible proprioception; RGB is passed separately as an image."""
    robot = payload.get("robot") or {}
    if not isinstance(robot, dict):
        raise ValueError("missing robot observation")
    pose = robot.get("end_effector_pose")
    grip = robot.get("gripper_state")
    if not isinstance(pose, dict):
        raise ValueError("missing TCP pose")
    if not isinstance(grip, dict):
        raise ValueError("missing gripper state")
    result = {
        "joint_positions": robot.get("joint_positions"),
        "joint_velocities": robot.get("joint_velocities"),
        "end_effector_pose": {key: pose.get(key) for key in ("xyz", "quat_xyzw")},
        "gripper_state": {key: grip.get(key) for key in ("openness", "open")},
    }
    return {"robot": result}


def build_prompt(role: str, visible: dict[str, Any], *, turn: int, max_turns: int,
                 remaining_control_steps: int, feedback: dict[str, Any] | None = None) -> str:
    task = ALICE_TASK if role == "alice" else BOB_TASK
    intro = (
        "You control a Panda using PickCube-v1 pd_ee_delta_pose. "
        "The seven normalized action slots are TCP x,y,z translation, TCP x,y,z rotation, "
        "and gripper (+1 open, -1 close). A step repeats the same vector for 1-10 "
        "environment control steps. Read the RGB image and robot state. "
        + protocol_prompt(7)
    ) if turn == 0 else "Return one complete XML decision using openeta.two_tool_xml.v1."
    return (f"{intro}\nTask: {task}\nTurn {turn}/{max_turns}; "
            f"remaining control steps: {remaining_control_steps}.\n"
            f"Robot: {json.dumps(visible['robot'], separators=(',', ':'))}\n"
            f"Previous tool feedback: {json.dumps(feedback or {}, separators=(',', ':'))}")
