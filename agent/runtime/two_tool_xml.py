"""Strict, shared model boundary for baseline, SFT and GRPO.

No repair, format guessing, parameter correction or hidden tool access.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from typing import Any

from agent.runtime.planner_xml import parse_xml_decision

PROTOCOL_VERSION = "openeta.two_tool_xml.v1"
PROTOCOL = PROTOCOL_VERSION
PROMPT_VERSION = "openeta.two_tool_xml.prompt.v1"
INVALID_POLICY = {"version": "openeta.two_tool_invalid.v1", "consumes_turn": True,
                  "executes_action": False, "max_consecutive_invalid": 3,
                  "max_consecutive_observe": 3, "repair": "reject"}


def split_thinking(text: str) -> tuple[str, str]:
    text = str(text).strip()
    thinking = ""
    if text.startswith("<think>"):
        end = text.find("</think>")
        if end < 0:
            raise ValueError("unclosed thinking block")
        thinking, text = text[7:end], text[end + 8:].strip()
    if "<think>" in text or "</think>" in text or "<think>" in thinking:
        raise ValueError("unexpected thinking block")
    return thinking, text


split_answer = split_thinking


def validate_parameters(name: str, parameters: dict, *, action_low: list,
                        action_high: list, remaining_control_steps: int | None = None) -> dict:
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be an object")
    if name in ("observe_env", "task_complete"):
        if parameters:
            raise ValueError(f"{name} takes no model-controlled parameters")
        return {}
    if name != "step_env" or set(parameters) != {"action", "num_steps"}:
        raise ValueError("step_env requires exactly action and num_steps")
    action, steps = parameters["action"], parameters["num_steps"]
    if not action_low or len(action_low) != len(action_high):
        raise ValueError("missing native action contract")
    if not isinstance(action, list) or len(action) != len(action_low):
        raise ValueError(f"action must contain exactly {len(action_low)} numbers")
    for i, (x, low, high) in enumerate(zip(action, action_low, action_high)):
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError(f"action[{i}] must be a finite number")
        if not math.isfinite(low) or not math.isfinite(high) or low > high or not low <= x <= high:
            raise ValueError(f"action[{i}] outside [{low}, {high}]")
    if type(steps) is not int or not 1 <= steps <= 10:
        raise ValueError("num_steps must be an integer in [1, 10]")
    if remaining_control_steps is not None and steps > remaining_control_steps:
        raise ValueError("step exceeds remaining control-step budget")
    return {"action": list(action), "num_steps": steps}


def action_contract(created: dict) -> dict:
    dim, low, high = (created.get(k) for k in ("action_dim", "action_low", "action_high"))
    if type(dim) is not int or dim < 1 or not isinstance(low, list) or not isinstance(high, list) or len(low) != dim or len(high) != dim:
        raise ValueError("missing native action contract")
    for lo, hi in zip(low, high):
        if type(lo) not in (int, float) or type(hi) not in (int, float) or not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError("invalid native action bounds")
    return {"action_dim": dim, "action_low": list(low), "action_high": list(high)}


def parse_decision(text: str, *, action_low: list | None = None, action_high: list | None = None,
                   contract: dict | None = None, remaining_steps: int | None = None,
                   remaining_control_steps: int | None = None,
                   finish_reason: str | None = None) -> dict[str, Any]:
    if contract is not None:
        checked = action_contract(contract)
        action_low, action_high = checked["action_low"], checked["action_high"]
    if remaining_steps is not None:
        remaining_control_steps = remaining_steps
    if finish_reason in ("length", "max_tokens"):
        raise ValueError("truncated model output")
    thinking, answer = split_thinking(text)
    if "<!DOCTYPE" in answer.upper() or "<!ENTITY" in answer.upper() or "<?" in answer or "<!--" in answer:
        raise ValueError("XML declarations, entities and comments are not allowed")
    try:
        root = ET.fromstring(answer)
    except ET.ParseError as exc:
        raise ValueError(f"invalid XML (repair disabled): {exc}") from exc
    if root.tag != "decision" or root.attrib:
        raise ValueError("expected one unadorned decision")
    children = list(root)
    if len(children) != 4 or {c.tag for c in children} != {"kind", "name", "reasoning", "parameters"}:
        raise ValueError("decision requires unique kind, name, reasoning and parameters")
    for node in root.iter():
        if node.tail and node.tail.strip():
            raise ValueError("unexpected mixed XML text")
        if list(node) and node.text and node.text.strip():
            raise ValueError("unexpected mixed XML text")
        if set(node.attrib) - {"type"}:
            raise ValueError("unknown XML attribute")
    for tag in ("kind", "name", "reasoning"):
        node = root.find(tag)
        if list(node) or node.attrib:
            raise ValueError(f"{tag} must be plain text or CDATA")
    parameters = root.find("parameters")
    if parameters.attrib or (parameters.text or "").strip():
        raise ValueError("parameters must contain XML fields")
    tags = [c.tag for c in parameters]
    if len(tags) != len(set(tags)):
        raise ValueError("duplicate parameter")
    for child in parameters:
        if child.tag == "action":
            if child.attrib != {"type": "array"} or not list(child):
                raise ValueError("action must use native XML array items")
            for item in child:
                if item.tag != "item" or item.attrib != {"type": "number"} or list(item):
                    raise ValueError("action items must be numbers")
        elif child.tag == "num_steps":
            if child.attrib != {"type": "integer"} or list(child) or not re.fullmatch(r"[+-]?\d+", (child.text or "").strip()):
                raise ValueError("num_steps must be an XML integer")
        else:
            raise ValueError(f"forbidden parameter: {child.tag}")
    decision, errors = parse_xml_decision(answer)
    if errors or decision.get("_xml_wire_repair"):
        raise ValueError("; ".join(errors) or "XML repair is forbidden")
    if (decision["kind"], decision["name"]) not in {
        ("tool_call", "observe_env"), ("tool_call", "step_env"), ("response", "task_complete")
    }:
        raise ValueError("only observe_env, step_env and response task_complete are allowed")
    decision["parameters"] = validate_parameters(decision["name"], decision["parameters"],
        action_low=action_low, action_high=action_high, remaining_control_steps=remaining_control_steps)
    return {"decision": decision, "thinking": thinking, "answer": answer,
            "protocol_version": PROTOCOL_VERSION, "valid": True, "repaired": False}


def serialize_decision(name: str, parameters: dict | None = None, reasoning: str = "") -> str:
    if name not in ("observe_env", "step_env", "task_complete"):
        raise ValueError("forbidden decision name")
    root = ET.Element("decision")
    ET.SubElement(root, "kind").text = "response" if name == "task_complete" else "tool_call"
    ET.SubElement(root, "name").text = name
    ET.SubElement(root, "reasoning").text = reasoning
    parent = ET.SubElement(root, "parameters")
    for key, value in (parameters or {}).items():
        if key == "action":
            array = ET.SubElement(parent, key, type="array")
            for x in value:
                if type(x) not in (int, float) or not math.isfinite(x):
                    raise ValueError("action items must be finite numbers")
                ET.SubElement(array, "item", type="number").text = str(x)
        elif key == "num_steps" and type(value) is int:
            ET.SubElement(parent, key, type="integer").text = str(value)
        else:
            raise ValueError("invalid parameter")
    return ET.tostring(root, encoding="unicode")


def protocol_prompt(action_dim: int | dict) -> str:
    contract_text = ""
    if isinstance(action_dim, dict):
        contract = action_contract(action_dim)
        contract_text = f"Native action contract: {contract}. "
        action_dim = contract["action_dim"]
    example = serialize_decision("step_env", {"action": [0] * action_dim, "num_steps": 1})
    return (
        contract_text + f"Model protocol: {PROTOCOL_VERSION}. Return exactly one complete XML <decision>. "
        "Only observe_env and step_env are tools. task_complete is a response, not proof of success. "
        "Use unique kind/name/reasoning/parameters fields. reasoning may be empty; escape XML special characters. "
        "If thinking is enabled, close </think> before the decision; keep thinking concise. "
        "Do not output JSON, API tool_calls, markdown fences, multiple decisions or host parameters. "
        "action uses type=array with item type=number; num_steps is required, type=integer, 1–10. "
        "Use the supplied native action bounds and remaining control budget. No actions are corrected. "
        "Invalid output consumes a turn without movement; three consecutive invalid turns end the attempt. "
        "More than three consecutive observe calls without progress are invalid. "
        "Initial RGB and robot state are provided by the host. "
        + serialize_decision("observe_env") + "\n" + example + "\n" + serialize_decision("task_complete")
    )
