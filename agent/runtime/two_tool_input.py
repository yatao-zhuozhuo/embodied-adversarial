"""Versioned, provider-independent inputs for real two-tool trajectories."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from PIL import Image

from agent.runtime.two_tool_xml import protocol_prompt, split_answer

INPUT_PROTOCOL = "openeta.two_tool_input.v1"
ADAPTER_VERSION = "openeta.two_tool_input_messages.v1"
CAMERAS = {
    "metaworld": (["render"], []),
    "libero": (["agentview"], ["wrist"]),
    "robocasa": (["agentview_left", "agentview_right"], ["wrist"]),
}
MANISKILL_WRIST = {"PullCube-v1": False, "PokeCube-v1": False, "LiftPegUpright-v1": False,
                   "AssemblingKits-v1": True, "TurnFaucet-v1": True,
                   "FMBAssembly1Easy-v1": False, "PickSingleYCB-v1": True, "PickCube-v1": False}
MANISKILL_WRIST.update({'DrawTriangle-v1': False, 'DrawSVG-v1': False,
                       'RotateSingleObjectInHandLevel3-v1': False, 'InsertFlower-v1': False})
MANISKILL_WRIST.update({'PegInsertionSide-v1': True, 'StackCube-v1': True,
                       'StackPyramid-v1': True, 'PlaceSphere-v1': False,
                       'PullCubeTool-v1': False, 'PushCube-v1': False, 'RollBall-v1': False})
MANISKILL_CONTROL_MODES = {'RotateSingleObjectInHandLevel3-v1': 'pd_joint_pos',
                          'InsertFlower-v1': 'pd_joint_pos'}
TASK_TEXT = {
    "AssemblingKits-v1": "Insert the misplaced shape on the board completely into its matching empty slot.",
    "TurnFaucet-v1": "Turn the faucet handle from its initial position to the fully open end of its range.",
    "FMBAssembly1Easy-v1": "Assemble the parts onto the red board using the reorientation fixture.",
    "PickSingleYCB-v1": "Pick up the object on the table and move it to the task's goal position, then keep the robot still.",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def robot_projection(observation):
    raw = observation.get("robot") or {}
    pose = raw.get("end_effector_pose") or {}
    gripper = raw.get("gripper_state") or {}
    base = raw.get("base_pose") or {}
    return {"joint_positions": raw.get("joint_positions"), "joint_velocities": raw.get("joint_velocities"),
            "tcp": {key: pose.get(key) for key in ["xyz", "quat_xyzw"]},
            "measured_gripper": {key: gripper.get(key) for key in ["openness", "open"]},
            "base_pose": {key: base.get(key) for key in ["xyz", "quat_xyzw"]}}


def make_profile(bench, task_id, created, contract):
    native = created.get("input_control_contract") or {}
    if not native or native.get("error"):
        raise ValueError(f"Controller preflight failed: {native.get('error', 'missing live contract')}")
    if len(native.get("slots", [])) != contract["action_dim"]:
        raise ValueError("Live controller slots disagree with action dimension")
    if bench == "maniskill":
        if task_id not in MANISKILL_WRIST:
            raise ValueError("ManiSkill task needs an audited camera subprofile")
        primary, wrist = ["base_camera"], ["hand_camera"] if MANISKILL_WRIST[task_id] else []
        if native.get("mode") != MANISKILL_CONTROL_MODES.get(task_id, 'pd_ee_delta_pose'):
            raise ValueError("ManiSkill controller mode changed")
    else:
        primary, wrist = CAMERAS[bench]
    profile = {"id": f"{bench}.{task_id}.rgb256.v1", "version": 1,
               "primary": primary, "wrist": wrist, "image_limit": 2 * len(primary) + len(wrist),
               "history_groups": 4, "native_control": native, "action_contract": contract,
               "camera_dimensions": {camera: [128, 128] if camera == "hand_camera" else [256, 256]
                                     for camera in primary + wrist},
               "metaworld_view_alias": bench == "metaworld",
               "image_transform": "native RGB; no crop, resize, or camera substitution"}
    return {**profile, "hash": digest(profile)}


def task_description(task_id, task):
    if task_id in TASK_TEXT:
        return TASK_TEXT[task_id]
    if not task or task.startswith("ManiSkill task:"):
        raise ValueError("Task requires an audited natural-language goal")
    return task


def snapshot(observation, frames, directory, profile, observation_id, after_decision):
    paths = {Path(relative).stem.split("-", 1)[1]: str((directory / relative).resolve()) for relative in frames}
    cameras = {camera["frame_id"]: camera for camera in observation.get("cameras", [])}
    selected = []
    for name in profile["primary"] + profile["wrist"]:
        source = name
        if name == "render" and name not in paths and profile["metaworld_view_alias"]:
            source = "view"
        if source not in paths or source not in cameras:
            raise ValueError(f"Required camera missing: {name}")
        path = paths[source]
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if [width, height] != profile["camera_dimensions"][name]:
                raise ValueError(f"Camera resolution differs from profile: {name}: {width}x{height}")
            pixels = image.convert("RGB").tobytes()
        camera = cameras[source]
        selected.append({"camera_id": name, "source_camera_id": source,
                         "camera_role": "primary" if name in profile["primary"] else "wrist",
                         "observation_id": observation_id, "after_decision": after_decision,
                         "path": path, "width": width, "height": height,
                         "pixel_sha256": hashlib.sha256(pixels).hexdigest(),
                         "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         "intrinsics": {key: camera.get("intrinsics", {}).get(key) for key in ["fx", "fy", "cx", "cy", "width", "height"]},
                         "extrinsics": {key: camera.get("extrinsics", {}).get(key) for key in ["pos", "mat", "matrix_layout", "frame_transform", "camera_frame"]}})
    return {"observation_id": observation_id, "after_decision": after_decision,
            "robot": robot_projection(observation), "cameras": selected,
            "camera_availability": {name: name in cameras for name in profile["primary"] + profile["wrist"]}}


def robot_delta(current, previous):
    result = {"tcp_displacement_m": None, "tcp_distance_m": None, "gripper_openness_change": None,
              "frame": "world", "orientation_delta": None, "orientation_reason": "not computed"}
    if previous is None:
        return {**result, "reason": "initial observation; no previous decision"}
    before, after = previous["robot"], current["robot"]
    first, second = before["tcp"].get("xyz"), after["tcp"].get("xyz")
    if isinstance(first, list) and isinstance(second, list) and len(first) == len(second) == 3:
        delta = [end - start for start, end in zip(first, second)]
        if all(math.isfinite(value) for value in delta):
            result.update(tcp_displacement_m=delta, tcp_distance_m=math.sqrt(sum(value * value for value in delta)))
    start, end = before["measured_gripper"].get("openness"), after["measured_gripper"].get("openness")
    if isinstance(start, (float, int)) and isinstance(end, (float, int)):
        result["gripper_openness_change"] = end - start
    result["reason"] = None if result["tcp_displacement_m"] is not None else "TCP measurement unavailable"
    return result


def execution_feedback(event):
    if event is None:
        return {"execution": "initial", "last_action": None, "result_known": True,
                "terminated": False, "truncated": False}
    executed = event.get("valid") is True
    uncertain = bool(event.get("tool_request")) and not executed
    parameters = copy.deepcopy(event.get("arguments")) if event.get("validation", {}).get("valid") else None
    return {"execution": "executed" if executed else "error" if uncertain else "rejected",
            "result_known": not uncertain,
            "last_action": {"decision_id": event["turn"], "tool": event.get("tool"), "parameters": parameters,
                            "requested_steps": (parameters or {}).get("num_steps", 0),
                            "executed_steps": event.get("executed_control_steps", None if uncertain else 0)},
            "error": event.get("error"), "terminated": bool(event.get("terminated")),
            "truncated": bool(event.get("truncated"))}


def build_input(*, task, profile, current, previous, history, turn, max_turns, remaining_steps,
                role="task solver", last_event=None):
    selected = history[-4:]
    feedback = execution_feedback(last_event if last_event is not None else history[-1] if history else None)
    candidates = []
    if previous is not None:
        candidates.extend((camera, "previous_decision_before") for camera in previous["cameras"]
                          if camera["camera_role"] == "primary")
    candidates.extend((camera, "current") for camera in current["cameras"])
    images, by_pixels = [], {}
    for camera, timing in candidates:
        reference = {key: camera[key] for key in ["camera_id", "camera_role", "observation_id", "after_decision"]}
        reference.update(turn=turn if timing == "current" else turn - 1, timing=timing)
        key = (camera["width"], camera["height"], camera["pixel_sha256"])
        if key in by_pixels:
            by_pixels[key]["references"].append(reference)
        else:
            item = {**camera, "image_id": f"image-{len(images)}", "references": [reference]}
            images.append(item)
            by_pixels[key] = item
    if len(images) > profile["image_limit"]:
        raise ValueError("Camera profile image budget exceeded")
    dynamic = {"turn": turn, "observation_id": current["observation_id"],
               "budget": {"remaining_decisions": max_turns - turn + 1, "remaining_control_steps": remaining_steps},
               **feedback, "robot": current["robot"], "robot_delta": robot_delta(current, previous),
               "observation": {"is_new": previous is None or current["observation_id"] != previous["observation_id"],
                               "camera_availability": current["camera_availability"],
                               "images": [{key: value for key, value in image.items() if key != "path"} for image in images]}}
    system = ("You control a robot through OpenETA. Role: " + role + ". Task: " + task + "\n"
              "Only observe_env and step_env are available. The initial observation is already supplied. "
              "observe_env only reads the current state: it does not move cameras, increase resolution or reveal new views. "
              "Commands and measured gripper state are separate. Executed does not mean task success. "
              "Use measured feedback; do not invent collisions, object positions or success. "
              "Historical images are labelled; identical images retain all time references. "
              "Control contract: " + json.dumps(profile["native_control"], ensure_ascii=False) + "\n" +
              protocol_prompt(profile["action_contract"]))
    messages = [{"role": "system", "content": system}]
    if selected:
        messages.append({"role": "user", "content": "Recent decision history follows. No images are attached to these historical messages."})
    for index, event in enumerate(selected):
        try:
            _, answer = split_answer(event["model_message"].get("content") or "")
        except ValueError:
            answer = "[invalid answer: incomplete thinking block]"
        messages.append({"role": "assistant", "content": answer})
        if index < len(selected) - 1:
            messages.append({"role": "user", "content": json.dumps(execution_feedback(event), ensure_ascii=False)})
    content = [{"type": "text", "text": json.dumps(dynamic, ensure_ascii=False)}]
    for item in images:
        label = {"image_id": item["image_id"], "references": item["references"],
                 "same_image": len(item["references"]) > 1}
        content.extend([{"type": "text", "text": "Image evidence: " + json.dumps(label)},
                        {"type": "image_url", "image_url": {"url": item["path"]}}])
    messages.append({"role": "user", "content": content})
    return {"input_protocol": INPUT_PROTOCOL, "adapter_version": ADAPTER_VERSION,
            "profile_id": profile["id"], "profile_hash": profile["hash"], "dynamic": dynamic,
            "images": images, "history_decision_ids": [event["turn"] for event in selected],
            "trimming": {"dropped_history_groups": max(0, len(history) - 4), "dropped_images": [],
                         "reason": "four-group window", "preflight_token_count": None},
            "messages": messages, "messages_sha256": digest(messages)}


def build_bounded_input(*, task, role, profile, current, previous, history, turn,
                        max_turns, remaining_steps, generation_budget, context_limit,
                        render_request, encode_request):
    """Shared deterministic clipping using the caller's actual multimodal encoder."""
    selected = copy.deepcopy(history[-4:])
    prior = previous
    clipping = []
    while True:
        package = build_input(task=task, role=role, profile=profile, current=current,
            previous=prior, history=selected, turn=turn, max_turns=max_turns,
            remaining_steps=remaining_steps, last_event=history[-1] if history else None)
        request = render_request(package)
        ids = list(encode_request(copy.deepcopy(request)))
        if len(ids) + generation_budget <= context_limit:
            package['trimming'].update(preflight_token_count=len(ids), budget_clipping=clipping,
                                      dropped_history_groups=len(history)-len(selected))
            return package, request, ids
        if selected:
            clipping.append({'decision':selected.pop(0)['turn'], 'reason':'context_budget'})
        elif prior is not None and prior['cameras']:
            prior = copy.deepcopy(prior)
            prior['cameras'] = []
            clipping.append({'images':'historical', 'reason':'context_budget'})
        else:
            raise ValueError('minimal input exceeds context budget')
