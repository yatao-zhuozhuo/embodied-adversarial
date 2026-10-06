"""Live, whitelisted native-controller metadata for model input profiles."""

import hashlib
import inspect
import importlib.metadata
from pathlib import Path


def values(value):
    return value.tolist() if hasattr(value, "tolist") else value


def controller_contract(env, backend):
    direct = env._unwrap()
    raw = getattr(direct, "unwrapped", direct)
    result = {"version": "openeta.input_control.v1",
              "normalization": "bounded normalized commands; scales are commands, not measured displacement",
              "tcp": {"frame": "world", "position_unit": "metre", "quaternion_order": "xyzw"},
              "repeat": "Repeat the same command for 1–10 control steps; actual steps may be fewer"}
    controller = raw
    if backend == "metaworld":
        result.update(controller="mocap_xyz", mode="delta_position", frame="world",
                      slots=["dx", "dy", "dz", "gripper"], position_scale_m=float(raw.action_scale),
                      rotation="not controlled", gripper={"open_value": -1, "close_value": 1})
    elif backend == "maniskill":
        agent = raw.agent
        agents = getattr(agent, "agents_dict", {"robot": agent})
        # Flatten in the same ordered keys as ManiSkill's FlattenActionSpaceWrapper.
        agents = {key: agents[key] for key in raw.single_action_space.spaces} if isinstance(agent.controller, dict) else agents
        blocks, slots, neutral = [], [], []
        mode = None
        for agent_name, robot in agents.items():
            if mode is not None and mode != robot.control_mode:
                raise ValueError("Mixed robot controller modes require a separate profile")
            mode = robot.control_mode
            combined = robot.controller
            controllers = getattr(combined, "controllers", {"joint": combined})
            for part, controller in controllers.items():
                config = controller.config
                prefix = agent_name + "." if len(agents) > 1 else ""
                offset = len(slots)
                if hasattr(config, "pos_lower"):
                    names = ["dx", "dy", "dz", "rx", "ry", "rz"]
                    block = dict(controller=type(controller).__name__, frame=config.frame,
                        rotation="XYZ Euler delta; normalized rotation vector norm-clipped to 1, multiplied by rot_lower",
                        position_low_m=values(config.pos_lower), position_high_m=values(config.pos_upper),
                        rotation_multiplier_rad=values(config.rot_lower), use_delta=config.use_delta)
                elif part == "gripper":
                    names = ["gripper"]
                    block = dict(controller=type(controller).__name__, gripper={"open_value": 1, "close_value": -1,
                        "semantics": "finger joint position"})
                else:
                    names = ["delta_" + name for name in config.joint_names]
                    if not config.use_delta or not config.normalize_action:
                        raise ValueError("Unaudited ManiSkill joint mode")
                    block = dict(controller=type(controller).__name__, frame="robot joints",
                        joint_names=list(config.joint_names), joint_delta_low_rad=values(config.lower),
                        joint_delta_high_rad=values(config.upper), use_delta=True,
                        use_target=bool(config.use_target), semantics="normalized joint position delta in listed joint order")
                dimension = controller.single_action_space.shape[0]
                if len(names) != dimension:
                    raise ValueError("Controller metadata does not match action dimension")
                slots.extend(prefix + name for name in names)
                neutral.extend([1.] if part == "gripper" else [0.] * dimension)
                controller_source = Path(inspect.getfile(type(controller)))
                block['source'] = {'class': type(controller).__module__ + '.' + type(controller).__name__,
                                   'sha256': hashlib.sha256(controller_source.read_bytes()).hexdigest()}
                blocks.append(dict(agent=agent_name, part=part, offset=offset, length=dimension, **block))
        result.update(controller="ManiSkill native controller", mode=mode, slots=slots,
                      components=blocks, neutral_action=neutral)
        if len(agents) == 1 and len(blocks) == 2 and blocks[0]['part'] == 'arm' and blocks[1]['part'] == 'gripper':
            # Preserve the established Panda profile fields for existing consumers.
            result.update({k:v for k,v in blocks[0].items() if k not in ('agent','part','offset','length')})
            result['gripper'] = blocks[1]['gripper']
    elif backend == "libero":
        spec = direct.openeta_control_spec
        cartesian = spec["cartesian_delta"]
        result.update(controller=spec["controller"]["controller_id"], mode="OSC_POSE",
                      slots=["dx", "dy", "dz", "axis_angle_x", "axis_angle_y", "axis_angle_z", "gripper"],
                      frame=cartesian["command_frame"], rotation="axis-angle delta, not Euler angles",
                      position_scale_m=cartesian["position_scale_m"], rotation_scale_rad=cartesian["rotation_scale_rad"],
                      gripper={key: spec["gripper"][key] for key in ["open_value", "close_value"]})
    elif backend == "robocasa":
        controller = direct.unwrapped_env.robots[0].part_controllers["right"]
        result.update(controller=type(controller).__name__, mode=controller.input_type,
                      frame=controller.input_ref_frame, rotation="axis-angle delta, not Euler angles",
                      slots=["arm_dx", "arm_dy", "arm_dz", "axis_angle_x", "axis_angle_y", "axis_angle_z", "gripper", "base_forward", "base_left", "base_yaw", "torso", "mode"],
                      output_min=values(controller.output_min), output_max=values(controller.output_max),
                      gripper={"open_value": -1, "close_value": 1},
                      mobile={"frame": "base", "velocity_scale": "unknown", "torso_scale": "unknown",
                              "mode": "positive selects base, nonpositive selects arm; keep inactive motion slots zero"})
    else:
        raise ValueError(f"No input control profile for {backend}")
    source = Path(inspect.getfile(type(controller)))
    result["source"] = {"class": type(controller).__module__ + "." + type(controller).__name__,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    result["environment_versions"] = {}
    for package in ["metaworld", "mani_skill", "robosuite", "robocasa"]:
        try:
            result["environment_versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result
