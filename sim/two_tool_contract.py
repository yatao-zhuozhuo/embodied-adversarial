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
        aggregate = raw.agent.controller
        parts = getattr(aggregate, 'controllers', {'joints': aggregate})
        if raw.control_mode == 'pd_ee_delta_pose':
            controller = parts['arm']
            config = controller.config
            slots = ["dx", "dy", "dz", "rx", "ry", "rz"]
            if 'gripper' in parts:
                slots.append('gripper')
            result.update(controller=type(controller).__name__, mode=raw.control_mode,
                          slots=slots, frame=config.frame,
                          rotation="XYZ Euler delta; normalized rotation vector norm-clipped to 1, multiplied by rot_lower",
                          position_low_m=values(config.pos_lower), position_high_m=values(config.pos_upper),
                          rotation_multiplier_rad=values(config.rot_lower), use_delta=config.use_delta,
                          gripper={"open_value": 1, "close_value": -1, "semantics": "finger joint position"} if 'gripper' in parts else {"supported": False})
        elif raw.control_mode == 'pd_joint_pos':
            controller = aggregate
            slots, components = [], []
            for name, part in parts.items():
                space = getattr(part, 'single_action_space', None)
                if space is None:
                    space = part.action_space
                count = int(space.shape[-1])
                names = list(part.config.joint_names)
                labels = names if len(names) == count else [f'command_{i}' for i in range(count)]
                slots.extend(f'{name}:{label}' for label in labels)
                components.append({'name': name, 'joint_names': names,
                                   'normalized': bool(part.config.normalize_action),
                                   'use_delta': bool(part.config.use_delta),
                                   'action_low': values(space.low), 'action_high': values(space.high)})
            result.update(controller=type(controller).__name__, mode=raw.control_mode,
                          slots=slots, frame='native_joint_coordinates', components=components,
                          normalization='Native bounded joint-position targets; consult component normalization and action limits.',
                          rotation='Controlled by native joints, not Cartesian Euler or axis-angle commands',
                          gripper={'supported': False, 'semantics': 'Finger joints are commanded individually'},
                          tcp={'supported': False})
        else:
            raise ValueError(f'Unaudited ManiSkill control mode: {raw.control_mode}')
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
