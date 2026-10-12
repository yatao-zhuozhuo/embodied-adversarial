"""Generate native-seed Cartesian expert actions for subsequent OpenETA replay."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
import shutil
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ADAPTER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

SOLUTIONS = {
    "PickCube-v1": "pick_cube", "PickSingleYCB-v1": "pick_cube",
    "PlaceSphere-v1": "place_sphere", "PegInsertionSide-v1": "peg_insertion_side",
    "StackCube-v1": "stack_cube", "StackPyramid-v1": "stack_pyramid",
    "PullCubeTool-v1": "pull_cube_tool", "PushCube-v1": "push_cube",
    "DrawSVG-v1": "draw_svg", "DrawTriangle-v1": "draw_triangle",
    "FMBAssembly1Easy-v1": "bridge_assembly", "RollBall-v1": "roll_ball",
}


class Finished(Exception):
    pass


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def world_obb(actor, to_world_frame=True, vis=False):
    import sapien.physx as physx
    import trimesh
    from mani_skill.utils.geometry.trimesh_utils import get_component_mesh
    mesh = get_component_mesh(actor._objs[0].find_component_by_type(physx.PhysxRigidDynamicComponent),
                              to_world_frame=False)
    try:
        local = mesh.bounding_box_oriented
    except ValueError:
        local = mesh.bounding_box
    transform = np.array(local.primitive.transform, dtype=np.float64, copy=True)
    if to_world_frame:
        transform = array(actor.pose.to_transformation_matrix())[0] @ transform
    from scipy.spatial.transform import Rotation
    transform[:3, :3] = Rotation.from_matrix(transform[:3, :3]).as_matrix()
    return trimesh.primitives.Box(extents=np.array(local.primitive.extents, copy=True),
                                  transform=transform, mutable=True)


class CartesianExpert:
    OPEN = 1
    CLOSED = -1

    def __init__(self, env, **kwargs):
        self.env = env
        self.base_env = env.unwrapped
        self.gripper_state = self.OPEN
        self.motion_planner = None
        if env.motion_planning and env.task_id in {"FMBAssembly1Easy-v1", "PegInsertionSide-v1", "StackPyramid-v1"}:
            import mplib
            robot = self.base_env.agent.robot
            urdf = self.base_env.agent.urdf_path
            self.motion_planner = mplib.Planner(
                urdf=urdf, srdf=urdf.replace(".urdf", ".srdf"),
                user_link_names=[link.get_name() for link in robot.get_links()],
                user_joint_names=[joint.get_name() for joint in robot.get_active_joints()],
                move_group="panda_hand_tcp")
            base = robot.pose.sp
            self.motion_planner.set_base_pose(mplib.Pose(base.p, base.q))

    def hold(self, steps=12):
        action = np.zeros(self.env.dimension)
        if self.env.dimension == 7:
            action[-1] = self.gripper_state
        for _ in range(steps):
            self.env.step(action)

    def open_gripper(self, steps=12, gripper_state=None):
        self.gripper_state = self.OPEN if gripper_state is None else gripper_state
        self.hold(steps)

    def close_gripper(self, steps=12, gripper_state=None):
        self.gripper_state = self.CLOSED if gripper_state is None else gripper_state
        self.hold(steps)

    def move_to_pose_with_screw(self, pose, dry_run=False, refine_steps=0):
        import sapien
        from scipy.spatial.transform import Rotation
        if hasattr(pose, "sp"):
            pose = pose.sp
        if self.motion_planner is not None:
            import mplib
            target = mplib.Pose(pose.p, pose.q)
            current = array(self.base_env.agent.robot.get_qpos())[0]
            planned = self.motion_planner.plan_screw(target, current, time_step=self.base_env.control_timestep)
            if planned["status"] != "Success":
                planned = self.motion_planner.plan_pose(target, current, time_step=self.base_env.control_timestep)
            if planned["status"] != "Success":
                return -1
            if dry_run:
                return planned
            pin_model = self.motion_planner.pinocchio_model
            for joints in planned["position"]:
                full = array(self.base_env.agent.robot.get_qpos())[0].copy()
                full[:len(joints)] = joints
                pin_model.compute_forward_kinematics(full)
                local = pin_model.get_link_pose(self.motion_planner.move_group_link_id)
                target = self.base_env.agent.robot.pose.sp * sapien.Pose(local.p, local.q)
                self.track_pose(target, iterations=1)
            return self.track_pose(pose, iterations=40)
        if dry_run:
            return {"status": "Success"}
        return self.track_pose(pose, iterations=100, refine_steps=refine_steps)

    def track_pose(self, pose, iterations=100, refine_steps=0):
        import sapien
        from scipy.spatial.transform import Rotation
        arm = self.base_env.agent.controller.controllers["arm"]
        drawing = self.env.task_id.startswith("Draw")
        if drawing:
            position = pose.p.copy()
            position[2] = max(position[2], float(self.base_env.CANVAS_THICKNESS) + 0.002)
            pose = sapien.Pose(position, pose.q)
        target = self.base_env.agent.robot.pose.sp.inv() * pose
        for _ in range(iterations):
            current = arm.ee_pose_at_base.sp
            position = target.p - current.p
            rotation = Rotation.from_quat(target.q[[1, 2, 3, 0]]) * Rotation.from_quat(current.q[[1, 2, 3, 0]]).inv()
            if np.linalg.norm(position) < (0.008 if drawing else 0.002) and rotation.magnitude() < 0.025:
                break
            if drawing and np.linalg.norm(position) > 0.015:
                position *= 0.015 / np.linalg.norm(position)
            action = np.zeros(self.env.dimension)
            action[:3] = np.clip(position / np.asarray(arm.config.pos_upper), -1, 1)
            angular = rotation.as_euler("XYZ") / float(arm.config.rot_lower)
            action[3:6] = angular / max(1, np.linalg.norm(angular))
            if self.env.dimension == 7:
                action[-1] = self.gripper_state
            self.env.step(action)
        if refine_steps:
            self.hold(refine_steps)
        return self.env.last

    move_to_pose_with_RRTConnect = move_to_pose_with_screw
    move_to_pose_with_RRTStar = move_to_pose_with_screw

    def close(self):
        self.hold(15)


class RecordingEnv:
    def __init__(self, env, case, horizon):
        self.env = env
        self.unwrapped = env.unwrapped
        self.case = case
        self.task_id = case["task_id"]
        self.dimension = int(np.prod(env.action_space.shape))
        self.horizon = horizon
        self.actions = []
        self.success = False
        self.last = None
        self.initial_state = None
        self.initial_observation = None
        self.motion_planning = False

    def reset(self, seed=None):
        if seed != self.case["seed"] or self.initial_state is not None:
            raise ValueError("expert must reset exactly once at the frozen SFT seed")
        observation, info = self.env.reset(seed=seed)
        self.initial_state = self.unwrapped.get_state_dict()
        if bool(array(info.get("success", False)).reshape(-1)[0]):
            raise ValueError("initial state already successful")
        hold = np.zeros(self.dimension, dtype=np.float32)
        if self.dimension == 7:
            hold[-1] = 1
        for _ in range(5):
            observation, _, terminated, truncated, info = self.env.step(hold)
            if bool(array(terminated).reshape(-1)[0]) or bool(array(truncated).reshape(-1)[0]):
                raise ValueError("environment ended during reset settling")
        self.initial_observation = observation
        return observation, info

    def step(self, action):
        if len(self.actions) >= self.horizon:
            raise Finished("control_budget_exhausted")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.dimension,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1.000001):
            raise ValueError("invalid native expert action")
        self.last = self.env.step(action)
        self.actions.append(action.tolist())
        _, _, terminated, truncated, info = self.last
        self.success = bool(array(info.get("success", False)).reshape(-1)[0])
        if self.success or bool(array(terminated).reshape(-1)[0]) or bool(array(truncated).reshape(-1)[0]):
            raise Finished("official_success" if self.success else "environment_done_without_success")
        return self.last


def custom_solve(env, seed):
    import sapien
    env.reset(seed=seed)
    raw = env.unwrapped
    planner = CartesianExpert(env)
    if env.task_id == "DrawTriangle-v1":
        rotation = raw.agent.tcp.pose.sp.q
        vertices = array(raw.vertices)[0]
        for point in [*vertices, vertices[0]]:
            planner.move_to_pose_with_screw(sapien.Pose(point, rotation))
    elif env.task_id == "RollBall-v1":
        planner.close_gripper()
        ball = raw.ball.pose.sp.p
        goal = raw.goal_region.pose.sp.p
        direction = goal[:2] - ball[:2]
        direction /= np.linalg.norm(direction)
        behind = ball.copy()
        behind[:2] -= direction * (float(raw.ball_radius) + 0.03)
        rotation = raw.agent.tcp.pose.sp.q
        planner.move_to_pose_with_screw(sapien.Pose(behind + [0, 0, 0.1], rotation))
        planner.move_to_pose_with_screw(sapien.Pose(behind, rotation))
        target = behind.copy()
        target[:2] += direction * 0.25
        planner.move_to_pose_with_screw(sapien.Pose(target, rotation))
        planner.hold(100)
    elif env.task_id == "PickSingleYCB-v1":
        from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
        approaching = np.array([0, 0, -1])
        closing = array(raw.agent.tcp.pose.to_transformation_matrix())[0, :3, 1]
        grasp = compute_grasp_info_by_obb(world_obb(raw.obj), approaching, closing, depth=0.025)
        pose = raw.agent.build_grasp_pose(approaching, grasp["closing"], grasp["center"])
        planner.move_to_pose_with_screw(sapien.Pose(pose.p + [0, 0, 0.08], pose.q))
        planner.move_to_pose_with_screw(pose)
        planner.close_gripper()
        planner.move_to_pose_with_screw(sapien.Pose(pose.p + [0, 0, 0.12], pose.q))
        offset = raw.goal_site.pose.sp.p - raw.obj.pose.sp.p
        current = raw.agent.tcp.pose.sp
        planner.move_to_pose_with_screw(sapien.Pose(current.p + offset, current.q))
        planner.hold(25)
    else:
        from mani_skill.examples.motionplanning.base_motionplanner.utils import compute_grasp_info_by_obb, get_actor_obb
        approaching = np.array([0, 0, -1])
        closing = array(raw.agent.tcp.pose.to_transformation_matrix())[0, :3, 1]
        grasp = compute_grasp_info_by_obb(world_obb(raw.bridge), approaching, closing, depth=0.025)
        held = False
        candidates = [(offset, sign) for sign in [1, -1]
                      for offset in ([0, 0, 0.03], [0, 0, -0.03], [0.03, 0, 0], [-0.03, 0, 0], [0, 0, 0])]
        for offset, sign in candidates:
            planner.open_gripper()
            center = (raw.bridge.pose.sp * sapien.Pose(offset)).p
            pose = raw.agent.build_grasp_pose(approaching, grasp["closing"] * sign, center)
            predicted_relation = raw.bridge.pose.sp.inv() * pose
            if planner.move_to_pose_with_screw(raw.goal_bridge_pose.sp * predicted_relation, dry_run=True) == -1:
                continue
            planner.move_to_pose_with_screw(sapien.Pose(pose.p + [0, 0, 0.1], pose.q))
            planner.move_to_pose_with_screw(pose)
            planner.close_gripper(steps=20)
            held = bool(array(raw.agent.is_grasping(raw.bridge)).reshape(-1)[0])
            if held:
                actual_relation = raw.bridge.pose.sp.inv() * raw.agent.tcp.pose.sp
                if planner.move_to_pose_with_screw(raw.goal_bridge_pose.sp * actual_relation, dry_run=True) != -1:
                    break
                held = False
        if not held:
            raise Finished("bridge_grasp_failed_after_feedback_retries")
        planner.move_to_pose_with_screw(sapien.Pose(pose.p + [0, 0, 0.15], pose.q))
        object_to_tcp = raw.bridge.pose.sp.inv() * raw.agent.tcp.pose.sp
        target = raw.goal_bridge_pose.sp * object_to_tcp
        planner.move_to_pose_with_screw(sapien.Pose(target.p + [0, 0, 0.1], target.q))
        for _ in range(8):
            object_to_tcp = raw.bridge.pose.sp.inv() * raw.agent.tcp.pose.sp
            target = raw.goal_bridge_pose.sp * object_to_tcp
            planner.move_to_pose_with_screw(target)
            planner.hold(4)
        planner.open_gripper()
    planner.close()


def adapted_solve(module):
    tree = ast.parse(inspect.getsource(module.solve))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
                and ast.unparse(node.test.left) == "env.unwrapped.control_mode"):
            for comparator in node.test.comparators:
                if isinstance(comparator, ast.List):
                    comparator.elts.append(ast.Constant("pd_ee_delta_pose"))
    namespace = dict(vars(module))
    namespace["PandaArmMotionPlanningSolver"] = CartesianExpert
    namespace["PandaStickMotionPlanningSolver"] = CartesianExpert
    namespace["get_actor_obb"] = world_obb
    exec(compile(ast.fix_missing_locations(tree), module.__file__, "exec"), namespace)
    return namespace["solve"]


def main():
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-planning", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        archive = args.output.parent / (args.output.stem + "-attempts")
        archive.mkdir(exist_ok=True)
        shutil.copy2(args.output, archive / f"{time.time_ns()}.json")
    spec = json.loads(args.case_file.read_text())
    case, task = spec["case"], spec["task"]
    if case["phase"] != "sft_train" or case["initial_state"]["kind"] != "native_seed":
        raise ValueError("only frozen native-seed SFT cases are supported")
    from sim.env_registry import _make_maniskill_direct
    raw = _make_maniskill_direct(case["task_id"], image_width=256, image_height=256)
    env = RecordingEnv(raw, case, task["horizon"])
    env.motion_planning = args.motion_planning
    error = None
    try:
        if env.dimension != task["expected_action_dim"]:
            raise ValueError("native action dimension differs from frozen manifest")
        if raw.unwrapped.control_mode != task["environment_config"]["control_mode"]:
            raise ValueError("native controller differs from frozen manifest")
        if case["task_id"] in {"DrawTriangle-v1", "RollBall-v1", "FMBAssembly1Easy-v1", "PickSingleYCB-v1"}:
            custom_solve(env, case["seed"])
        else:
            module = importlib.import_module("mani_skill.examples.motionplanning.panda.solutions." + SOLUTIONS[case["task_id"]])
            adapted_solve(module)(env, seed=case["seed"], debug=False, vis=False)
    except Finished as finished:
        error = str(finished)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    finally:
        result = {"schema_version": "openeta.expert_action_source.v1", "bench": "maniskill",
                  "task_id": case["task_id"], "env_id": case["env_id"], "seed": case["seed"],
                  "source_kind": "adapted_cartesian_waypoint_expert_live_native_seed",
                  "source_dataset": "installed_mani_skill_waypoint_solutions",
                  "policy": SOLUTIONS[case["task_id"]], "source_success": env.success,
                  "source_steps": len(env.actions), "actions": env.actions,
                  "unified_case": case, "source_stop_reason": error or "expert_completed_without_success",
                  "start_protocol": "native_reset_plus_five_open_gripper_settle_steps",
                  "initial_state": env.initial_state}
        result["control_adapter_mode"] = "planned_joint_fk_to_cartesian" if env.motion_planning else "cartesian_waypoints"
        result["source_numpy_version"] = np.__version__
        result["terminal_state"] = raw.unwrapped.get_state_dict()
        result["terminal_task_poses"] = {
            name: actor.pose.raw_pose for name, actor in raw.unwrapped.scene.actors.items()
            if name not in {"table", "ground", "floor"} and not name.startswith("dot_")}
        result["adapter_sha256"] = ADAPTER_SHA256
        args.output.parent.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        frames = args.output.parent / (args.output.stem + f"-source-frames-{os.getpid()}")
        frames.mkdir(exist_ok=True)
        for label, observation in [("initial", env.initial_observation),
                                   ("terminal", env.last[0] if env.last else None)]:
            if isinstance(observation, dict):
                for camera, data in observation.get("sensor_data", {}).items():
                    if "rgb" in data:
                        Image.fromarray(array(data["rgb"])[0, ..., :3].astype(np.uint8)).save(frames / f"{label}-{camera}.png")
        result["source_frames"] = str(frames.resolve())
        temporary = args.output.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(result, default=lambda value: array(value).tolist(), ensure_ascii=False) + "\n")
        temporary.replace(args.output)
        raw.close()
        print(json.dumps({key: result[key] for key in ["task_id", "source_success", "source_steps", "source_stop_reason"]}), flush=True)


if __name__ == "__main__":
    main()
