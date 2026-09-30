"""Role-correct prompts for embodied Alice/Bob self-play.

Alice creates a new goal state.  Bob independently reaches that state.  The
functions here intentionally do not depend on Transformers, vLLM, or Swift so
both rollout backends can share and test the same model-visible projection.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from agent.training.embodied_grpo import ALLOWED_ACTIONS

ALICE_TASK = (
    "Create a difficult but solvable task by using real actions to move the red cube "
    "to a new reachable position. You are creating the goal state, not solving an "
    "existing goal. Finish with DONE only after the new state is established."
)

BOB_TASK = (
    "Independently move the red cube to the visible green target created from Alice's "
    "final state. You do not have Alice's trajectory and may use any legal action sequence."
)

WORKSPACE_BOUNDS = {
    "x": [-0.35, 0.35],
    "y": [-0.35, 0.35],
    "z": [0.015, 0.45],
}


def _positions(observation: Any) -> dict[str, list[float]]:
    return {
        str(item.get("name")): list(map(float, item["position"][:3]))
        for item in observation.objects
        if isinstance(item, dict)
        and isinstance(item.get("position"), list)
        and len(item["position"]) >= 3
    }


def _delta(target: Iterable[float] | None, source: Iterable[float] | None) -> list[float]:
    if target is None or source is None:
        return []
    target_values = list(target)
    source_values = list(source)
    if len(target_values) < 3 or len(source_values) < 3:
        return []
    return [round(float(target_values[i]) - float(source_values[i]), 4) for i in range(3)]


def _distance(left: Iterable[float] | None, right: Iterable[float] | None) -> float:
    delta = _delta(left, right)
    return math.sqrt(sum(value * value for value in delta)) if delta else 0.0


def build_embodied_prompt(
    role: str,
    observation: Any,
    *,
    step_index: int,
    max_steps: int,
    initial_cube_position: list[float],
    instruction: str | None = None,
    enable_thinking: bool = False,
) -> str:
    """Build one role-correct action prompt from the current trusted observation."""

    role = role.lower().strip()
    if role not in {"alice", "bob"}:
        raise ValueError(f"unsupported embodied role: {role!r}")

    positions = _positions(observation)
    cube = positions.get("cube")
    goal = positions.get("goal")
    tcp = list(map(float, observation.robot.end_effector_pose.get("xyz", [])))
    cube_from_tcp = _delta(cube, tcp)
    is_grasped = bool(observation.metadata.get("is_grasped", False))
    remaining_steps = max(0, int(max_steps) - int(step_index) - 1)
    actions = ", ".join(ALLOWED_ACTIONS)
    if enable_thinking:
        output_contract = (
            "Reason inside <think>...</think>. Check the current phase, the sign of the relevant "
            "coordinate error, and whether a coarse or fine action is appropriate. After "
            "</think>, output exactly one legal action code on the final line, with no other text."
        )
        short_output_contract = (
            "Reason inside <think>...</think>, then output exactly one legal action code "
            "on the final line."
        )
    else:
        output_contract = (
            "Return only the action code, with no JSON, markdown, reasoning, or explanation."
        )
        short_output_contract = "Output exactly one legal action code."

    if role == "alice":
        task = instruction or ALICE_TASK
        cube_displacement = _distance(cube, initial_cube_position)
        role_text = (
            "You are Alice, the task proposer. Create a new physical goal for Bob by "
            "actually moving the cube. Do not solve or follow any pre-existing environment goal."
        )
        state_text = (
            f"Step: {step_index}/{max_steps}; remaining_steps: {remaining_steps}; "
            f"end_effector_xyz: {tcp}; cube_position: {cube}; "
            f"initial_cube_position: {initial_cube_position}; "
            f"cube_minus_tcp_xyz: {cube_from_tcp}; "
            f"cube_displacement_m: {round(cube_displacement, 4)}; "
            f"is_grasped: {is_grasped}."
        )
        control = (
            "Before grasping, align the TCP with the cube in x, then y, then z, and GRASP. "
            "For each axis, use the coarse MOVE action while the absolute error is at least "
            "0.025 m; use *_FINE only below 0.025 m. "
            "After grasping, lift the cube and deliberately choose a new collision-free position "
            "inside workspace x/y [-0.35, 0.35], z [0.015, 0.45]. Move it at least 0.04 m "
            "from its initial position. Prefer a novel position that is hard but still visually "
            "observable and physically reachable by Bob. RELEASE is optional; use DONE only when "
            "the cube is stable at the intended new position. The old green marker is irrelevant."
        )
    else:
        task = instruction or BOB_TASK
        goal_from_cube = _delta(goal, cube)
        role_text = (
            "You are Bob, the task solver. Independently reproduce Alice's final cube state. "
            "You cannot see Alice's actions and do not need to imitate her trajectory."
        )
        state_text = (
            f"Step: {step_index}/{max_steps}; remaining_steps: {remaining_steps}; "
            f"end_effector_xyz: {tcp}; cube_position: {cube}; goal_position: {goal}; "
            f"cube_minus_tcp_xyz: {cube_from_tcp}; goal_minus_cube_xyz: {goal_from_cube}; "
            f"is_grasped: {is_grasped}."
        )
        control = (
            "Before grasping, align the TCP with the cube in x, then y, then z, and GRASP. "
            "For each axis, use the coarse MOVE action while the absolute error is at least "
            "0.025 m; use *_FINE only below 0.025 m. "
            "After grasping, lift for safe transport, align the cube with the green goal in x/y, "
            "then align z. Use *_FINE when the remaining axis error is below 0.025 m. "
            "Return DONE when the cube has reached the target state."
        )

    # The complete role/task/control contract is already present in turn 0 and
    # remains in the chat history.  Repeating it on every step made a 24-action
    # trajectory roughly 15k tokens long and needlessly exhausted 4090 memory
    # during GRPO backpropagation.  Later turns carry only the trusted state
    # delta and a short reminder; no task information is removed.
    if step_index > 0:
        rounded_tcp = [round(value, 3) for value in tcp]
        rounded_cube = [round(value, 3) for value in cube] if cube else []
        rounded_delta = [round(value, 3) for value in cube_from_tcp]
        if role == "alice":
            return (
                f"Step {step_index}/{max_steps}; remaining {remaining_steps}; "
                f"tcp={rounded_tcp}; cube={rounded_cube}; cube-tcp={rounded_delta}; "
                f"displacement={cube_displacement:.3f}m; grasped={is_grasped}. "
                "Continue creating a new reachable cube state. Use coarse MOVE at axis error "
                f">=0.025m, *_FINE below it. {short_output_contract}"
            )
        rounded_goal = [round(value, 3) for value in goal] if goal else []
        rounded_goal_delta = [round(value, 3) for value in goal_from_cube]
        return (
            f"Step {step_index}/{max_steps}; remaining {remaining_steps}; "
            f"tcp={rounded_tcp}; cube={rounded_cube}; goal={rounded_goal}; "
            f"cube-tcp={rounded_delta}; goal-cube={rounded_goal_delta}; grasped={is_grasped}. "
            "Continue solving Alice's goal. Use coarse MOVE at axis error >=0.025m, *_FINE "
            f"below it. {short_output_contract}"
        )

    return (
        f"{role_text}\n"
        f"Task: {task}\n"
        f"{state_text}\n"
        f"Control guidance: {control}\n"
        f"Choose exactly one action from: {actions}.\n"
        f"{output_contract}"
    )
