"""使用本地 Transformers 模型运行一轮 VLM -> OpenETA 仿真 rollout。

这是 Alice/Bob 集成的轻量级 smoke test。未安装 ManiSkill 时使用 OpenETA 的
dummy simulator；无论使用哪种后端，模型都读取模拟器产生的 RGB 图像，动作都
通过统一的 ``EnvAction``/``StepResult`` 协议执行。

默认 checkpoint 是 Qwen3.5-4B，因为当前工作区没有 Qwen3-4B checkpoint。
如果有其它本地模型，可以通过 ``--model`` 指定。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from adapter.protocol import EnvAction


DEFAULT_MODEL = "/inspire/hdd/global_public/public_models/Qwen/Qwen3.5-4B"
# 模型只能输出这些动作；未知输出会在 _parse_action 中降级为 DONE。
ALLOWED_ACTIONS = (
    "MOVE", "MOVE_X_POS", "MOVE_X_NEG", "MOVE_Y_POS", "MOVE_Y_NEG",
    "MOVE_Z_POS", "MOVE_Z_NEG", "MOVE_X_POS_FINE", "MOVE_X_NEG_FINE",
    "MOVE_Y_POS_FINE", "MOVE_Y_NEG_FINE", "MOVE_Z_POS_FINE", "MOVE_Z_NEG_FINE",
    "GRASP", "RELEASE", "DONE",
)


def _parse_action(text: str) -> tuple[str, str]:
    """从模型输出中提取受限动作，不直接信任模型生成的自由文本。"""

    raw = str(text or "").strip()
    direct = raw.upper()
    # 优先处理模型只输出一个动作代码的情况，例如 ``GRASP``。
    if direct in ALLOWED_ACTIONS:
        return direct, ""
    try:
        # 兼容模型输出 JSON 的情况，例如 {"action": "GRASP", "reason": "..."}。
        payload = json.loads(raw)
        action = str(payload.get("action", "")).upper()
        reason = str(payload.get("reason", "")).strip()
    except (json.JSONDecodeError, AttributeError):
        action = ""
        reason = "invalid action"
    # 安全起见，非法动作不传给模拟器，改成结束当前 episode。
    if action not in ALLOWED_ACTIONS:
        action = "DONE"
        reason = reason or "invalid action fallback"
    return action, reason


def _load_model(model_path: str):
    """加载 Qwen 处理器和模型，并切换到 CUDA 推理模式。"""
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
    ).to("cuda").eval()
    return processor, model


def _decide(processor: Any, model: Any, observation: Any, instruction: str) -> tuple[str, str, str]:
    """根据当前图像和任务文本，让 VLM 选择下一步离散动作。"""
    # 当前 observation 的第一路摄像头作为模型视觉输入。
    camera = observation.cameras[0]
    image = Image.fromarray(np.asarray(camera.rgb, dtype=np.uint8))
    # 注意：这里的 prompt 文本目前写的是 Bob；run() 会把同一个函数用于
    # Alice 和 Bob，因此这是一个简化 smoke path，不是完整的角色区分实现。
    prompt = (
        "You are the Bob embodied agent. Observe the image and follow the task.\n"
        f"Task: {instruction}\n"
        "Return only one action code with no JSON or explanation: "
        "MOVE|MOVE_X_POS|MOVE_X_NEG|MOVE_Y_POS|MOVE_Y_NEG|MOVE_Z_POS|MOVE_Z_NEG|"
        "MOVE_X_POS_FINE|MOVE_X_NEG_FINE|MOVE_Y_POS_FINE|MOVE_Y_NEG_FINE|"
        "MOVE_Z_POS_FINE|MOVE_Z_NEG_FINE|GRASP|RELEASE|DONE"
    )
    # Qwen 的多模态 chat 格式：一张图像 + 一段文字任务说明。
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    # 将结构化消息转换成模型需要的最终 prompt 字符串。
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # 同时处理文本和图像，并把 tensor 搬到 GPU。
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.cuda() if hasattr(value, "cuda") else value for key, value in inputs.items()}
    # smoke rollout 只推理，不计算梯度；这里使用贪心生成。
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    # 去掉输入 prompt 对应的 token，只保留模型新生成的部分。
    generated = output[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
    action, reason = _parse_action(raw)
    return action, reason, raw


def run(
    model_path: str,
    output_path: Path,
    instruction: str,
    *,
    backend: str = "dummy",
    env_id: str = "PickCube-v1",
    max_steps: int = 1,
) -> dict[str, Any]:
    """运行一轮简化的 Alice -> Bob rollout，并保存 JSON 结果。"""
    # 加载一次模型，随后 Alice 和 Bob 共用该模型实例。
    processor, model = _load_model(model_path)
    # 根据 backend 选择真实 ManiSkill 或轻量 dummy 环境。
    if backend == "maniskill":
        from adapter.maniskill_sim import ManiSkillSimulatorAdapter

        env = ManiSkillSimulatorAdapter(env_id=env_id, camera_resolution=128)
    elif backend == "dummy":
        from adapter.dummy_sim import DummySimulatorAdapter

        env = DummySimulatorAdapter()
    else:
        raise ValueError(f"unsupported backend: {backend}")

    # reset 返回第一帧 observation；任务文本本身仍由模型 prompt 使用。
    observation = env.reset(task=instruction, seed=0)
    initial_observation = observation
    # 只有 ManiSkill 后端支持完整 simulator snapshot，用于 Bob 严格回到 Alice 起点。
    snapshot = env.capture_snapshot() if backend == "maniskill" else None

    # ------------------------- Alice 阶段 -------------------------
    # Alice 连续执行最多 max_steps 个动作，并记录每一步的模型输出和环境回执。
    alice_steps: list[dict[str, Any]] = []
    for _ in range(max(1, int(max_steps))):
        action, reason, raw = _decide(processor, model, observation, instruction)
        # EnvAction 是 OpenETA 的统一动作协议；环境适配器再负责转换成原生动作。
        result = env.step(EnvAction(action_type=action, code=action))
        alice_steps.append({
            "action": action,
            "reason": reason,
            "raw": raw,
            "result": result.to_mcp_dict(),
        })
        if result.terminated or result.truncated or action == "DONE":
            break
        observation = result.observation

    # ------------------------- Bob 阶段 -------------------------
    bob_steps: list[dict[str, Any]] = []
    replay_equal = None
    if snapshot is not None:
        # 恢复 Alice 之前的完整状态，而不是只用相同 seed 重置。
        env.restore_snapshot(snapshot)
        replay_observation = env.observe()
        # 这里只比较关节位置，作为简化的恢复一致性检查。
        replay_equal = (
            replay_observation.robot.joint_positions == initial_observation.robot.joint_positions
        )
        # 当前 smoke path 只运行一次 Bob 动作；完整 self-play 会在同一 snapshot
        # 上采样多条 Bob trajectory，再计算 success rate。
        bob_action, bob_reason, bob_raw = _decide(processor, model, replay_observation, instruction)
        bob_result = env.step(EnvAction(action_type=bob_action, code=bob_action))
        bob_steps.append({
            "action": bob_action,
            "reason": bob_reason,
            "raw": bob_raw,
            "result": bob_result.to_mcp_dict(),
        })
    # 汇总为一个 JSON-friendly payload，便于后续评估或人工查看。
    payload = {
        "model": model_path,
        "backend": f"openeta/{backend}_{env_id}" if backend == "maniskill" else "openeta.dummy_sim-v0",
        "instruction": instruction,
        "observation": initial_observation.to_mcp_dict(),
        "snapshot": snapshot.to_dict() if snapshot is not None else None,
        "snapshot_replay_equal": replay_equal,
        "alice_steps": alice_steps,
        "bob_steps": bob_steps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 输出目录不存在时自动创建；ensure_ascii=False 保留中文文本。
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    """解析命令行参数，运行 rollout，并打印摘要。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("runs/qwen_embodied_rollout.json"))
    # 这是本脚本的默认任务文本；可通过 --instruction 覆盖。
    parser.add_argument("--instruction", default="Move the dummy robot toward the cube.")
    parser.add_argument("--backend", choices=("dummy", "maniskill"), default="dummy")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()
    payload = run(
        args.model,
        args.output,
        args.instruction,
        backend=args.backend,
        env_id=args.env_id,
        max_steps=args.max_steps,
    )
    # 终端只打印最简摘要，完整轨迹保存在 --output 指定的 JSON 文件中。
    alice_action = payload["alice_steps"][0]["action"] if payload["alice_steps"] else None
    bob_action = payload["bob_steps"][0]["action"] if payload["bob_steps"] else None
    print(json.dumps({
        "model": payload["model"],
        "backend": payload["backend"],
        "alice_action": alice_action,
        "bob_action": bob_action,
        "snapshot_replay_equal": payload["snapshot_replay_equal"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
