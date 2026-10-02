"""ms-swift multi-turn schedulers for OpenETA embodied Alice/Bob GRPO.

The model emits exactly one bounded action per turn.  The host owns ManiSkill,
snapshot restore, goal checking, task compilation, rewards, and artifacts.  Bob
reaches Alice's final state independently; Alice's trajectory is never exposed
to Bob.
"""

# ============================ 中文总览 ============================
# 本文件是 ms-swift 框架的外部插件（external plugin），通过 --external_plugins 加载，
# 为具身 Alice/Bob 非对称 self-play 的 GRPO 训练提供两类注册组件：
#   1) multi-turn scheduler（注册名 embodied_alice_scheduler / embodied_bob_scheduler）：
#      在 rollout 阶段托管 ManiSkill 环境（PickCube-v1 机械臂抓方块），负责快照恢复、
#      逐轮执行模型输出的动作、目标判定、artifact 落盘，以及 Alice 出题后的"冻结 Bob"
#      嵌套评估（走外部 Bob server，或 staged colocate 模式下交给同 engine 的冻结 Bob LoRA）。
#   2) reward function（注册名 embodied_alice_reward / embodied_bob_reward）：
#      训练阶段从 scheduler 写入的 rollout_infos 中取出"宿主侧可信 reward"，喂给 GRPO。
# 数据流向：dataset 的 env_request（含快照/目标谓词等）→ on_trajectory_start 建环境
#   → 模型每轮输出一个动作 → on_turn_end 执行动作并追加观测图像/prompt → episode 结束
#   时 _finalize 计算 reward 并写 rollout_infos → reward function 读取该 reward → GRPO
#   按 num_generations 分组算组内相对优势更新策略。
# 此外文件头部还安装了两个显存优化 monkey-patch（分块 log-softmax、bf16 logits），
# 只对齐数值语义、不改变训练算法。
# ==================================================================

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import sys
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest, RolloutOutput
from swift.rewards import ORM, orms
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

from adapter.maniskill_sim import ManiSkillSimulatorAdapter
from adapter.maniskill_process import IsolatedManiSkillSimulatorAdapter
from adapter.protocol import EnvAction
from agent.runtime.embodied_goal import PositionGoalChecker
from agent.runtime.embodied_selfplay_prompt import ALICE_TASK, BOB_TASK, build_embodied_prompt
from agent.runtime.embodied_snapshot import SnapshotRef
from agent.runtime.embodied_task_compiler import CompiledTask, compile_cube_position_task
from agent.training.embodied_grpo import ALLOWED_ACTIONS, parse_action
from agent.training.embodied_rollout_budget import RolloutBudget
from agent.training.embodied_staged_contracts import (
    AliceProposal,
    BobEvalReceipt,
    boundary_score,
    build_bob_eval_requests,
    finalize_alice_reward,
)

# Put longer alternatives first so a regex backend never accepts MOVE as the
# prefix of MOVE_X_POS.  vLLM structured output applies full grammar matching.
# 中文说明：按动作名长度从长到短排列后再拼正则，保证 MOVE_X_POS 这类长动作
# 优先于 MOVE 被匹配，避免 vLLM 结构化输出把长动作错误截断成短前缀。
ACTION_REGEX = "(?:" + "|".join(
    re.escape(action) for action in sorted(ALLOWED_ACTIONS, key=len, reverse=True)
) + ")"
RECOVERABLE_ACTION_LINE = re.compile(
    rf"^(?:[-*]\s*)?(?:action\s*:\s*)?`?({ACTION_REGEX})`?[.!]?$",
    flags=re.IGNORECASE,
)
HIDDEN_ALICE_GOAL = [0.0, 0.0, -1.0]


def _chunked_selective_log_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """Compute selected log-probabilities without a sequence-sized temp.

    TRL's bf16 implementation applies ``log_softmax`` to every completion
    token at once.  With Qwen3.5's 248,320-token vocabulary, a long embodied
    trajectory can therefore allocate another 9+ GiB even though only one
    selected log-probability per token is retained.  Chunking over token rows
    is mathematically equivalent and preserves every completion token while
    bounding the temporary tensor.  This path is primarily used for the
    no-grad old/reference-policy passes; Liger still owns the differentiable
    policy-loss path.
    """

    # 中文说明：把 (token 数 × 词表) 的 log_softmax 按 token 行分块计算，
    # 数学上与一次性计算完全等价，但避免为整条长轨迹一次性分配
    # token数×248320 词表的巨型临时张量（可省 9+ GiB 显存）。
    squeeze = index.ndim == logits.ndim - 1
    if squeeze:
        index = index.unsqueeze(-1)

    vocab_size = logits.shape[-1]
    selected_per_row = index.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_index = index.reshape(-1, selected_per_row)
    chunk_size = int(os.environ.get("OPENETA_LOGPS_CHUNK_SIZE", "128"))
    if chunk_size < 1:
        raise ValueError("OPENETA_LOGPS_CHUNK_SIZE must be positive")

    chunks: list[torch.Tensor] = []
    for start in range(0, flat_logits.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_logits.shape[0])
        chunk_logits = flat_logits[start:stop]
        chunk_index = flat_index[start:stop]
        if chunk_logits.dtype in (torch.float32, torch.float64):
            selected = torch.gather(chunk_logits, dim=-1, index=chunk_index)
            normalizer = torch.logsumexp(chunk_logits, dim=-1, keepdim=True)
            chunks.append(selected - normalizer)
        else:
            chunk_logps = F.log_softmax(chunk_logits, dim=-1)
            chunks.append(torch.gather(chunk_logps, dim=-1, index=chunk_index))

    result = torch.cat(chunks, dim=0).reshape(index.shape)
    if squeeze:
        result = result.squeeze(-1)
    return result


def _install_chunked_logps_patch() -> None:
    """Patch both TRL and ms-swift's imported alias in this process."""

    # 中文说明：TRL 与 ms-swift 各自持有 selective_log_softmax 的引用，
    # 两处都必须替换，否则 ms-swift 模块内早已导入的别名仍指向旧实现。
    if not _env_bool("OPENETA_CHUNK_LOGPS", True):
        return
    from trl.trainer import utils as trl_utils

    trl_utils.selective_log_softmax = _chunked_selective_log_softmax
    # ms-swift imports the function into its module namespace before loading
    # external plugins, so replacing only trl_utils is insufficient.
    from swift.rlhf_trainers import grpo_trainer as swift_grpo_trainer

    swift_grpo_trainer.selective_log_softmax = _chunked_selective_log_softmax


def _install_bf16_logit_forward_patch() -> None:
    """Keep no-grad old/reference-policy logits in their autocast dtype.

    Accelerate wraps a bf16 model forward with ``convert_outputs_to_fp32``.
    That is generally convenient for small outputs, but GRPO requests a logit
    row for every completion token.  A 23k-token Qwen3.5 trajectory therefore
    creates an additional ~22 GiB fp32 tensor *before* selective log-softmax
    can apply its token-row chunks.  The old/reference-policy passes are under
    ``torch.no_grad()`` and immediately reduce logits to one selected log-prob
    per token, so retaining the model's bf16 output is numerically appropriate
    and removes only the redundant conversion, not any trajectory token.
    """

    # 中文说明：替换 GRPOTrainer 的本地 forward，让 no-grad 的 old/reference
    # 策略前向保留 bf16 logits，绕过 Accelerate 默认的 fp32 输出转换；
    # 否则一条 2 万多 token 的轨迹会凭空多出约 22 GiB 的 fp32 张量。
    # 这些前向本来就不回传梯度，跳过 fp32 转换不影响数值正确性。
    if not _env_bool("OPENETA_KEEP_LOGITS_BF16", True):
        return

    from swift.rlhf_trainers import grpo_trainer as swift_grpo_trainer

    trainer_cls = swift_grpo_trainer.GRPOTrainer
    current = trainer_cls._get_logps_via_local_forward
    if getattr(current, "_openeta_keep_logits_bf16", False):
        return

    def _bf16_local_forward(
        self: Any,
        model: torch.nn.Module,
        model_inputs: dict[str, Any],
        logits_to_keep: int,
        input_ids: torch.Tensor,
        compute_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if "logits_to_keep" in self.model_kwarg_keys:
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        # Accelerate stores the pre-autocast model method before installing its
        # output-to-fp32 wrapper.  DDP may add a transparent ``module`` shell,
        # so find the first owner that carries that method.  These passes are
        # no-grad, hence bypassing DDP's gradient hooks is intentional.
        owner: torch.nn.Module = model
        while not hasattr(owner, "_original_forward") and hasattr(owner, "module"):
            owner = owner.module
        original_forward = getattr(owner, "_original_forward", None)
        if original_forward is None:
            outputs = model(**model_inputs)
        else:
            try:
                parameter_dtype = next(owner.parameters()).dtype
            except StopIteration:
                parameter_dtype = torch.bfloat16
            autocast_enabled = (
                input_ids.device.type == "cuda"
                and parameter_dtype in (torch.float16, torch.bfloat16)
            )
            with torch.autocast(
                device_type=input_ids.device.type,
                dtype=parameter_dtype if autocast_enabled else torch.bfloat16,
                enabled=autocast_enabled,
            ):
                outputs = original_forward(**model_inputs)

        logits = outputs.logits[:, -(logits_to_keep + 1):-1, :]
        logits.div_(self.temperature)
        input_ids_for_logps = input_ids[:, -logits_to_keep:]

        if self.template.padding_free:
            logits_rmpad = logits.squeeze(0)
            input_ids_rmpad = input_ids_for_logps.squeeze(0)
            logps = swift_grpo_trainer.selective_log_softmax(
                logits_rmpad,
                input_ids_rmpad,
            ).unsqueeze(0)
            entropies = (
                swift_grpo_trainer.entropy_from_logits(logits_rmpad).unsqueeze(0)
                if compute_entropy else None
            )
        else:
            logps = swift_grpo_trainer.selective_log_softmax(logits, input_ids_for_logps)
            entropies = (
                swift_grpo_trainer.entropy_from_logits(logits)
                if compute_entropy else None
            )
        return logps, entropies

    _bf16_local_forward._openeta_keep_logits_bf16 = True  # type: ignore[attr-defined]
    trainer_cls._get_logps_via_local_forward = _bf16_local_forward


def _request_id(infer_request: RolloutInferRequest) -> str:
    return str(getattr(infer_request, "uuid", "") or "request")


def _choice_content(response_choice: Any) -> str:
    message = getattr(response_choice, "message", None)
    return str(getattr(message, "content", "") or "")


def _parsed_model_action(text: str) -> Any:
    """Parse the answer channel, recovering a legal action from its final line.

    The requested model format remains ``<think>...</think>\nACTION``.  Small
    models sometimes add answer-channel prose before the final action.  The
    host may execute that final allow-listed line safely, while
    :func:`_strict_action_format` records the protocol violation for reward
    shaping.  An unclosed thinking block is never recovered.
    """

    # 中文说明：宽松解析路径——先取 </think> 之后的答案段；解析失败时
    # 从答案段最后一行"找回"一个白名单内的合法动作以保证执行安全；
    # 但未闭合 <think> 的输出绝不找回（防止把被截断的思考当动作执行）。
    raw = str(text or "").strip()
    if "<think>" in raw:
        if "</think>" not in raw:
            return parse_action(raw)
        answer = raw.rsplit("</think>", 1)[-1].strip()
    else:
        answer = raw
    direct = parse_action(answer)
    if direct.valid:
        return direct
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    for line in reversed(lines):
        match = RECOVERABLE_ACTION_LINE.fullmatch(line)
        if match is not None:
            recovered = match.group(1).upper()
            parsed = parse_action(recovered)
            parsed.reason = "recovered from final answer line"
            return parsed
    return direct


def _strict_action_format(text: str) -> bool:
    """Whether the answer channel contains exactly one allow-listed action."""

    # 中文说明：严格格式判定——要求 </think> 之后"恰好只有一个白名单动作"，
    # 与上面的宽松解析分离：宽松解析保证执行安全，严格判定用于 reward 罚分。
    raw = str(text or "").strip()
    if "<think>" in raw:
        if "</think>" not in raw:
            return False
        answer = raw.rsplit("</think>", 1)[-1].strip()
    else:
        answer = raw
    return answer.upper() in ALLOWED_ACTIONS


def _format_adherence(steps: list[dict[str, Any]]) -> tuple[float, float]:
    """Return strict-format fraction and its separately reported penalty."""

    # 中文说明：统计整条 episode 中严格合规轮次占比，并据此线性折算格式罚分
    # （罚分上限为 OPENETA_FORMAT_PENALTY，默认 0.10）；空轨迹按全罚处理。
    if not steps:
        return 0.0, _format_penalty_weight()
    strict_fraction = sum(bool(step.get("strict_action_format")) for step in steps) / len(steps)
    return strict_fraction, _format_penalty_weight() * (1.0 - strict_fraction)


def _format_penalty_weight() -> float:
    # 中文说明：格式罚分权重取自环境变量 OPENETA_FORMAT_PENALTY（默认 0.10），
    # 限定在 [0,1] 区间内，越界直接报错以免静默产生异常 reward。
    value = float(os.environ.get("OPENETA_FORMAT_PENALTY", "0.10"))
    if not 0.0 <= value <= 1.0:
        raise ValueError("OPENETA_FORMAT_PENALTY must be in [0, 1]")
    return value


def _completion_loss_mask(token_ids: list[int]) -> list[int]:
    """Train every generated reasoning/action token in this environment turn.

    ``response_token_ids`` contains only tokens generated by the assistant, so
    there is no user observation or chat-template prefix to exclude here.  A
    trajectory-level GRPO advantage must therefore apply to the complete
    completion, including the reasoning before ``</think>`` and the final
    action.  Formatting remains an independent host-computed reward penalty.
    """

    # 中文说明：loss mask 全置 1——response_token_ids 里本来就只有 assistant
    # 生成的 token（不含用户观测与模板前缀），因此包括 </think> 前的推理段
    # 在内的每个生成 token 都要参与 GRPO 策略损失；格式问题另由 reward 罚分处理。
    return [1] * len(token_ids)


def _snapshot_from_mapping(payload: dict[str, Any]) -> SnapshotRef:
    return SnapshotRef(
        snapshot_id=str(payload["snapshot_id"]),
        env_id=str(payload["env_id"]),
        state_uri=str(payload["state_uri"]),
        state_sha256=str(payload["state_sha256"]),
        seed=payload.get("seed"),
        metadata=dict(payload.get("metadata") or {}),
    )


def _image_base64(observation: Any) -> tuple[str, str]:
    rgb = np.asarray(observation.cameras[0].rgb, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    content = buffer.getvalue()
    return base64.b64encode(content).decode("ascii"), hashlib.sha256(content).hexdigest()


def _distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _plain_bool(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else False
    return bool(value)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be one of true/false/1/0/yes/no/on/off")


_install_chunked_logps_patch()
_install_bf16_logit_forward_patch()


def _bob_reward(steps: list[dict[str, Any]]) -> float:
    # 中文说明：Bob（复现方）的 episode 级 reward，由宿主侧可信信号合成：
    #   成功 +1.0（任一回合达成 goal）；曾抓起方块 +0.15（部分完成的塑形信号）；
    #   环境稠密奖励截断到 [0, 0.2]；goal_score 进展 ×0.30（失败 rollout 也有梯度信号，
    #   且 goal_score 由宿主的 PositionGoalChecker 计算，模型无法自我虚报成功）；
    #   再按非法动作比例与格式违规分别扣分，最终裁剪到 [-1, 1]。
    if not steps:
        return -1.0
    success = any(bool(step["goal_success"]) for step in steps)
    grasped = any(bool(step["is_grasped"]) for step in steps)
    valid_fraction = sum(bool(step["valid_action"]) for step in steps) / len(steps)
    environment_dense = max(
        0.0,
        min(0.2, max(float(step["environment_reward"]) for step in steps)),
    )
    # goal_score is computed by the host-owned PositionGoalChecker after each
    # simulator step.  It gives failed Bob rollouts a useful learning signal
    # without allowing the model to claim success itself.
    goal_progress = max(float(step.get("goal_score", 0.0)) for step in steps)
    dense = environment_dense + 0.30 * goal_progress
    reward = (1.0 if success else 0.0) + (0.15 if grasped else 0.0) + dense
    reward -= 0.1 * (1.0 - valid_fraction)
    _, format_penalty = _format_adherence(steps)
    reward -= format_penalty
    return max(-1.0, min(1.0, reward))


def _alice_feasibility_reward(
    state: _EpisodeState,
    compiled: CompiledTask,
    *,
    replay_equal: bool,
) -> tuple[float, dict[str, float | bool]]:
    """Dense pre-validity reward; it never turns an invalid task into a task."""

    # 中文说明：Alice 出题"无效"时的稠密塑形 reward（最高只能到 0，永远不会
    # 把无效题目变成有效题目）：从 -1 起步，按合法动作比例、接近方块进度、
    # 是否抓起过、方块位移进度、回放一致性逐项加分，引导 Alice 学会产出
    # "真实改变方块位置且可复现"的候选题目。
    if not state.steps:
        return -1.0, {
            "valid_action_fraction": 0.0,
            "strict_action_format_fraction": 0.0,
            "ever_grasped": False,
            "displacement_progress": 0.0,
            "replay_equal": replay_equal,
        }
    valid_fraction = sum(bool(step["valid_action"]) for step in state.steps) / len(state.steps)
    strict_format_fraction, _ = _format_adherence(state.steps)
    ever_grasped = any(bool(step["is_grasped"]) for step in state.steps)
    closest_tcp_distance = min(
        float(step.get("cube_tcp_distance", state.initial_tcp_distance)) for step in state.steps
    )
    approach_progress = max(
        -1.0,
        min(
            1.0,
            (state.initial_tcp_distance - closest_tcp_distance)
            / max(state.initial_tcp_distance, 1e-9),
        ),
    )
    displacement_progress = min(
        1.0,
        max(0.0, float(compiled.displacement)) / max(state.min_goal_displacement, 1e-9),
    )
    reward = (
        -1.0
        + 0.10 * valid_fraction
        + 0.30 * approach_progress
        + 0.25 * float(ever_grasped)
        + 0.20 * displacement_progress
        + 0.15 * float(replay_equal)
    )
    return max(-1.0, min(0.0, reward)), {
        "valid_action_fraction": valid_fraction,
        "strict_action_format_fraction": strict_format_fraction,
        "approach_progress": approach_progress,
        "closest_cube_tcp_distance": closest_tcp_distance,
        "ever_grasped": ever_grasped,
        "displacement_progress": displacement_progress,
        "replay_equal": replay_equal,
    }


def _boundary_score(success_rate: float, target_success_rate: float) -> float:
    """Score tasks highest near Bob's configured competence boundary."""

    # 中文说明：Alice 的出题质量分——冻结 Bob 的复现成功率越接近目标值
    # （OPENETA_TARGET_SUCCESS_RATE，默认 0.45，即"难度卡在 Bob 能力边界上"）
    # 得分越高；太简单（成功率≈1）或太难（成功率≈0）的题目都得低分。
    return boundary_score(success_rate, target_success_rate)


def _find_rollout_info(value: Any) -> dict[str, Any] | None:
    # 中文说明：在任意嵌套结构中递归查找带 openeta.embodied_rollout.v1 标记的
    # rollout 记录；列表从尾部倒序查找，优先取最新（最终）一条回执。
    if isinstance(value, dict):
        if value.get("schema_version") == "openeta.embodied_rollout.v1":
            return value
        for nested in value.values():
            found = _find_rollout_info(nested)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in reversed(value):
            found = _find_rollout_info(nested)
            if found is not None:
                return found
    return None


# 中文说明：单条 episode 的全部宿主侧状态。每个 rollout 请求（按 request_id 索引）
# 对应一个实例，持有仿真环境句柄、初始快照、目标谓词、逐轮 step 记录以及
# Bob 评估相关配置；Alice 与 Bob 两个角色共用同一结构，由 role 字段区分语义。
@dataclass(slots=True)
class _EpisodeState:
    request_id: str
    role: str
    env: ManiSkillSimulatorAdapter | IsolatedManiSkillSimulatorAdapter
    snapshot: SnapshotRef
    instruction: str
    max_steps: int
    initial_state: dict[str, Any]
    initial_tcp_distance: float
    goal_predicate: dict[str, Any] | None
    min_goal_displacement: float
    goal_tolerance: float
    replay_tolerance: float
    thinking_enabled: bool
    camera_resolution: int = 128
    seen_goal_cells: set[str] = field(default_factory=set)
    bob_evaluator_url: str | None = None
    bob_evaluations: int = 4
    bob_max_steps: int = 40
    bob_evaluator_timeout: float = 300.0
    bob_temperature: float = 0.7
    bob_policy_version: str = "unknown"
    target_success_rate: float = 0.45
    proposal_id: str = ""
    alice_policy_version: str = "unknown"
    steps: list[dict[str, Any]] = field(default_factory=list)
    image_hashes: list[str] = field(default_factory=list)
    pending_prompt: str | None = None
    generated_tokens_total: int = 0
    turn_token_counts: list[int] = field(default_factory=list)
    last_allowed_tokens: int = 0
    budget_termination_reason: str | None = None
    budget: RolloutBudget | None = None
    done: bool = False
    final_info: dict[str, Any] = field(default_factory=dict)


def _align_rollout_output(scheduler: Any, item: RolloutOutput) -> RolloutOutput:
    """Idempotently align final-turn IDs, masks and sampled log probabilities."""

    assistant_turns = sum(
        message.get("role") == "assistant" for message in (item.messages or [])
    )
    if len(item.response_token_ids) < assistant_turns:
        choice = item.response.choices[0]
        final_ids = list(choice.token_ids or [])
        item.response_token_ids.append(final_ids)
        item.response_loss_mask.append([1] * len(final_ids))
        final_logprobs = scheduler._extract_logprobs_from_choice(choice)
        if final_logprobs:
            item.rollout_logprobs.append(final_logprobs)
    if len(item.response_token_ids) > assistant_turns:
        raise RuntimeError("rollout contains more token turns than assistant messages")
    while len(item.response_loss_mask) < len(item.response_token_ids):
        item.response_loss_mask.append([])
    assistant_messages = [
        str(message.get("content") or "")
        for message in (item.messages or [])
        if message.get("role") == "assistant"
    ]
    for index, token_ids in enumerate(item.response_token_ids):
        if index >= len(assistant_messages):
            item.response_loss_mask[index] = [0] * len(token_ids)
            if index < len(item.rollout_logprobs):
                item.rollout_logprobs[index] = []
            continue
        item.response_loss_mask[index] = _completion_loss_mask(token_ids)
        if index < len(item.rollout_logprobs):
            logprobs = item.rollout_logprobs[index]
            if len(logprobs) == len(token_ids):
                item.rollout_logprobs[index] = [
                    value
                    for value, include in zip(logprobs, item.response_loss_mask[index])
                    if include
                ]
    if item.rollout_logprobs:
        trained_tokens = sum(sum(mask) for mask in item.response_loss_mask)
        logprob_tokens = sum(len(values) for values in item.rollout_logprobs)
        if trained_tokens != logprob_tokens:
            item.rollout_logprobs = []
    return item


# 中文说明：Alice/Bob 共用的 multi-turn scheduler 基类。它架在 ms-swift 的
# MultiTurnScheduler 之上，把"模型每轮输出一个动作"接到真实 ManiSkill 仿真上：
# on_trajectory_start 建环境并恢复快照，on_turn_end 执行动作并判定回合结束，
# step 把新观测（图像+文本 prompt）喂回模型，episode 结束时 _finalize 算 reward。
# 子类仅以 role 类属性区分 alice / bob。
class _EmbodiedScheduler(MultiTurnScheduler):
    role = ""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._episodes: dict[str, _EpisodeState] = {}
        self._finished: set[str] = set()
        self._active_request_configs: dict[str, RequestConfig] = {}
        self._checker = PositionGoalChecker()
        # 中文说明：运行模式开关全部来自环境变量——是否启用思考模板、是否走
        # staged colocate 三阶段调度、是否把 ManiSkill 放到独立 spawn 子进程
        # （进程隔离可避免 SAPIEN 的 CUDA/Vulkan 上下文与 vLLM sleep 模式的
        # CuMem 分配器在同一进程内互相干扰）。
        self._thinking_enabled = _env_bool("OPENETA_ENABLE_THINKING", False)
        self._staged_colocate = _env_bool("OPENETA_STAGED_COLOCATE", False)
        self._isolate_maniskill = _env_bool("OPENETA_MANISKILL_PROCESS_ISOLATION", False)
        self._thinking_max_tokens = int(os.environ.get("OPENETA_THINKING_MAX_TOKENS", "1024"))
        if self._thinking_max_tokens < 16:
            raise ValueError("OPENETA_THINKING_MAX_TOKENS must be at least 16")
        self._trajectory_max_tokens = int(
            os.environ.get("OPENETA_TRAJECTORY_MAX_TOKENS", "24576")
        )
        self._context_max_tokens = int(
            os.environ.get("OPENETA_CONTEXT_MAX_TOKENS", "32768")
        )
        if self._trajectory_max_tokens < 1:
            raise ValueError("OPENETA_TRAJECTORY_MAX_TOKENS must be positive")
        if self._context_max_tokens < 1:
            raise ValueError("OPENETA_CONTEXT_MAX_TOKENS must be positive")
        self._artifact_root = Path(
            os.environ.get("OPENETA_SWIFT_ARTIFACT_ROOT", str(REPO_ROOT / "runs/swift_rollouts"))
        ).resolve()
        self._artifact_root.mkdir(parents=True, exist_ok=True)

    async def run(
        self,
        infer_request: RolloutInferRequest,
        request_config: RequestConfig,
        **kwargs: Any,
    ) -> RolloutOutput | list[RolloutOutput]:
        bounded = self.prepare_request_config(request_config)
        request_id = _request_id(infer_request)
        self._active_request_configs[request_id] = bounded
        try:
            output = await super().run(infer_request, bounded, **kwargs)
        finally:
            self._active_request_configs.pop(request_id, None)
        outputs = output if isinstance(output, list) else [output]
        for item in outputs:
            # ms-swift 4.4.2's base scheduler omits the final turn IDs when a
            # non-continuation multi-turn rollout already has earlier IDs.
            # Repair that locally so every assistant action participates in
            # the policy loss and rollout-importance correction.
            # 中文说明：此处修复上游 ms-swift 4.4.2 的一个遗漏——多轮 rollout 的
            # 最后一轮 token ids 可能没写入 response_token_ids，导致该轮动作
            # 不参与策略损失与 rollout importance 修正；本地补齐 token ids、
            # loss mask 和对应的 rollout logprobs。
            _align_rollout_output(self, item)
        return output

    def remaining_generation_tokens(self, infer_request: RolloutInferRequest) -> int:
        """Return the request-local cumulative generation budget."""

        state = self._episodes.get(_request_id(infer_request))
        if state is None or state.budget is None:
            return self._trajectory_max_tokens
        return state.budget.remaining_generation_tokens

    @property
    def per_turn_token_limit(self) -> int:
        return self._thinking_max_tokens if self._thinking_enabled else 12

    def note_generation_allowance(
        self, infer_request: RolloutInferRequest, allowed_tokens: int
    ) -> None:
        state = self._episodes.get(_request_id(infer_request))
        if state is not None:
            state.last_allowed_tokens = int(allowed_tokens)

    def prepare_request_config(self, request_config: RequestConfig) -> RequestConfig:
        """Apply identical per-turn bounds in server and colocate paths."""

        # Copy per request: mutating a trainer-shared RequestConfig would create
        # cross-request races under async rollout.
        # 中文说明：RequestConfig 在 trainer 内是跨请求共享的，必须 deepcopy 后再改，
        # 否则异步并发 rollout 之间会互相污染采样参数。
        bounded = deepcopy(request_config)
        if self._thinking_enabled:
            # Qwen's thinking template must be free to emit its reasoning and
            # closing </think> before the final bounded action.  Execution is
            # still safe because parse_action maps anything outside the action
            # allowlist to invalid DONE.
            bounded.max_tokens = self._thinking_max_tokens
            bounded.structured_outputs_regex = None
        else:
            # 中文说明：非思考模式下把每轮生成上限压到 12 token，并用 ACTION_REGEX
            # 结构化输出约束，使模型只能吐出白名单内的单个动作码。
            bounded.max_tokens = min(int(bounded.max_tokens or 12), 12)
            bounded.structured_outputs_regex = ACTION_REGEX
        bounded.return_details = True
        bounded.logprobs = True
        return bounded

    async def on_trajectory_start(self, requests: list[RolloutInferRequest]) -> None:
        for infer_request in requests:
            request_id = _request_id(infer_request)
            await self._close(request_id)
            self._finished.discard(request_id)
            data = dict(getattr(infer_request, "data_dict", None) or {})
            config = dict(data.get("env_request") or {})
            # 中文说明：环境配置必须随数据集经 env_request 传入（需要
            # --vllm_server_pass_dataset true），否则无法确定要恢复哪份快照。
            if not config:
                raise ValueError("env_request is required; enable --vllm_server_pass_dataset true")
            configured_role = str(data.get("role") or config.get("role") or self.role).lower()
            if configured_role != self.role:
                raise ValueError(f"{type(self).__name__} requires role={self.role!r}")
            snapshot_payload = config.get("snapshot")
            if not isinstance(snapshot_payload, dict):
                raise TypeError("env_request.snapshot must contain the full SnapshotRef mapping")
            snapshot = _snapshot_from_mapping(snapshot_payload)
            env_id = str(config.get("env_id") or snapshot.metadata.get("env_id") or "PickCube-v1")
            # 中文说明：按 OPENETA_MANISKILL_PROCESS_ISOLATION 选择适配器——
            # 隔离模式把仿真放到独立 spawn 子进程跑，非隔离模式则与 vLLM 同进程。
            adapter_cls = (
                IsolatedManiSkillSimulatorAdapter
                if self._isolate_maniskill
                else ManiSkillSimulatorAdapter
            )
            env = adapter_cls(
                env_id=env_id,
                camera_resolution=int(config.get("camera_resolution", 128)),
                translation_step_m=float(config.get("translation_step_m", 0.05)),
                fine_translation_step_m=float(config.get("fine_translation_step_m", 0.01)),
                max_episode_steps=int(config.get("episode_horizon", 120)),
                snapshot_dir=self._artifact_root / "snapshots",
            )
            try:
                # 中文说明：先 reset 再 restore_snapshot，把环境精确恢复到数据集
                # 指定的初始状态（含随机种子），保证 Alice/Bob/评估 Bob 看到的
                # 起点完全一致、可复现。
                env.reset(seed=snapshot.seed if snapshot.seed is not None else config.get("seed"))
                env.restore_snapshot(snapshot)
                initial_state = env.task_state()
                predicate = config.get("goal_predicate")
                if self.role == "alice":
                    # Alice creates the goal.  Hide the original PickCube marker
                    # so it cannot silently turn proposal generation into the
                    # stock environment task.
                    # 中文说明：Alice 是"出题方"，不需要预设目标；把原生 PickCube 的
                    # 绿色 goal marker 藏到远处，防止她把原生任务当成自己出的题。
                    env.set_goal_position(HIDDEN_ALICE_GOAL)
                    instruction = str(config.get("instruction") or ALICE_TASK)
                    predicate = None
                else:
                    if not isinstance(predicate, dict):
                        position = config.get("goal_position")
                        predicate = {
                            "type": "cube_at_position",
                            "position": position,
                            "tolerance": float(config.get("goal_tolerance", 0.025)),
                            "source": "alice_final_state",
                        }
                    if not isinstance(predicate.get("position"), list):
                        raise ValueError("Bob requires goal_predicate.position")
                    # 中文说明：Bob 是"复现方"，把绿色 goal marker 摆到 Alice 题目
                    # 指定的目标位置，让 Bob 看着 marker 与指令去复现方块状态。
                    env.set_goal_position(predicate["position"])
                    instruction = str(config.get("instruction") or BOB_TASK)
                observation = env.observe()
            except Exception:
                env.close()
                raise

            configured_max_steps = int(config.get("max_steps") or self.max_turns or 40)
            if self.max_turns:
                configured_max_steps = min(configured_max_steps, int(self.max_turns))
            state = _EpisodeState(
                request_id=request_id,
                role=self.role,
                env=env,
                snapshot=snapshot,
                instruction=instruction,
                max_steps=configured_max_steps,
                initial_state=initial_state,
                initial_tcp_distance=_distance(
                    list(initial_state["cube_position"]),
                    list(observation.robot.end_effector_pose.get("xyz") or [0.0, 0.0, 0.0]),
                ),
                goal_predicate=predicate,
                min_goal_displacement=float(config.get("min_goal_displacement", 0.04)),
                goal_tolerance=float(config.get("goal_tolerance", 0.025)),
                replay_tolerance=float(config.get("replay_tolerance", 0.005)),
                thinking_enabled=self._thinking_enabled,
                camera_resolution=int(config.get("camera_resolution", 128)),
                seen_goal_cells=set(map(str, config.get("seen_goal_cells") or [])),
                bob_evaluator_url=(
                    str(config.get("bob_evaluator_url") or os.environ.get("OPENETA_BOB_EVALUATOR_URL") or "").strip()
                    or None
                ),
                bob_evaluations=int(
                    config.get("bob_evaluations")
                    or os.environ.get("OPENETA_BOB_EVALUATIONS", "4")
                ),
                bob_max_steps=int(
                    config.get("bob_max_steps")
                    or os.environ.get("OPENETA_BOB_MAX_STEPS", str(config.get("max_steps") or 40))
                ),
                bob_evaluator_timeout=float(
                    config.get("bob_evaluator_timeout")
                    or os.environ.get("OPENETA_BOB_EVALUATOR_TIMEOUT", "300")
                ),
                bob_temperature=float(
                    config.get("bob_temperature")
                    or os.environ.get("OPENETA_BOB_TEMPERATURE", "0.7")
                ),
                bob_policy_version=str(
                    config.get("bob_policy_version")
                    or os.environ.get("OPENETA_BOB_POLICY_VERSION", "unknown")
                ),
                target_success_rate=float(
                    config.get("target_success_rate")
                    or os.environ.get("OPENETA_TARGET_SUCCESS_RATE", "0.45")
                ),
                proposal_id=str(config.get("proposal_id") or request_id),
                alice_policy_version=str(
                    config.get("alice_policy_version")
                    or os.environ.get("OPENETA_ALICE_POLICY_VERSION", "unknown")
                ),
            )
            state.budget = RolloutBudget(
                per_turn_limit=self.per_turn_token_limit,
                trajectory_limit=self._trajectory_max_tokens,
                context_limit=self._context_max_tokens,
            )
            if state.role == "alice":
                if state.bob_evaluations < 1:
                    raise ValueError("bob_evaluations must be positive")
                if state.bob_max_steps < 1:
                    raise ValueError("bob_max_steps must be positive")
                if not 0.0 < state.target_success_rate < 1.0:
                    raise ValueError("target_success_rate must be in (0, 1)")
            image, image_hash = _image_base64(observation)
            state.image_hashes.append(image_hash)
            # 中文说明：首帧观测编码为 base64 图像 + 文本 prompt 一起作为第一条
            # user 消息；图像哈希同时记录下来，供 artifact 追溯与去重。
            prompt = build_embodied_prompt(
                self.role,
                observation,
                step_index=0,
                max_steps=state.max_steps,
                initial_cube_position=list(initial_state["cube_position"]),
                instruction=instruction,
                enable_thinking=state.thinking_enabled,
            )
            infer_request.messages = [{"role": "user", "content": f"<image>\n{prompt}"}]
            infer_request.images = [image]
            infer_request.chat_template_kwargs = {
                **dict(getattr(infer_request, "chat_template_kwargs", None) or {}),
                "enable_thinking": state.thinking_enabled,
            }
            self._episodes[request_id] = state

    async def on_turn_end(
        self,
        infer_request: RolloutInferRequest,
        response_choice: Any,
        current_turn: int,
    ) -> dict[str, Any]:
        request_id = _request_id(infer_request)
        state = self._episodes.get(request_id)
        if state is None:
            return {"done": True, "rollout_infos": self._failure_info(request_id, "missing session")}

        raw_completion = _choice_content(response_choice)
        parsed = _parsed_model_action(raw_completion)
        generated_token_ids = list(getattr(response_choice, "token_ids", None) or [])
        if not generated_token_ids and self.tokenizer is not None:
            generated_token_ids = self.tokenizer.encode(
                raw_completion, add_special_tokens=False
            )
        generated_tokens = len(generated_token_ids)
        allowed_tokens = state.last_allowed_tokens or min(
            self.per_turn_token_limit,
            self._trajectory_max_tokens - state.generated_tokens_total,
        )
        if generated_tokens > allowed_tokens:
            raise RuntimeError(
                f"engine generated {generated_tokens} tokens with allowance {allowed_tokens}"
            )
        if state.budget is None:
            raise RuntimeError("rollout budget was not initialized")
        state.budget.consume(generated_tokens, allowed_tokens=allowed_tokens)
        state.generated_tokens_total = state.budget.generated_tokens_total
        state.turn_token_counts.append(generated_tokens)
        state.budget_termination_reason = state.budget.termination_reason
        active_config = self._active_request_configs.get(request_id)
        if active_config is not None and state.budget.remaining_generation_tokens > 0:
            active_config.max_tokens = min(
                self.per_turn_token_limit,
                state.budget.remaining_generation_tokens,
            )
        try:
            result = state.env.step(EnvAction(action_type=parsed.action, code=parsed.action))
            task_state = state.env.task_state()
            observation = result.observation
            tcp_position = list(observation.robot.end_effector_pose.get("xyz") or [])
            cube_tcp_distance = (
                _distance(list(task_state["cube_position"]), tcp_position)
                if len(tcp_position) == 3
                else state.initial_tcp_distance
            )
            goal_success = False
            goal_score = 0.0
            goal_details: dict[str, Any] = {}
            if self.role == "bob" and state.goal_predicate is not None:
                evaluation = self._checker.evaluate(
                    state=task_state,
                    predicate=state.goal_predicate,
                )
                goal_success = evaluation.success
                goal_score = evaluation.score
                goal_details = evaluation.details
            # 中文说明：把本轮的原始输出、解析后的动作、环境奖励、goal 判定、
            # 方块/夹爪位置等全部记入 step 日志，是后续 reward 计算与 artifact
            # 落盘的唯一事实来源；goal 判定仅 Bob 角色做（Alice 没有预设目标）。
            state.steps.append({
                "turn": current_turn,
                "completion": raw_completion,
                "allowed_max_tokens": allowed_tokens,
                "generated_tokens": generated_tokens,
                "generated_tokens_total": state.generated_tokens_total,
                "action": parsed.action,
                "valid_action": parsed.valid,
                "strict_action_format": _strict_action_format(raw_completion),
                "invalid_reason": parsed.reason,
                "environment_reward": float(result.reward),
                "goal_success": goal_success,
                "goal_score": goal_score,
                "goal_details": goal_details,
                "cube_position": list(task_state["cube_position"]),
                "tcp_position": tcp_position,
                "cube_tcp_distance": cube_tcp_distance,
                "is_grasped": _plain_bool(task_state.get("is_grasped", False)),
                "terminated": bool(result.terminated),
                "truncated": bool(result.truncated),
                "source_image_width": state.camera_resolution,
                "source_image_height": state.camera_resolution,
            })
            # ``parse_action`` deliberately maps malformed model output to the
            # inert DONE action so it can never execute an untrusted command.
            # That fallback is not, however, an intentional model request to
            # end the trajectory.  Treat only a *valid* DONE as terminal;
            # otherwise one token-limit truncation would turn every remaining
            # environment step into a one-turn episode.
            # 中文说明：parse_action 会把非法/被截断的输出映射成惰性 DONE 以保证
            # 执行安全，但那并不代表模型真的想结束 episode；因此只有"合法解析出
            # 的 DONE"才算主动结束，否则一次 token 截断就会让后续每轮都立刻终止。
            done = bool(
                goal_success
                or (parsed.valid and parsed.action == "DONE")
                or result.terminated
                or result.truncated
                or current_turn >= state.max_steps
                or state.budget_termination_reason is not None
            )
            if done:
                state.done = True
                final_info = await self._finalize(state)
                state.final_info = final_info
                self._write_artifact(state)
                await self._close(request_id, mark_finished=True)
                return {"done": True, "rollout_infos": final_info | {"images": list(infer_request.images)}}

            next_image, image_hash = _image_base64(observation)
            state.image_hashes.append(image_hash)
            infer_request.images.append(next_image)
            state.pending_prompt = build_embodied_prompt(
                self.role,
                observation,
                step_index=current_turn,
                max_steps=state.max_steps,
                initial_cube_position=list(state.initial_state["cube_position"]),
                instruction=state.instruction,
                enable_thinking=state.thinking_enabled,
            )
            if not parsed.valid:
                # 中文说明：动作非法时在下一轮 prompt 里追加纠错提示，明确告知
                # 上一轮的输出没有产生任何物理动作，引导模型回到规定格式。
                state.pending_prompt += (
                    " Your previous response was invalid or truncated and no physical action "
                    "was applied. Keep the next reasoning brief, close </think>, then put exactly "
                    "one legal action code on the final line."
                )
            return {
                "done": False,
                "rollout_infos": self._progress_info(state) | {"images": list(infer_request.images)},
            }
        # The scheduler is an infrastructure boundary: a simulator exception
        # must close this request and become an explicit receipt instead of
        # crashing every concurrent rollout.
        # 中文说明：scheduler 是基础设施边界——仿真异常只终止当前这一条请求，
        # 生成一条带 infrastructure_error 的失败回执，绝不让异常扩散到
        # 同批并发的其它 rollout。
        except Exception as exc:  # noqa: BLE001
            info = self._failure_info(request_id, f"{type(exc).__name__}: {exc}")
            state.final_info = info
            self._write_artifact(state)
            await self._close(request_id, mark_finished=True)
            return {"done": True, "rollout_infos": info | {"images": list(infer_request.images)}}

    def step(
        self,
        infer_request: RolloutInferRequest,
        response_choice: Any,
        current_turn: int,
    ) -> dict[str, Any]:
        state = self._episodes.get(_request_id(infer_request))
        if state is None or state.pending_prompt is None:
            return {"infer_request": infer_request}
        # 中文说明：把 on_turn_end 准备好的下一轮观测（新图像 + 文本 prompt）
        # 追加为 user 消息，驱动模型继续下一轮交互；同时回传本轮生成的
        # token ids 与全 1 loss mask，供训练侧使用。
        infer_request.messages.append({
            "role": "user",
            "content": f"<image>\n{state.pending_prompt}",
        })
        state.pending_prompt = None
        token_ids = list(getattr(response_choice, "token_ids", None) or [])
        if not token_ids and self.tokenizer is not None:
            token_ids = self.tokenizer.encode(_choice_content(response_choice), add_special_tokens=False)
        return {
            "infer_request": infer_request,
            "response_token_ids": token_ids,
            "response_loss_mask": [1] * len(token_ids),
        }

    def check_finished(
        self,
        infer_request: RolloutInferRequest,
        response_choice: Any,
        current_turn: int,
    ) -> bool:
        request_id = _request_id(infer_request)
        return request_id in self._finished or super().check_finished(
            infer_request, response_choice, current_turn
        )

    def _progress_info(self, state: _EpisodeState) -> dict[str, Any]:
        return {
            "schema_version": "openeta.embodied_rollout.v1",
            "role": state.role,
            "request_id": state.request_id,
            "snapshot_sha256": state.snapshot.state_sha256,
            "step_count": len(state.steps),
            "generated_tokens_total": state.generated_tokens_total,
            "turn_token_counts": list(state.turn_token_counts),
            "trajectory_token_limit": self._trajectory_max_tokens,
            "per_turn_token_limit": self.per_turn_token_limit,
            "camera_resolution": state.camera_resolution,
            "budget_termination_reason": state.budget_termination_reason,
            "done": False,
        }

    async def _finalize(self, state: _EpisodeState) -> dict[str, Any]:
        # 中文说明：episode 结束时的汇总入口。Bob 直接用宿主侧信号算最终 reward；
        # Alice 则先编译/校验她出的题目，无效题给塑形 reward，有效题还需要
        # 冻结 Bob 的复现评估结果才能定分。
        if state.role == "bob":
            success = any(bool(step["goal_success"]) for step in state.steps)
            strict_format_fraction, format_penalty = _format_adherence(state.steps)
            return {
                **self._progress_info(state),
                "done": True,
                "success": success,
                "reward": _bob_reward(state.steps),
                "valid_action_fraction": (
                    sum(bool(step["valid_action"]) for step in state.steps) / len(state.steps)
                    if state.steps else 0.0
                ),
                "strict_action_format_fraction": strict_format_fraction,
                "format_penalty": format_penalty,
                "ever_grasped": any(bool(step["is_grasped"]) for step in state.steps),
                "termination_reason": self._termination_reason(state, success=success),
                "goal_predicate": state.goal_predicate,
                "image_hashes": list(state.image_hashes),
                "infrastructure_error": None,
            }

        proposal = self._finalize_alice_proposal(state)
        if not proposal.compiled_valid:
            return proposal.alice_rollout_info
        # 中文说明：staged colocate 模式下不在这里调用 Bob，而是把可序列化的
        # proposal 随 rollout_info 一并传出，由编排层在同一 vLLM engine 内
        # 切到冻结 Bob LoRA 完成评估后再补算 Alice 的最终 reward。
        if self._staged_colocate:
            return {
                **proposal.alice_rollout_info,
                "proposal": proposal.to_dict(),
            }

        evaluator_error: str | None = None
        receipts: list[BobEvalReceipt] = []
        try:
            # 中文说明：server 模式下，在这里同步嵌套调用外部"冻结 Bob"评估服务，
            # 对 Alice 的题目做 K 次独立复现，收集回执后折算 Alice 最终 reward。
            receipts = await self._evaluate_alice_task_with_bob(state, proposal)
        except Exception as exc:  # noqa: BLE001
            evaluator_error = f"{type(exc).__name__}: {exc}"
        return self._finalize_alice_with_bob_receipts(
            proposal,
            receipts,
            evaluator_error=evaluator_error,
        )

    def _finalize_alice_proposal(self, state: _EpisodeState) -> AliceProposal:
        """Compile and validate Alice without invoking Bob.

        The returned object is fully serializable and is the only data passed
        across the Alice/Bob phase boundary in staged-colocate mode.
        """

        # 中文说明：先把 Alice 的轨迹编译成候选题目（含回放一致性校验），
        # 再按题目是否有效分流：无效题走稠密塑形 reward 并叠加格式罚分；
        # 有效题此阶段 reward 暂记 0、标记 reward_pending，等待冻结 Bob 评估。
        compiled, final_state, replay_equal = self._compile_alice(state)
        repeated = compiled.goal_cell in state.seen_goal_cells if compiled.goal_cell else False
        feasibility_metrics: dict[str, float | bool] | None = None
        if not compiled.valid:
            reward, feasibility_metrics = _alice_feasibility_reward(
                state,
                compiled,
                replay_equal=replay_equal,
            )
            reward_stage = "invalid_proposal_shaping"
        else:
            reward = 0.0
            reward_stage = "awaiting_frozen_bob"
        strict_format_fraction, format_penalty = _format_adherence(state.steps)
        if not compiled.valid:
            reward = max(-1.0, min(1.0, reward - format_penalty))
        rollout_info = {
            **self._progress_info(state),
            "done": True,
            "success": compiled.valid,
            "reward": reward,
            "reward_stage": reward_stage,
            "strict_action_format_fraction": strict_format_fraction,
            "format_penalty": format_penalty,
            "termination_reason": self._termination_reason(state, success=compiled.valid),
            "compiled_task": {
                "task_id": compiled.task_id,
                "instruction": compiled.instruction,
                "goal_predicate": compiled.goal_predicate,
                "goal_cell": compiled.goal_cell,
                "displacement": compiled.displacement,
                "valid": compiled.valid,
                "reason": compiled.reason,
            },
            "final_state": final_state,
            "replay_equal": replay_equal,
            "repeated": repeated,
            "target_success_rate": state.target_success_rate,
            "bob_evaluations": state.bob_evaluations,
            "bob_max_steps": state.bob_max_steps,
            "camera_resolution": state.camera_resolution,
            "thinking_enabled": state.thinking_enabled,
            "bob_evaluation": None,
            "bob_evaluator_error": None,
            "feasibility_metrics": feasibility_metrics,
            "image_hashes": list(state.image_hashes),
            "infrastructure_error": None,
            "reward_pending": compiled.valid,
            "phase": "alice_rollout",
            "adapter_name": "openeta_current_alice",
            "policy_version": state.alice_policy_version,
            "proposal_id": state.proposal_id,
        }
        return AliceProposal(
            proposal_id=state.proposal_id,
            alice_policy_version=state.alice_policy_version,
            bob_policy_version=state.bob_policy_version,
            snapshot_sha256=state.snapshot.state_sha256,
            snapshot=state.snapshot.to_dict(),
            goal_predicate=dict(compiled.goal_predicate or {}),
            compiled_valid=compiled.valid,
            replay_equal=replay_equal,
            repeated=repeated,
            target_success_rate=state.target_success_rate,
            alice_rollout_info=rollout_info,
        )

    @staticmethod
    def _finalize_alice_with_bob_receipts(
        proposal: AliceProposal,
        receipts: list[BobEvalReceipt],
        *,
        evaluator_error: str | None = None,
    ) -> dict[str, Any]:
        """Pure receipt validation/reward path shared by server and staged modes."""

        # 中文说明：纯函数式的最终定分路径（server 与 staged 两种模式共用）——
        # 汇总 K 份 Bob 回执算出复现成功率，再按目标成功率折算 Alice 最终 reward。
        return finalize_alice_reward(
            proposal,
            receipts,
            evaluator_error=evaluator_error,
        )

    async def _evaluate_alice_task_with_bob(
        self,
        state: _EpisodeState,
        proposal: AliceProposal,
    ) -> list[BobEvalReceipt]:
        """Evaluate one valid Alice task K times with a separate frozen Bob server."""

        if not state.bob_evaluator_url:
            raise ValueError(
                "OPENETA_BOB_EVALUATOR_URL (or env_request.bob_evaluator_url) is required "
                "for valid Alice proposals"
            )
        # 中文说明：Bob 嵌套评估——把 Alice 的快照与目标谓词原样转发给独立的
        # 冻结 Bob server，让它从同一初始状态出发、用不同随机种子独立复现 K 次
        # （OPENETA_BOB_EVALUATIONS）；Alice 完全看不到自己的轨迹被如何复现，
        # 评估的独立性是 Alice reward 可信的前提。
        base_url = state.bob_evaluator_url.rstrip("/")
        endpoint = base_url if base_url.endswith("/infer") else f"{base_url}/infer"
        endpoint += "/"
        bob_env_request = {
            "role": "bob",
            "env_id": str(state.snapshot.metadata.get("env_id") or "PickCube-v1"),
            "seed": state.snapshot.seed,
            "snapshot": state.snapshot.to_dict(),
            "instruction": BOB_TASK,
            "goal_predicate": proposal.goal_predicate,
            "max_steps": state.bob_max_steps,
            "camera_resolution": state.camera_resolution,
            "goal_tolerance": state.goal_tolerance,
        }
        bob_requests = build_bob_eval_requests(proposal, state.bob_evaluations)
        infer_requests = []
        for bob_request in bob_requests:
            request_env = dict(bob_env_request)
            request_env["seed"] = bob_request.seed
            row = {
                "role": "bob",
                "env_request": request_env,
                "chat_template_kwargs": {"enable_thinking": state.thinking_enabled},
            }
            infer_requests.append({
                "messages": [{"role": "user", "content": "Environment bootstrap pending."}],
                "data_dict": row,
                "uuid": bob_request.evaluation_id,
            })
        payload = {
            "infer_requests": infer_requests,
            "request_config": {
                "max_tokens": 12,
                "temperature": state.bob_temperature,
                "top_p": 0.9,
                "logprobs": False,
                "return_details": True,
                "n": 1,
            },
            "use_tqdm": False,
        }
        timeout = aiohttp.ClientTimeout(total=state.bob_evaluator_timeout)
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.post(endpoint, json=payload) as response,
        ):
            response.raise_for_status()
            outputs = await response.json()
        if not isinstance(outputs, list) or len(outputs) != state.bob_evaluations:
            raise RuntimeError(
                f"Bob evaluator returned {len(outputs) if isinstance(outputs, list) else 'non-list'} "
                f"outputs for {state.bob_evaluations} requests"
            )
        receipts: list[BobEvalReceipt] = []
        # 中文说明：对每条 Bob 回执做严格的防伪造校验——必须带有可信的 bob 角色
        # rollout_info、恢复快照哈希一致、目标谓词逐字段一致（位置距离 < 1e-9、
        # 容差一致）、且无基础设施错误；任何一项不符都直接判本次评估失败，
        # 防止评估方用不同的初始状态或不同的题目"替考"。
        for bob_request, output in zip(bob_requests, outputs):
            info = _find_rollout_info(output.get("rollout_infos") if isinstance(output, dict) else output)
            if not info or info.get("role") != "bob":
                raise RuntimeError("Bob evaluator response has no trusted Bob rollout receipt")
            if info.get("snapshot_sha256") != state.snapshot.state_sha256:
                raise RuntimeError("Bob evaluator restored a different initial snapshot")
            returned_goal = info.get("goal_predicate")
            expected_position = proposal.goal_predicate.get("position")
            if (
                not isinstance(returned_goal, dict)
                or returned_goal.get("type") != proposal.goal_predicate.get("type")
                or not isinstance(returned_goal.get("position"), list)
                or not isinstance(expected_position, list)
                or _distance(returned_goal["position"], expected_position) > 1e-9
                or abs(
                    float(returned_goal.get("tolerance", -1.0))
                    - float(proposal.goal_predicate.get("tolerance", -2.0))
                ) > 1e-9
            ):
                raise RuntimeError("Bob evaluator used a different goal predicate")
            if info.get("infrastructure_error"):
                raise RuntimeError(str(info["infrastructure_error"]))
            receipts.append(BobEvalReceipt(
                evaluation_id=bob_request.evaluation_id,
                proposal_id=proposal.proposal_id,
                evaluation_index=bob_request.evaluation_index,
                bob_policy_version=state.bob_policy_version,
                snapshot_sha256=state.snapshot.state_sha256,
                goal_predicate=dict(returned_goal),
                success=bool(info.get("success")),
                reward=float(info.get("reward", 0.0)),
                step_count=int(info.get("step_count", 0)),
                infrastructure_error=None,
            ))
        return receipts

    def _compile_alice(
        self,
        state: _EpisodeState,
    ) -> tuple[CompiledTask, dict[str, Any], bool]:
        # 中文说明：编译 Alice 题目的核心校验——先记录轨迹产生的终态，然后把
        # 环境恢复到初始快照、逐步回放整条动作序列得到"回放终态"。只有两者
        # 方块位置一致（误差 ≤ replay_tolerance）才说明仿真确定、题目可复现；
        # 之后由 compile_cube_position_task 检查位移量是否达到最小出题门槛，
        # 并额外排除"全程没有真实操作"的无效轨迹。
        generated_final = state.env.task_state()
        state.env.restore_snapshot(state.snapshot)
        state.env.set_goal_position(HIDDEN_ALICE_GOAL)
        for step in state.steps:
            state.env.step(EnvAction(action_type=step["action"], code=step["action"]))
        replay_final = state.env.task_state()
        replay_equal = _distance(
            list(generated_final["cube_position"]),
            list(replay_final["cube_position"]),
        ) <= state.replay_tolerance
        compiled = compile_cube_position_task(
            snapshot=state.snapshot,
            initial_state=state.initial_state,
            final_state=replay_final,
            min_displacement=state.min_goal_displacement,
            tolerance=state.goal_tolerance,
        )
        has_real_operation = any(
            step["valid_action"] and step["action"] not in {"DONE"}
            for step in state.steps
        )
        if not has_real_operation:
            compiled = replace(compiled, valid=False, reason="Alice produced no real operation")
        elif not replay_equal:
            compiled = replace(compiled, valid=False, reason="Alice trajectory replay mismatch")
        return compiled, replay_final, replay_equal

    @staticmethod
    def _termination_reason(state: _EpisodeState, *, success: bool) -> str:
        if success:
            return "goal_success" if state.role == "bob" else "valid_proposal"
        if state.budget_termination_reason:
            return state.budget_termination_reason
        if not state.steps:
            return "empty_trajectory"
        last = state.steps[-1]
        if last["valid_action"] and last["action"] == "DONE":
            return "model_done"
        if last["terminated"]:
            return "environment_terminated"
        if last["truncated"]:
            return "environment_truncated"
        if len(state.steps) >= state.max_steps:
            return "max_steps"
        return "unknown"

    def _failure_info(self, request_id: str, error: str) -> dict[str, Any]:
        return {
            "schema_version": "openeta.embodied_rollout.v1",
            "role": self.role,
            "request_id": request_id,
            "done": True,
            "success": False,
            "reward": 0.0,
            "termination_reason": "infrastructure_error",
            "infrastructure_error": error,
        }

    def _write_artifact(self, state: _EpisodeState) -> None:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", state.request_id)[:160] or "request"
        path = self._artifact_root / state.role / f"{safe_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "openeta.embodied_rollout.v1",
            "request_id": state.request_id,
            "role": state.role,
            "thinking_enabled": state.thinking_enabled,
            "snapshot": state.snapshot.to_dict(),
            "instruction": state.instruction,
            "initial_state": state.initial_state,
            "steps": state.steps,
            "image_hashes": state.image_hashes,
            "final_info": state.final_info,
        }
        # 中文说明：artifact 落盘到 OPENETA_SWIFT_ARTIFACT_ROOT/<role>/<id>.json；
        # 采用"先写临时文件再原子 rename"的方式，避免进程中断留下半个 JSON
        # 被下游误读。
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    async def _close(self, request_id: str, *, mark_finished: bool = False) -> None:
        # 中文说明：释放 episode 资源——关闭仿真环境（隔离模式下同时终止 spawn
        # 子进程），关闭异常用 suppress 吞掉以避免影响其它请求；mark_finished
        # 用于让 check_finished 识别该请求已彻底结束。
        state = self._episodes.pop(request_id, None)
        if state is not None:
            with suppress(Exception):
                state.env.close()
        if mark_finished:
            self._finished.add(request_id)


class EmbodiedBobScheduler(_EmbodiedScheduler):
    role = "bob"


class EmbodiedAliceScheduler(_EmbodiedScheduler):
    role = "alice"


# 中文说明：GRPO 训练侧的 reward 基类。它不自己打分，而是从 scheduler 写入的
# rollout_infos 里递归找出可信的 rollout 记录（schema 标记为
# openeta.embodied_rollout.v1），校验角色匹配后取其 reward 字段；
# 找不到可信记录或角色不符的一律给 0 分，防止模型输出伪造 reward。
class _TrustedRolloutReward(ORM):
    expected_role = ""

    def __call__(self, completions: list[str], **kwargs: Any) -> list[float]:
        infos = kwargs.get("rollout_infos")
        per_completion = infos if isinstance(infos, list) and len(infos) == len(completions) else [infos] * len(completions)
        rewards: list[float] = []
        for value in per_completion:
            info = _find_rollout_info(value)
            if not info or info.get("role") != self.expected_role:
                rewards.append(0.0)
                continue
            rewards.append(float(info.get("reward", 0.0)))
        return rewards


class EmbodiedBobReward(_TrustedRolloutReward):
    expected_role = "bob"


class EmbodiedAliceReward(_TrustedRolloutReward):
    """Trusted Alice validity plus frozen-Bob competence-boundary reward."""

    expected_role = "alice"


# 中文说明：注册入口——ms-swift 通过 --multi_turn_scheduler 与 --reward_funcs
# 按下面的名字查表实例化组件。embodied_alice_proposal_reward 是兼容旧配置的
# 别名，与 embodied_alice_reward 指向同一个类。
multi_turns["embodied_bob_scheduler"] = EmbodiedBobScheduler
multi_turns["embodied_alice_scheduler"] = EmbodiedAliceScheduler
orms["embodied_bob_reward"] = EmbodiedBobReward
orms["embodied_alice_reward"] = EmbodiedAliceReward
# Compatibility alias for datasets/configs produced before frozen-Bob
# evaluation was connected.  It resolves to the same final reward now.
orms["embodied_alice_proposal_reward"] = EmbodiedAliceReward
