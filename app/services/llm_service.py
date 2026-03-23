from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging
from time import perf_counter
import re
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import settings
from app.services.agent_profile_service import (
    build_agent_output_hint,
    get_agent_no_answer_strategy,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMAnswerResult:
    answer: str
    mode: str
    reasoning: str | None = None


@dataclass(frozen=True)
class AnswerStyleProfile:
    style: str
    context_limit: int
    context_chars: int
    max_tokens: int
    temperature_floor: float
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    organization_hint: str


LLM_MAX_OUTPUT_TOKENS = 340

STYLE_PROFILES: dict[str, AnswerStyleProfile] = {
    "direct": AnswerStyleProfile(
        style="direct",
        context_limit=3,
        context_chars=820,
        max_tokens=220,
        temperature_floor=0.12,
        top_p=0.82,
        frequency_penalty=0.2,
        presence_penalty=0.05,
        organization_hint="先直接回答，再补充 1-2 条关键依据，避免冗长铺垫。",
    ),
    "analysis": AnswerStyleProfile(
        style="analysis",
        context_limit=4,
        context_chars=760,
        max_tokens=300,
        temperature_floor=0.18,
        top_p=0.86,
        frequency_penalty=0.25,
        presence_penalty=0.08,
        organization_hint="优先给出判断，再分点解释主要原因与影响。",
    ),
    "comparison": AnswerStyleProfile(
        style="comparison",
        context_limit=4,
        context_chars=720,
        max_tokens=300,
        temperature_floor=0.2,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.12,
        organization_hint="围绕差异维度进行并列对比，再给出选择建议。",
    ),
    "guidance": AnswerStyleProfile(
        style="guidance",
        context_limit=4,
        context_chars=720,
        max_tokens=320,
        temperature_floor=0.22,
        top_p=0.9,
        frequency_penalty=0.3,
        presence_penalty=0.12,
        organization_hint="按可执行步骤输出，每步写清目标与动作。",
    ),
    "summary": AnswerStyleProfile(
        style="summary",
        context_limit=4,
        context_chars=740,
        max_tokens=280,
        temperature_floor=0.16,
        top_p=0.84,
        frequency_penalty=0.24,
        presence_penalty=0.06,
        organization_hint="提炼 3-5 条重点，按主题归并，不要重复句式。",
    ),
}


@lru_cache(maxsize=16)
def get_chat_llm(
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    request_timeout: int,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        timeout=max(6, int(request_timeout)),
        max_retries=0,
    )


def answer_with_llm(
    question: str,
    contexts: list[str],
    llm_enabled: bool | None = None,
    system_instruction: str | None = None,
    agent_key: str | None = None,
    timeout_seconds: int | None = None,
    allow_general_knowledge: bool = False,
    retry_on_failure: bool = True,
    kb_hit: bool | None = None,
    retrieval_contexts: list[str] | None = None,
    background_contexts: list[str] | None = None,
) -> LLMAnswerResult:
    enabled = settings.llm_enabled if llm_enabled is None else llm_enabled
    if not enabled or not settings.api_key:
        return LLMAnswerResult(answer="\n\n".join(contexts), mode="disabled")

    style = _detect_answer_style(question)
    profile = STYLE_PROFILES.get(style, STYLE_PROFILES["direct"])
    raw_retrieval_contexts = (
        list(retrieval_contexts)
        if retrieval_contexts is not None
        else ([] if kb_hit is False else list(contexts))
    )
    compact_retrieval_contexts = _compact_contexts(
        contexts=raw_retrieval_contexts,
        limit=profile.context_limit,
        max_chars=profile.context_chars,
    )
    if kb_hit is None:
        kb_hit = bool(compact_retrieval_contexts)
    if not kb_hit:
        compact_retrieval_contexts = []

    raw_background_contexts = (
        list(background_contexts)
        if background_contexts is not None
        else ([] if retrieval_contexts is None else list(contexts))
    )
    compact_background_contexts = _compact_contexts(
        contexts=raw_background_contexts,
        limit=2,
        max_chars=360,
    )
    agent_hint = build_agent_output_hint(
        agent_key,
        kb_hit=kb_hit,
        allow_general_knowledge=allow_general_knowledge,
    )

    academic_prompt = ""
    if _is_academic_analysis_query(question):
        academic_prompt = (
            "当前问题属于学业分析场景，请按以下维度输出：\n"
            "1) 学业现状（成绩/课程完成）\n"
            "2) 学习行为（课堂互动/学习时长）\n"
            "3) 风险点（最多3条，给出证据）\n"
            "4) 下周行动计划（3条，可执行）\n"
        )

    prompt = _build_structured_prompt(
        question=question,
        system_instruction=system_instruction,
        academic_prompt=academic_prompt,
        style_profile=profile,
        retrieval_contexts=compact_retrieval_contexts,
        background_contexts=compact_background_contexts,
        kb_hit=bool(kb_hit),
        allow_general_knowledge=allow_general_knowledge,
        agent_key=agent_key,
    )

    if allow_general_knowledge and not kb_hit:
        prompt = _build_cloud_direct_prompt(
            question=question,
            compact_contexts=compact_background_contexts,
            system_instruction=system_instruction,
            no_answer_rules=get_agent_no_answer_strategy(agent_key),
        )
    if agent_hint:
        prompt = f"{agent_hint}\n{prompt}"

    effective_timeout = max(
        2,
        int(
            timeout_seconds
            if timeout_seconds is not None
            else settings.llm_timeout_seconds
        ),
    )
    if allow_general_knowledge and not kb_hit:
        # Keep cloud direct-chat mode under the frontend request budget.
        timeout_cap = 30 if not compact_retrieval_contexts else 26
        effective_timeout = min(effective_timeout, timeout_cap)

    effective_temperature = max(settings.llm_temperature, profile.temperature_floor)
    effective_tokens = max(120, min(LLM_MAX_OUTPUT_TOKENS, profile.max_tokens))
    if allow_general_knowledge and not kb_hit:
        effective_tokens = min(effective_tokens, 180)
    if allow_general_knowledge and not compact_retrieval_contexts:
        effective_tokens = min(effective_tokens, 140)

    base_timeout = max(6, effective_timeout)
    call_plan: list[tuple[str, str, int, int]] = [
        ("primary", prompt, base_timeout, effective_tokens)
    ]
    if allow_general_knowledge and retry_on_failure and not kb_hit:
        retry_prompt = _build_compact_retry_prompt(
            question=question,
            compact_contexts=compact_background_contexts,
            system_instruction=system_instruction,
            no_answer_rules=get_agent_no_answer_strategy(agent_key),
        )
        if agent_hint:
            retry_prompt = f"{agent_hint}\n{retry_prompt}"
        retry_timeout = 6
        retry_tokens = max(96, min(180, effective_tokens))
        call_plan.append(("retry", retry_prompt, retry_timeout, retry_tokens))

    timeout_detected = False
    last_exc: Exception | None = None
    for attempt_name, attempt_prompt, attempt_timeout, attempt_tokens in call_plan:
        started = perf_counter()
        try:
            llm = get_chat_llm(
                temperature=round(effective_temperature, 2),
                top_p=round(profile.top_p, 2),
                frequency_penalty=round(profile.frequency_penalty, 2),
                presence_penalty=round(profile.presence_penalty, 2),
                request_timeout=max(6, int(attempt_timeout)),
            )
            response = llm.invoke(
                attempt_prompt,
                max_tokens=attempt_tokens,
            )
            answer = _normalize_answer_text(
                _extract_text_content(getattr(response, "content", ""))
            )
            elapsed_ms = int((perf_counter() - started) * 1000)
            logger.info(
                "llm invoke success: mode=%s timeout=%ss tokens=%s elapsed_ms=%s contexts=%s",
                attempt_name,
                attempt_timeout,
                attempt_tokens,
                elapsed_ms,
                len(compact_retrieval_contexts),
            )
            if not answer or len(answer) < 18:
                continue
            return LLMAnswerResult(
                answer=answer,
                mode="llm" if attempt_name == "primary" else "llm_retry",
                reasoning=_extract_reasoning_text(response),
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((perf_counter() - started) * 1000)
            timeout_like = _looks_like_timeout_error(exc)
            timeout_detected = timeout_detected or timeout_like
            last_exc = exc
            logger.warning(
                "llm invoke failed: mode=%s timeout=%ss elapsed_ms=%s timeout_like=%s err=%s",
                attempt_name,
                attempt_timeout,
                elapsed_ms,
                timeout_like,
                exc.__class__.__name__,
            )
            if timeout_like:
                # Timeout often indicates upstream congestion; avoid stacking waits.
                break
            continue

    if last_exc is not None:
        logger.debug(
            "llm invoke final failure",
            exc_info=(type(last_exc), last_exc, last_exc.__traceback__),
        )

    fallback_mode = "timeout_fallback" if timeout_detected else "error_fallback"
    return LLMAnswerResult(
        answer=_fallback_natural_answer(
            question,
            compact_retrieval_contexts,
            allow_general_knowledge=allow_general_knowledge,
            agent_key=agent_key,
        ),
        mode=fallback_mode,
    )


def _build_structured_prompt(
    *,
    question: str,
    system_instruction: str | None,
    academic_prompt: str,
    style_profile: AnswerStyleProfile,
    retrieval_contexts: list[str],
    background_contexts: list[str],
    kb_hit: bool,
    allow_general_knowledge: bool,
    agent_key: str | None,
) -> str:
    evidence_block = "\n".join(f"- {item}" for item in retrieval_contexts[:4])
    background_block = "\n".join(f"- {item}" for item in background_contexts[:2])
    no_answer_rules = get_agent_no_answer_strategy(agent_key)
    no_answer_hint = "；".join(no_answer_rules[:2]) if no_answer_rules else "明确边界并给下一步。"

    if kb_hit:
        evidence_policy = (
            "知识库已命中：先给结论，再给1-2条关键证据；"
            "证据不足处明确写“暂无法确认”。"
        )
    elif allow_general_knowledge:
        evidence_policy = (
            "知识库未命中：可结合通用知识回答，但需要明确哪些是通用判断。"
        )
    else:
        evidence_policy = (
            "知识库未命中：不编造事实；"
            f"按该智能体无答案策略执行：{no_answer_hint}"
        )

    return (
        f"{(system_instruction or '').strip()}\n"
        f"{academic_prompt}"
        "你是知识库增强问答助手。\n"
        f"{evidence_policy}\n"
        f"回答类型：{style_profile.style}\n"
        f"组织建议：{style_profile.organization_hint}\n"
        "写作要求：\n"
        "1) 第一段直接回答用户问题，不写寒暄。\n"
        "2) 有证据时尽量贴合证据原意，不要泛化改写成空话。\n"
        "3) 无证据时自然说明不足，并给最小可执行建议。\n"
        f"问题：{question}\n\n"
        f"知识库证据：\n{evidence_block or '（未命中高置信知识库证据）'}\n\n"
        f"背景补充（可选）：\n{background_block or '（无额外背景）'}"
    )


def _build_compact_retry_prompt(
    question: str,
    compact_contexts: list[str],
    system_instruction: str | None = None,
    no_answer_rules: tuple[str, ...] = (),
) -> str:
    context_block = ""
    if compact_contexts:
        context_block = "\n".join(f"- {item}" for item in compact_contexts[:2])
    fallback_line = "；".join(no_answer_rules[:2]) if no_answer_rules else "明确边界并给可执行下一步"
    return (
        f"{(system_instruction or '').strip()}\n"
        "你是对话助手，请直接回答用户问题，优先给出可执行结论。"
        f"若信息不足，请{fallback_line}，不要编造事实。\n"
        f"问题：{question}\n"
        f"参考信息：\n{context_block or '（暂无可靠参考信息）'}"
    )


def _build_cloud_direct_prompt(
    question: str,
    compact_contexts: list[str],
    system_instruction: str | None = None,
    no_answer_rules: tuple[str, ...] = (),
) -> str:
    context_block = ""
    if compact_contexts:
        context_block = "\n".join(f"- {item}" for item in compact_contexts[:2])
    fallback_line = "；".join(no_answer_rules[:2]) if no_answer_rules else "自然说明信息不足并给补充建议"
    return (
        f"{(system_instruction or '').strip()}\n"
        "你是对话助手，请像常规聊天模型一样直接回答用户问题，先给结论，再给简要依据。\n"
        f"如果信息不足，请{fallback_line}，不要编造事实。\n"
        f"问题：{question}\n"
        f"可选参考上下文：\n{context_block or '（无额外上下文）'}"
    )


def _looks_like_timeout_error(exc: Exception) -> bool:
    class_name = exc.__class__.__name__.lower()
    if "timeout" in class_name:
        return True
    message = str(exc).lower()
    timeout_markers = [
        "timed out",
        "timeout",
        "read operation timed out",
        "request timed out",
    ]
    return any(marker in message for marker in timeout_markers)


def _fallback_natural_answer(
    question: str,
    contexts: list[str],
    allow_general_knowledge: bool = False,
    agent_key: str | None = None,
) -> str:
    style = _detect_answer_style(question)
    q = (question or "").strip().lower()
    question_focus = (question or "").strip()[:80] or "当前问题"
    no_answer_rules = get_agent_no_answer_strategy(agent_key)
    no_answer_text = "；".join(no_answer_rules[:2]) if no_answer_rules else "补充关键事实后继续"

    if not contexts:
        if allow_general_knowledge:
            return _render_general_knowledge_fallback(style, question_focus)
        return _render_no_context_fallback(style, question_focus, no_answer_text)

    merged = "\n".join(contexts)

    def _is_noise_line(text: str) -> bool:
        t = text.lower()
        if len(text) <= 2:
            return True
        noise_markers = [
            "pip install",
            "npm install",
            "python -m",
            "requirements.txt",
            "localhost:",
            "127.0.0.1",
            "create table",
            "insert into",
            " key `",
            "idx_",
            "外部业务库用户画像",
            "登录名:",
            "角色:",
            "学院/部门:",
            "学号:",
        ]
        return any(marker in t for marker in noise_markers)

    sentences = re.split(r"[\n。！？!?]", merged)
    cleaned: list[str] = []
    for item in sentences:
        text = " ".join(item.strip().split())
        if not text or text in cleaned or _is_noise_line(text):
            continue
        cleaned.append(text)
        if len(cleaned) >= 6:
            break

    if not cleaned:
        if allow_general_knowledge:
            return _render_general_knowledge_fallback(style, question_focus)
        return _render_no_context_fallback(style, question_focus, no_answer_text)

    evidence_lines = cleaned[:4]
    return _render_context_fallback(style, question_focus, evidence_lines, q)


def _is_academic_analysis_query(question: str) -> bool:
    q = (question or "").strip().lower()
    flags = ["学业分析", "学习分析", "成绩分析", "学情分析", "学习情况"]
    return any(flag in q for flag in flags)


def _detect_answer_style(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "direct"
    if any(token in q for token in ["对比", "比较", "区别", "差异", "vs", "优缺点", "哪个好"]):
        return "comparison"
    if any(token in q for token in ["总结", "概述", "重点", "梳理", "归纳", "总览"]):
        return "summary"
    if any(token in q for token in ["怎么", "如何", "步骤", "方案", "计划", "建议", "修复", "排查", "优化"]):
        return "guidance"
    if any(token in q for token in ["为什么", "原因", "分析", "评估", "影响", "风险", "判断", "是否"]):
        return "analysis"
    return "direct"


def _compact_contexts(contexts: list[str], limit: int, max_chars: int) -> list[str]:
    compact: list[str] = []
    seen: set[str] = set()
    for ctx in contexts:
        text = " ".join((ctx or "").split())
        if not text:
            continue
        key = text[:80]
        if key in seen:
            continue
        seen.add(key)
        compact.append(text[:max_chars])
        if len(compact) >= limit:
            break
    return compact


def _normalize_answer_text(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        if cleaned_lines and stripped == cleaned_lines[-1]:
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines).strip()


def _render_no_context_fallback(
    style: str,
    question_focus: str,
    no_answer_text: str,
) -> str:
    if style == "comparison":
        return (
            f"当前还缺少可比依据，暂时不能对“{question_focus}”给出可靠对比结论。\n"
            f"建议：{no_answer_text}。"
        )
    if style == "summary":
        return (
            f"当前资料不足，暂时无法对“{question_focus}”做有效摘要。\n"
            f"建议：{no_answer_text}。"
        )
    if style == "guidance":
        return (
            f"关于“{question_focus}”，当前依据不足，但可以先按这三步推进：\n"
            "1) 明确目标与约束；2) 给出已知事实；3) 我据此输出可执行步骤和风险检查点。"
        )
    if style == "analysis":
        return (
            f"目前缺少支撑“{question_focus}”的关键证据，无法做可靠原因判断。\n"
            f"建议：{no_answer_text}。"
        )
    return (
        f"当前资料不足，暂时无法直接回答“{question_focus}”。\n"
        f"建议：{no_answer_text}。"
    )


def _render_general_knowledge_fallback(style: str, question_focus: str) -> str:
    if style == "comparison":
        return (
            f"关于“{question_focus}”，我先按通用经验给你比较框架：\n"
            "1) 先定评价维度（效果/成本/风险）；2) 再按维度打分；3) 最后按你的优先级做取舍。"
        )
    if style == "summary":
        return (
            f"关于“{question_focus}”，我先给你一个可直接使用的通用摘要结构：\n"
            "背景、核心要点、风险提示、下一步行动。"
        )
    if style == "guidance":
        return (
            f"关于“{question_focus}”，可先按这条通用路径执行：\n"
            "1) 明确目标与约束；2) 拆分可执行步骤；3) 每步设置检查点并滚动调整。"
        )
    if style == "analysis":
        return (
            f"关于“{question_focus}”，我先给你通用分析框架：\n"
            "现象 -> 可能成因 -> 影响范围 -> 验证指标 -> 处置优先级。"
        )
    return (
        f"我先基于通用知识回答“{question_focus}”；"
        "你补充业务上下文后，我可以继续细化到场景级结论。"
    )


def _render_context_fallback(
    style: str,
    question_focus: str,
    evidence_lines: list[str],
    normalized_question: str,
) -> str:
    summary = "；".join(evidence_lines[:4])
    if style == "comparison":
        return (
            f"基于当前命中内容，可先对“{question_focus}”做初步对比：\n"
            f"- 关键依据：{summary}\n"
            "- 若需最终结论，请补充明确的比较对象和决策优先级。"
        )
    if style == "summary":
        bullets = "\n".join(f"- {item}" for item in evidence_lines[:4])
        return f"本轮可提炼的重点如下：\n{bullets}"
    if style == "guidance":
        return (
            f"结合现有资料，建议你围绕“{question_focus}”先执行：\n"
            f"1) 锁定当前优先问题（依据：{evidence_lines[0]}）；\n"
            "2) 制定本周可落地动作并设置检查点；\n"
            "3) 下一轮补充执行结果，我再帮你迭代方案。"
        )
    if style == "analysis" or any(token in normalized_question for token in ["原因", "分析", "风险"]):
        return (
            f"初步判断：该问题可以从已有资料中部分解释。\n"
            f"主要依据：{summary}。\n"
            "如果你希望更深入，我可以继续拆解成“成因-影响-优先级”三层分析。"
        )
    return (
        f"目前能确认的是：{evidence_lines[0]}。\n"
        f"补充依据：{summary}。\n"
        "若需要更精确结论，请补充具体场景或约束条件。"
    )


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                item_type = str(item.get("type", "")).lower()
                if "reason" in item_type or "think" in item_type:
                    continue
                text = _flatten_text(item.get("text") or item.get("content"))
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_reasoning_text(response: Any) -> str | None:
    for container_name in ("additional_kwargs", "response_metadata"):
        container = getattr(response, container_name, None)
        text = _extract_reasoning_from_mapping(container)
        if text:
            return text

    content = getattr(response, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "")).lower()
            if "reason" not in item_type and "think" not in item_type:
                continue
            text = _flatten_text(
                item.get("text")
                or item.get("content")
                or item.get("reasoning")
                or item.get("thinking")
            )
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts).strip()
    return None


def _extract_reasoning_from_mapping(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None

    for key in ("reasoning_content", "reasoning", "thinking", "thoughts"):
        text = _flatten_text(value.get(key))
        if text:
            return text

    for nested in value.values():
        text = _extract_reasoning_from_mapping(nested)
        if text:
            return text
    return None


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_flatten_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "summary"):
            text = _flatten_text(value.get(key))
            if text:
                return text
    return ""
