from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import logging
import socket
from time import perf_counter
import re
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from app.core.config import settings
from app.services.agent_profile_service import (
    build_agent_output_hint,
    get_agent_no_answer_strategy,
    normalize_agent_key,
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


LLM_MAX_OUTPUT_TOKENS = 460

STYLE_PROFILES: dict[str, AnswerStyleProfile] = {
    "direct": AnswerStyleProfile(
        style="direct",
        context_limit=2,
        context_chars=620,
        max_tokens=210,
        temperature_floor=0.12,
        top_p=0.82,
        frequency_penalty=0.2,
        presence_penalty=0.05,
        organization_hint="先直接回答，再补充 1-2 条关键依据，避免冗长铺垫。",
    ),
    "analysis": AnswerStyleProfile(
        style="analysis",
        context_limit=3,
        context_chars=640,
        max_tokens=280,
        temperature_floor=0.18,
        top_p=0.86,
        frequency_penalty=0.25,
        presence_penalty=0.08,
        organization_hint="优先给出判断，再分点解释主要原因与影响。",
    ),
    "comparison": AnswerStyleProfile(
        style="comparison",
        context_limit=3,
        context_chars=620,
        max_tokens=280,
        temperature_floor=0.2,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.12,
        organization_hint="围绕差异维度进行并列对比，再给出选择建议。",
    ),
    "guidance": AnswerStyleProfile(
        style="guidance",
        context_limit=2,
        context_chars=500,
        max_tokens=280,
        temperature_floor=0.2,
        top_p=0.86,
        frequency_penalty=0.3,
        presence_penalty=0.12,
        organization_hint="按可执行步骤输出，每步写清目标与动作。",
    ),
    "course_selection": AnswerStyleProfile(
        style="course_selection",
        context_limit=3,
        context_chars=540,
        max_tokens=260,
        temperature_floor=0.16,
        top_p=0.84,
        frequency_penalty=0.2,
        presence_penalty=0.06,
        organization_hint="先给选课结论，再按要求、时间窗口、限制条件、注意事项说明。",
    ),
    "summary": AnswerStyleProfile(
        style="summary",
        context_limit=3,
        context_chars=620,
        max_tokens=260,
        temperature_floor=0.16,
        top_p=0.84,
        frequency_penalty=0.24,
        presence_penalty=0.06,
        organization_hint="提炼 3-5 条重点，按主题归并，不要重复句式。",
    ),
}


def _agent_role_focus(agent_key: str | None) -> str:
    role_focus = {
        "student-growth": "学习支持与成长行动",
        "teacher-assistant": "教学支持与课堂落地",
        "counselor-ideology": "辅导员思政与学生管理",
        "risk-warning": "学情风险识别与干预",
        "report-assistant": "学情汇总与报告表达",
        "policy-qa": "思政理论与政策解释",
    }
    return role_focus.get(
        normalize_agent_key(agent_key),
        "问题分析与可执行建议",
    )


def _agent_cloud_output_hint(agent_key: str | None, is_complex_query: bool) -> str:
    key = normalize_agent_key(agent_key)
    hints: dict[str, tuple[str, str]] = {
        "student-growth": (
            "优先给可执行学习动作和节奏建议。",
            "可增加阶段目标、风险点和复盘节奏。",
        ),
        "teacher-assistant": (
            "回答围绕教学目标、课堂组织和作业反馈。",
            "可补充课堂流程、评估指标和课后改进建议。",
        ),
        "counselor-ideology": (
            "回答围绕谈心沟通、学生管理和活动组织。",
            "可补充风险分级、沟通步骤和协同机制。",
        ),
        "risk-warning": (
            "回答围绕风险信号、判断依据和干预优先级。",
            "可补充风险等级、触发证据和本周处置动作。",
        ),
        "report-assistant": (
            "回答围绕结论摘要、关键数据和趋势归纳。",
            "可补充口径说明、结论边界和后续建议。",
        ),
        "policy-qa": (
            "回答围绕政策解释、适用范围和执行建议。",
            "可补充条款边界、适用对象与核验路径。",
        ),
    }
    brief, rich = hints.get(
        key,
        (
            "回答聚焦用户问题本身，避免泛化。",
            "可补充关键依据、边界和下一步建议。",
        ),
    )
    return rich if is_complex_query else brief


@lru_cache(maxsize=1)
def _chat_completion_endpoint() -> str:
    return urlparse.urljoin(settings.llm_base_url.rstrip("/") + "/", "chat/completions")


def _invoke_chat_completion_direct(
    *,
    prompt: str,
    temperature: float,
    top_p: float,
    frequency_penalty: float,
    presence_penalty: float,
    max_tokens: int,
    request_timeout: int,
) -> tuple[str, str | None]:
    payload = {
        "model": settings.llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "max_tokens": max_tokens,
        "stream": False,
    }
    request = urlrequest.Request(
        _chat_completion_endpoint(),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(
            request, timeout=max(6, int(request_timeout))
        ) as response:
            raw = response.read().decode("utf-8")
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else ""
        raise RuntimeError(f"http_{exc.code}:{detail[:240]}") from exc
    except (urlerror.URLError, TimeoutError, socket.timeout) as exc:
        raise TimeoutError(str(exc)) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid_json:{raw[:240]}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"empty_choices:{raw[:240]}")
    message = choices[0].get("message") or {}
    answer = _normalize_answer_text(_flatten_text(message.get("content")))
    reasoning = _extract_reasoning_from_mapping(
        message
    ) or _extract_reasoning_from_mapping(choices[0])
    return answer, reasoning


def _compact_instruction_text(text: str | None, max_chars: int = 560) -> str:
    compact = " ".join((text or "").split()).strip()
    if not compact:
        return ""
    return compact[:max_chars]


def _sanitize_context_line(raw_text: str, max_chars: int = 128) -> str:
    text = re.sub(r"\s+", " ", (raw_text or "").strip())
    if not text:
        return ""
    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    text = re.sub(r"^#{1,6}\s*", "", text)
    text = re.sub(r"\s*#{1,6}\s*$", "", text)
    text = re.sub(r"^[-*•]\s*", "", text)
    text = re.sub(r"^\d+[\.、)\]]\s*", "", text)
    text = re.sub(r"^[一二三四五六七八九十0-9]+[、.)．]\s*", "", text)
    text = re.sub(r"^第[一二三四五六七八九十0-9]+[章节条点、.)]\s*", "", text)
    text = text.replace("`", "")
    text = text.strip(" ：:;；,，。-")
    if len(text) < 6:
        return ""
    if (
        len(text) <= 24
        and not re.search(r"[，。；：:,.!?？！]", text)
        and any(
            token in text for token in ("概述", "总结", "目录", "要点", "说明", "资源")
        )
    ):
        return ""
    return text[:max_chars]


def _collect_fallback_evidence(contexts: list[str], max_items: int = 3) -> list[str]:
    evidence: list[str] = []
    for ctx in contexts[:4]:
        for segment in str(ctx or "").replace("\r\n", "\n").split("\n"):
            line = _sanitize_context_line(segment, max_chars=140)
            if not line:
                continue
            lowered = line.lower()
            if any(
                marker in lowered
                for marker in (
                    "pip install",
                    "npm install",
                    "localhost",
                    "127.0.0.1",
                    "create table",
                    "insert into",
                    "[会话背景]",
                    "[用户画像]",
                    "近期用户问题",
                    "近期助手回答",
                    "对话上下文摘要",
                )
            ):
                continue
            if line in evidence:
                continue
            evidence.append(line)
            if len(evidence) >= max_items:
                return evidence
    return evidence


def _build_timeout_guided_fallback(
    *,
    question: str,
    contexts: list[str],
    allow_general_knowledge: bool,
    agent_key: str | None,
) -> str:
    focus = " ".join((question or "").split()).strip()[:88] or "当前问题"
    evidence_lines = _collect_fallback_evidence(contexts, max_items=3)
    if not evidence_lines:
        return _fallback_natural_answer(
            question=question,
            contexts=contexts,
            allow_general_knowledge=allow_general_knowledge,
            agent_key=agent_key,
        )

    style = _detect_answer_style(question)
    complexity_mode = _detect_question_complexity(question, style)
    is_complex_query = complexity_mode == "complex"
    if not is_complex_query:
        best_evidence = evidence_lines[0] if evidence_lines else ""
        if style == "course_selection":
            return _apply_agent_fallback_tone(
                _render_course_selection_fallback(focus, best_evidence[:120]),
                agent_key,
            )
        if best_evidence:
            return _apply_agent_fallback_tone(
                (
                    f"先给你一个直接结论：围绕“{focus}”，目前更稳妥的做法是优先按已知线索推进，"
                    f"核心依据是“{best_evidence}”。"
                    "如果你愿意，我可以继续补充更细的原因和下一步执行清单。"
                ),
                agent_key,
            )
        return _apply_agent_fallback_tone(
            (
                f"围绕“{focus}”，我先给你一个可执行的短结论：先做一项最小可行动作并记录结果，"
                "再根据结果迭代下一步。"
            ),
            agent_key,
        )

    if style == "guidance" and _looks_like_stress_conflict_query(question):
        best_evidence = evidence_lines[0] if evidence_lines else ""
        return _apply_agent_fallback_tone(
            _render_stress_conflict_fallback(focus, best_evidence[:120]),
            agent_key,
        )

    lead_map = {
        "comparison": "先给你可执行结论",
        "analysis": "先给你关键判断",
        "guidance": "先给你可落地方案",
        "course_selection": "先给你选课要求",
        "summary": "先给你核心结论",
    }
    lead = lead_map.get(style, "先给你直接结论")
    evidence_block = "\n".join(f"- {item}" for item in evidence_lines[:3])
    no_answer_rules = get_agent_no_answer_strategy(agent_key)
    no_answer_hint = (
        "；".join(no_answer_rules[:2])
        if no_answer_rules
        else "补充对象、时间范围和约束条件后，我可以给你更精确方案"
    )
    if allow_general_knowledge:
        tail = "我先基于现有线索给你稳妥方案；你补充场景后我会继续细化。"
    else:
        tail = f"当前仅以可核验线索为准；若要更精确结论，建议：{no_answer_hint}。"
    return _apply_agent_fallback_tone(
        (
            f"**结论**\n{lead}：围绕“{focus}”，先按当前高置信线索执行，不要一次铺开太多动作。\n\n"
            f"**依据**\n{evidence_block}\n\n"
            "**行动建议**\n"
            "1) 先完成一项可验证动作；\n"
            "2) 记录结果与阻碍；\n"
            "3) 基于结果再做下一轮细化。\n\n"
            f"{tail}"
        ),
        agent_key,
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
    complexity_mode = _detect_question_complexity(question, style)
    is_complex_query = complexity_mode == "complex"
    role_focus = _agent_role_focus(agent_key)
    role_cloud_hint = _agent_cloud_output_hint(agent_key, is_complex_query)
    raw_retrieval_contexts = (
        list(retrieval_contexts)
        if retrieval_contexts is not None
        else ([] if kb_hit is False else list(contexts))
    )
    retrieval_limit = (
        profile.context_limit if is_complex_query else max(1, profile.context_limit - 1)
    )
    retrieval_chars = (
        profile.context_chars
        if is_complex_query
        else max(360, profile.context_chars - 220)
    )
    compact_retrieval_contexts = _compact_contexts(
        contexts=raw_retrieval_contexts,
        limit=retrieval_limit,
        max_chars=retrieval_chars,
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
        limit=2 if is_complex_query else 1,
        max_chars=360 if is_complex_query else 220,
    )
    system_instruction_text = _compact_instruction_text(system_instruction)
    if allow_general_knowledge and role_cloud_hint:
        system_instruction_text = _compact_instruction_text(
            f"{system_instruction_text} {role_cloud_hint}".strip(),
            max_chars=620,
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

    open_answer_mode = allow_general_knowledge and not _is_academic_analysis_query(
        question
    )
    if open_answer_mode:
        prompt = _build_open_answer_prompt(
            question=question,
            style_profile=profile,
            background_contexts=compact_background_contexts,
            retrieval_contexts=compact_retrieval_contexts,
            system_instruction=system_instruction_text,
            is_complex_query=is_complex_query,
            role_focus=role_focus,
            role_cloud_hint=role_cloud_hint,
        )
    else:
        prompt = _build_structured_prompt(
            question=question,
            system_instruction=system_instruction_text,
            academic_prompt=academic_prompt,
            style_profile=profile,
            retrieval_contexts=compact_retrieval_contexts,
            background_contexts=compact_background_contexts,
            kb_hit=bool(kb_hit),
            allow_general_knowledge=allow_general_knowledge,
            agent_key=agent_key,
            is_complex_query=is_complex_query,
        )

    if allow_general_knowledge and not kb_hit and not open_answer_mode:
        prompt = _build_cloud_direct_prompt(
            question=question,
            compact_contexts=compact_background_contexts,
            system_instruction=system_instruction_text,
            no_answer_rules=get_agent_no_answer_strategy(agent_key),
            is_complex_query=is_complex_query,
            role_focus=role_focus,
            role_cloud_hint=role_cloud_hint,
        )
    cloud_direct_fast_mode = allow_general_knowledge and not kb_hit
    if agent_hint and not cloud_direct_fast_mode:
        prompt = f"{agent_hint}\n{prompt}"

    effective_timeout = max(
        2,
        int(
            timeout_seconds
            if timeout_seconds is not None
            else settings.llm_timeout_seconds
        ),
    )
    # Keep the full caller-provided timeout budget for all question modes.

    effective_temperature = max(settings.llm_temperature, profile.temperature_floor)
    if is_complex_query:
        token_growth = 56 if open_answer_mode else 36
        effective_tokens = max(
            160,
            min(LLM_MAX_OUTPUT_TOKENS, profile.max_tokens + token_growth),
        )
    else:
        effective_tokens = max(
            96,
            min(LLM_MAX_OUTPUT_TOKENS, min(profile.max_tokens, 170)),
        )
    if open_answer_mode:
        effective_tokens = min(effective_tokens, 380 if is_complex_query else 230)
    if allow_general_knowledge and not kb_hit:
        effective_tokens = min(effective_tokens, 320 if is_complex_query else 180)
    if allow_general_knowledge and not compact_retrieval_contexts:
        effective_tokens = min(effective_tokens, 300 if is_complex_query else 165)
    if (
        cloud_direct_fast_mode
        and not is_complex_query
        and not compact_background_contexts
    ):
        effective_tokens = min(effective_tokens, 150)

    base_timeout = max(6, effective_timeout)
    retry_gate = 10 if cloud_direct_fast_mode else 14
    if retry_on_failure and base_timeout >= retry_gate:
        # Cloud direct mode benefits from a longer primary attempt to avoid premature fallback.
        if allow_general_knowledge and not kb_hit:
            retry_reserved_timeout = min(4, max(2, base_timeout // 8))
        else:
            retry_reserved_timeout = min(10, max(5, base_timeout // 5))
    else:
        retry_reserved_timeout = 0
    primary_timeout = max(
        5 if cloud_direct_fast_mode else 6, base_timeout - retry_reserved_timeout
    )
    call_plan: list[tuple[str, str, int, int]] = [
        ("primary", prompt, primary_timeout, effective_tokens)
    ]
    if retry_reserved_timeout > 0:
        retry_timeout = retry_reserved_timeout
        retry_contexts = (
            compact_retrieval_contexts[:1] + compact_background_contexts[:1]
        )[:2]
        retry_prompt = _build_compact_retry_prompt(
            question=question,
            compact_contexts=retry_contexts,
            system_instruction=system_instruction_text,
            no_answer_rules=get_agent_no_answer_strategy(agent_key),
        )
        if agent_hint and not cloud_direct_fast_mode:
            retry_prompt = f"{agent_hint}\n{retry_prompt}"
        retry_tokens = max(96, min(200, max(120, effective_tokens - 20)))
        call_plan.append(("retry_compact", retry_prompt, retry_timeout, retry_tokens))

    timeout_detected = False
    last_exc: Exception | None = None
    for attempt_index, (
        attempt_name,
        attempt_prompt,
        attempt_timeout,
        attempt_tokens,
    ) in enumerate(call_plan):
        started = perf_counter()
        try:
            answer, reasoning = _invoke_chat_completion_direct(
                prompt=attempt_prompt,
                temperature=round(effective_temperature, 2),
                top_p=round(profile.top_p, 2),
                frequency_penalty=round(profile.frequency_penalty, 2),
                presence_penalty=round(profile.presence_penalty, 2),
                request_timeout=max(6, int(attempt_timeout)),
                max_tokens=attempt_tokens,
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
                reasoning=reasoning,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((perf_counter() - started) * 1000)
            timeout_like = _looks_like_timeout_error(exc)
            timeout_detected = timeout_detected or timeout_like
            last_exc = exc
            if timeout_like:
                logger.info(
                    "llm invoke timeout: mode=%s timeout=%ss elapsed_ms=%s",
                    attempt_name,
                    attempt_timeout,
                    elapsed_ms,
                )
            else:
                logger.warning(
                    "llm invoke failed: mode=%s timeout=%ss elapsed_ms=%s err=%s",
                    attempt_name,
                    attempt_timeout,
                    elapsed_ms,
                    exc.__class__.__name__,
                )
            has_more_attempts = attempt_index < len(call_plan) - 1
            if timeout_like and not has_more_attempts:
                break
            continue

    if last_exc is not None:
        logger.debug(
            "llm invoke final failure",
            exc_info=(type(last_exc), last_exc, last_exc.__traceback__),
        )

    fallback_mode = "timeout_fallback" if timeout_detected else "error_fallback"
    fallback_contexts: list[str] = []
    if compact_background_contexts:
        fallback_contexts.extend(compact_background_contexts[:2])
    if compact_retrieval_contexts:
        fallback_contexts.extend(compact_retrieval_contexts[:1])
    if timeout_detected:
        fallback_answer = _build_timeout_guided_fallback(
            question=question,
            contexts=fallback_contexts,
            allow_general_knowledge=allow_general_knowledge,
            agent_key=agent_key,
        )
    else:
        fallback_answer = _fallback_natural_answer(
            question,
            fallback_contexts,
            allow_general_knowledge=allow_general_knowledge,
            agent_key=agent_key,
        )
    return LLMAnswerResult(
        answer=fallback_answer,
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
    is_complex_query: bool,
) -> str:
    evidence_block = "\n".join(f"- {item}" for item in retrieval_contexts[:3])
    background_block = "\n".join(f"- {item}" for item in background_contexts[:2])
    no_answer_rules = get_agent_no_answer_strategy(agent_key)
    no_answer_hint = (
        "；".join(no_answer_rules[:2]) if no_answer_rules else "明确边界并给下一步。"
    )

    if kb_hit:
        if style_profile.style == "course_selection":
            evidence_policy = (
                "知识库已命中：先直接给出选课要求，再按时间窗口、限制条件、注意事项组织回答；"
                "综合2-3条证据，不要照抄表格或原文。"
            )
        else:
            evidence_policy = (
                "知识库已命中：先独立给出判断，再用1条证据做校验；不要复读证据原文。"
            )
    elif allow_general_knowledge:
        evidence_policy = "知识库未命中：可结合通用知识回答，但需要明确哪些是通用判断。"
    else:
        evidence_policy = (
            f"知识库未命中：不编造事实；按该智能体无答案策略执行：{no_answer_hint}"
        )

    if style_profile.style == "course_selection":
        complexity_instruction = "选课/要求类问题：用清晰编号回答，至少覆盖要求、时间窗口、限制条件、注意事项。"
        length_instruction = "篇幅：控制在 4-6 行，优先给可执行清单，不要复读原文。"
    else:
        complexity_instruction = (
            "复杂问题：使用自然小标题或编号，至少覆盖结论、关键依据、风险边界、执行步骤；内容要具体，避免空泛。"
            if is_complex_query
            else "简单问题：直接自然回答，不要强行套固定三段式；1-3段即可说清楚。"
        )
        length_instruction = (
            "篇幅：更详细完整，可适度展开。"
            if is_complex_query
            else "篇幅：尽量简洁，避免重复和冗余铺垫。"
        )
    return (
        f"{(system_instruction or '').strip()}\n"
        f"{academic_prompt}"
        "你是知识库增强问答助手。\n"
        "请使用与用户提问一致的语言回答。\n"
        f"{evidence_policy}\n"
        f"{complexity_instruction}\n"
        f"{length_instruction}\n"
        "写作要求：\n"
        "1) 第一段直接回答问题，不写客套话。\n"
        "2) 以分析和判断为主，不要把回答写成机械摘要。\n"
        "3) 证据只在必要时引用，避免逐条复述原文。\n"
        "4) 信息不足时明确边界，并给最小可执行下一步。\n"
        f"问题：{question}\n\n"
        f"知识库证据：\n{evidence_block or '（未命中高置信证据）'}\n\n"
        f"背景补充（可选）：\n{background_block or '（无额外背景）'}"
    )


def _build_open_answer_prompt(
    *,
    question: str,
    style_profile: AnswerStyleProfile,
    background_contexts: list[str],
    retrieval_contexts: list[str],
    system_instruction: str | None,
    is_complex_query: bool,
    role_focus: str = "",
    role_cloud_hint: str = "",
) -> str:
    background_block = "\n".join(f"- {item}" for item in background_contexts[:2])
    evidence_hint = retrieval_contexts[0] if retrieval_contexts else ""
    evidence_block = (
        f"\n可选参考线索（仅在需要时使用）：\n- {evidence_hint}"
        if evidence_hint
        else ""
    )
    organization_hint = _compact_instruction_text(
        style_profile.organization_hint, max_chars=120
    )

    if is_complex_query:
        depth_instruction = "复杂问题：给出完整结论、关键依据、风险边界与可执行步骤。"
        length_instruction = "篇幅建议：260-460 字左右，信息完整但避免冗长。"
        structure_instruction = "可使用清晰结构，如“结论/依据/行动建议”。"
    else:
        depth_instruction = "简单问题：直接回答，再给一个可执行下一步。"
        length_instruction = "篇幅建议：90-170 字左右。"
        structure_instruction = "不要强行套模板，用 1-2 段自然表达即可。"

    style_hint = f"组织偏好：{organization_hint}\n" if organization_hint else ""
    role_focus_line = f"角色焦点：{role_focus}\n" if role_focus else ""
    role_hint_line = f"角色补充：{role_cloud_hint}\n" if role_cloud_hint else ""
    return (
        f"{(system_instruction or '').strip()}\n"
        "你处于云端直答模式，请直接回答用户问题。\n"
        "与用户提问保持同语种输出。\n"
        "不要暴露内部策略、模板标签或分类过程。\n"
        "不要机械复述资料原文，优先自然整合表达。\n"
        f"{role_focus_line}"
        f"{role_hint_line}"
        f"{style_hint}"
        f"{depth_instruction}\n"
        f"{length_instruction}\n"
        f"{structure_instruction}\n"
        "表达自然专业，避免空话和重复句。\n"
        f"问题：{question}\n\n"
        f"背景：\n{background_block or '（无）'}"
        f"{evidence_block}"
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
    fallback_line = (
        "；".join(no_answer_rules[:2])
        if no_answer_rules
        else "明确边界并给可执行下一步"
    )
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
    is_complex_query: bool = False,
    role_focus: str = "",
    role_cloud_hint: str = "",
) -> str:
    context_block = ""
    if compact_contexts:
        context_block = "\n".join(f"- {item}" for item in compact_contexts[:2])
    fallback_line = (
        "；".join(no_answer_rules[:2])
        if no_answer_rules
        else "自然说明信息不足并给补充建议"
    )
    depth_instruction = (
        "复杂问题：给出完整结论、关键依据和可执行动作。"
        if is_complex_query
        else "简单问题：直接回答并保持简洁。"
    )
    length_instruction = (
        "篇幅建议：复杂问题约 220-380 字。"
        if is_complex_query
        else "篇幅建议：简单问题约 90-160 字。"
    )
    role_focus_line = f"角色焦点：{role_focus}\n" if role_focus else ""
    role_hint_line = f"角色补充：{role_cloud_hint}\n" if role_cloud_hint else ""
    return (
        f"{(system_instruction or '').strip()}\n"
        "你处于云端直答模式。\n"
        "与用户提问保持同语种输出。\n"
        "表达简洁自然，不暴露内部流程或模板标签。\n"
        f"{role_focus_line}"
        f"{role_hint_line}"
        f"{depth_instruction}\n"
        f"{length_instruction}\n"
        f"若信息不足，{fallback_line}，不要编造事实。\n"
        f"问题：{question}\n"
        f"可选上下文：\n{context_block or '（无）'}"
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


def _apply_agent_fallback_tone(answer: str, agent_key: str | None) -> str:
    normalized_agent = normalize_agent_key(agent_key)
    addon_map = {
        "student-growth": "如你愿意补充近期学习状态和目标，我可以继续给你更具体的行动清单。",
        "teacher-assistant": "如你补充课程目标、课时和学生基础，我可以进一步细化成可直接执行的教学方案。",
        "counselor-ideology": "如你补充事件背景和风险线索，我可以继续细化沟通与协同处置步骤。",
        "risk-warning": "如你补充对象范围、时间窗口和异常指标，我可以继续细化风险分级与干预优先级。",
        "report-assistant": "如你补充统计口径和时间范围，我可以继续输出更完整的报告版本。",
        "policy-qa": "如你补充适用场景和对象范围，我可以继续给出更准确的条款边界说明。",
    }
    addon = addon_map.get(normalized_agent, "")
    if not addon:
        return answer
    compact = (answer or "").strip()
    if not compact:
        return addon
    if addon in compact or len(compact) >= 620:
        return compact
    return f"{compact}\n{addon}"


def _fallback_natural_answer(
    question: str,
    contexts: list[str],
    allow_general_knowledge: bool = False,
    agent_key: str | None = None,
) -> str:
    style = _detect_answer_style(question)
    complexity_mode = _detect_question_complexity(question, style)
    is_complex_query = complexity_mode == "complex"
    q = (question or "").strip().lower()
    question_focus = (question or "").strip()[:80] or "当前问题"
    no_answer_rules = get_agent_no_answer_strategy(agent_key)
    no_answer_text = (
        "；".join(no_answer_rules[:2]) if no_answer_rules else "补充关键事实后继续"
    )

    if not contexts:
        if allow_general_knowledge:
            base = _render_general_knowledge_fallback(
                style,
                question_focus,
                is_complex_query=is_complex_query,
            )
        else:
            base = _render_no_context_fallback(
                style,
                question_focus,
                no_answer_text,
                q,
                is_complex_query=is_complex_query,
            )
        return _apply_agent_fallback_tone(base, agent_key)

    evidence_lines = _collect_fallback_evidence(contexts, max_items=4)
    if not evidence_lines:
        if allow_general_knowledge:
            base = _render_general_knowledge_fallback(
                style,
                question_focus,
                is_complex_query=is_complex_query,
            )
        else:
            base = _render_no_context_fallback(
                style,
                question_focus,
                no_answer_text,
                q,
                is_complex_query=is_complex_query,
            )
        return _apply_agent_fallback_tone(base, agent_key)

    if allow_general_knowledge and is_complex_query:
        evidence_block = "\n".join(f"- {item}" for item in evidence_lines[:3])
        base = (
            f"**结论**\n围绕“{question_focus}”，先执行最可能有效的一步，并用结果驱动下一轮调整。\n\n"
            f"**依据**\n{evidence_block}\n\n"
            "**行动建议**\n"
            "1) 先完成一项可验证动作；\n"
            "2) 记录结果和障碍；\n"
            "3) 带着结果继续追问，我会给你更精准的下一步方案。"
        )
        return _apply_agent_fallback_tone(base, agent_key)

    base = _render_context_fallback(
        style,
        question_focus,
        evidence_lines,
        q,
        is_complex_query=is_complex_query,
    )
    return _apply_agent_fallback_tone(base, agent_key)


def _is_academic_analysis_query(question: str) -> bool:
    q = (question or "").strip().lower()
    flags = ["学业分析", "学习分析", "成绩分析", "学情分析", "学习情况"]
    return any(flag in q for flag in flags)


def _detect_answer_style(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "direct"
    if _looks_like_course_selection_query(q):
        return "course_selection"
    if _looks_like_club_overload_query(q):
        return "guidance"
    if _looks_like_doctoral_extension_query(q):
        return "guidance"
    if _looks_like_learning_support_query(q):
        return "guidance"
    if _looks_like_decision_guidance_query(q):
        return "guidance"
    if any(
        token in q
        for token in ["对比", "比较", "区别", "差异", "vs", "优缺点", "哪个好"]
    ):
        return "comparison"
    if any(
        token in q
        for token in [
            "怎么",
            "如何",
            "步骤",
            "方案",
            "计划",
            "建议",
            "修复",
            "排查",
            "优化",
        ]
    ):
        return "guidance"
    if any(token in q for token in ["总结", "概述", "重点", "梳理", "归纳", "总览"]):
        return "summary"
    if any(
        token in q
        for token in ["为什么", "原因", "分析", "评估", "影响", "风险", "判断", "是否"]
    ):
        return "analysis"
    return "direct"


def _detect_question_complexity(question: str, style: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "simple"
    if style in {"analysis", "comparison"}:
        return "complex"
    punctuation_score = q.count("，") + q.count(",") + q.count("；") + q.count(";")
    complex_markers = [
        "分析",
        "对比",
        "比较",
        "方案",
        "规划",
        "步骤",
        "风险",
        "策略",
        "路径",
        "优缺点",
        "取舍",
        "why",
        "how",
        "plan",
        "vs",
    ]
    if any(marker in q for marker in complex_markers):
        return "complex"
    if style == "guidance" and (len(q) >= 26 or punctuation_score >= 2):
        return "complex"
    if len(q) >= 54 or punctuation_score >= 3:
        return "complex"
    return "simple"


def _looks_like_decision_guidance_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    decision_markers = ["还是", "要不要", "选择", "纠结", "不知道"]
    path_markers = ["跨考", "考研", "读研", "实习", "就业", "方向", "大厂"]
    return any(marker in q for marker in decision_markers) and any(
        marker in q for marker in path_markers
    )


def _looks_like_stress_conflict_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    stress_markers = ["焦虑", "熬夜", "效率低", "看不进", "崩溃", "很累"]
    conflict_markers = ["期末", "考试", "换届", "社团", "撞在一起", "冲突"]
    return any(marker in q for marker in stress_markers) and any(
        marker in q for marker in conflict_markers
    )


def _looks_like_learning_support_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    support_markers = [
        "挂科",
        "重修",
        "辅导",
        "补习",
        "答疑",
        "学霸",
        "帮帮",
        "官方资源",
        "学习资源",
        "过不了",
        "40多分",
        "40 多分",
        "听天书",
    ]
    return any(marker in q for marker in support_markers)


def _looks_like_doctoral_extension_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    doctoral_markers = ["博士", "直博", "博士生"]
    duration_markers = ["最长", "几年", "延期", "修业年限", "毕业", "发够文章"]
    stress_markers = ["焦虑", "掉头发", "进展不顺", "根本不可能"]
    return (
        any(m in q for m in doctoral_markers)
        and any(m in q for m in duration_markers)
        and any(m in q for m in stress_markers)
    )


def _looks_like_club_overload_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    org_markers = ["学生会", "社团", "开会", "策划案", "退部", "学长学姐"]
    overload_markers = ["没时间", "绑架", "作业", "图书馆", "不好意思", "平衡"]
    return any(m in q for m in org_markers) and any(m in q for m in overload_markers)


def _looks_like_service_process_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if _looks_like_course_selection_query(question):
        return False
    service_tokens = [
        "挂失",
        "补办",
        "办理",
        "流程",
        "步骤",
        "校园卡",
        "一卡通",
        "饭卡",
        "门禁卡",
        "学生证",
        "请假",
        "缴费",
    ]
    return any(token in q for token in service_tokens)


def _looks_like_course_selection_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if "选课" in q:
        return True
    course_tokens = ["课程", "学分", "绩点", "培养方案", "先修", "修读"]
    requirement_tokens = ["要求", "规则", "限制", "时间", "窗口", "容量"]
    return any(token in q for token in course_tokens) and any(
        token in q for token in requirement_tokens
    )


def _render_course_selection_fallback(
    question_focus: str, evidence_hint: str = ""
) -> str:
    hint = f"\n可参考线索：{evidence_hint}。" if evidence_hint else ""
    return (
        f"先直接答复：围绕“{question_focus}”，选课前要先核对培养方案要求，再确认学分限制、先修条件和选课时间窗口。\n"
        "1) 先看培养方案：确认本学期必修、限选、任选和建议修读顺序；\n"
        "2) 再查系统限制：核对学分上下限、先修课程、时间冲突和课程容量；\n"
        "3) 提交前复核：确认退补改时间、是否需要学院审批、是否影响后续课程修读；\n"
        "4) 如仍拿不准，优先联系学院教务老师或辅导员核实。"
        f"{hint}"
    )


def _render_decision_guidance_fallback(question_focus: str) -> str:
    return (
        f"先直接回答：对于“{question_focus}”，先不要急着做一次性押注，建议用 2-4 周做双轨验证。\n"
        "1) 路线A：拿出固定时间验证更长期的方向，例如跨考或转向新领域；\n"
        "2) 路线B：同步投递能尽快得到市场反馈的实习/项目；\n"
        "3) 每周记录投入时长、完成度、反馈强弱；\n"
        "4) 第4周按真实反馈做决定：哪条路更可持续、反馈更正向，就先押哪条。"
    )


def _render_stress_conflict_fallback(question_focus: str) -> str:
    return (
        f"先直接回答：像“{question_focus}”这种期末和社团事务撞车的情况，你现在最需要的不是继续硬扛，而是先止损，把睡眠和任务优先级拉回可控。\n"
        "1) 今晚先做止损：不要再熬到很晚，只保留1个最关键学习任务，剩下的全部顺延；\n"
        "2) 明天先拆任务：把事情分成“必须亲自做、可以委托、可以延后”三类，社团换届里能交接的立刻交接；\n"
        "3) 复习只保主线：每次只学45分钟，先抓最可能影响期末结果的课程和题型，不追求全做完；\n"
        "4) 给自己一个硬标准：连续两天睡眠不足或效率继续下滑，就必须减少社团投入，优先保考试。"
    )


def _render_learning_support_fallback(
    question_focus: str, evidence_hint: str = ""
) -> str:
    hint = f"\n可参考线索：{evidence_hint}。" if evidence_hint else ""
    return (
        f"先直接回答：像“{question_focus}”这种已经出现明显挂科风险的情况，不建议你继续一个人死磕，最好马上把“老师答疑 + 学校辅导资源 + 同伴帮扶”三条线同时拉起来。\n"
        "1) 先找任课教师或助教：这周就去问清楚期中失分点、期末重点和补救顺序；\n"
        "2) 再找学院或辅导员：直接问有没有官方学业帮扶、答疑安排、朋辈辅导或补习资源；\n"
        "3) 同步找一位学得好的同学或学长学姐，先带你补最基础的章节和题型，不要自己从头乱补；\n"
        "4) 接下来两周只做一件事：把最可能决定期末及格的核心知识点和高频题型补起来。"
        f"{hint}"
    )


def _render_doctoral_extension_fallback(
    question_focus: str, evidence_hint: str = ""
) -> str:
    hint = f"\n可参考线索：{evidence_hint}。" if evidence_hint else ""
    return (
        f"先直接回答：像“{question_focus}”这种博士一年级就担心延期和毕业的情况，不代表你不适合读博，更不代表现在就能下结论说自己一定毕不了业。你现在最该做的，是先把“最长修业年限 + 导师预期 + 本学期最小可交付成果”这三件事弄清楚。\n"
        "1) 先问学院研究生秘书或培养办：确认博士最长修业年限、延期条件和毕业基本要求；\n"
        "2) 再和导师谈一次：把卡住你的具体问题、已有尝试、接下来3个月目标说清楚，不要只说自己焦虑；\n"
        "3) 把目标从“几年内发够文章”改成“先做出一个可验证的小结果”，先恢复研究节奏；\n"
        "4) 如果已经焦虑到明显影响睡眠、饮食或身体状态，就尽快找校内心理咨询/辅导员支持，不要硬扛。"
        f"{hint}"
    )


def _render_club_overload_fallback(question_focus: str, evidence_hint: str = "") -> str:
    hint = f"\n可参考线索：{evidence_hint}。" if evidence_hint else ""
    return (
        f"先直接回答：像“{question_focus}”这种情况，真正要优先保的是学业，你现在不是“得罪人”，而是在给自己恢复边界。\n"
        "1) 先做取舍：学生会和两个大社团不可能长期同时高投入，至少要砍掉一项核心承诺；\n"
        "2) 不要突然失联，直接和负责人说“开学初判断失误，学业已经明显受影响，需要从高频事务中退出或降频”；\n"
        "3) 先提出替代方案：把手头任务交接清楚、给出过渡时间，这样比硬拖到彻底崩掉更负责；\n"
        "4) 从这周开始固定晚间两到三个时段只留给上课、自习和作业，社团活动只能占剩余时间。"
        f"{hint}"
    )


def _render_service_process_fallback(
    question_focus: str, evidence_hint: str = ""
) -> str:
    hint = f"\n可参考线索：{evidence_hint}。" if evidence_hint else ""
    return (
        f"先直接答复：像“{question_focus}”这类校园事务，优先先挂失/冻结，再按学校流程补办或线下处理。\n"
        "1) 先做止损：立刻挂失，避免继续被消费或误用；\n"
        "2) 再找入口：优先查学校一卡通平台、校园服务大厅或相关服务窗口；\n"
        "3) 如需补办，确认要不要带证件、是否缴费、去哪里领取；\n"
        "4) 如果线上找不到，直接联系辅导员或校园卡服务点确认最快处理路径。"
        f"{hint}"
    )


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
    text = (answer or "").replace("\r\n", "\n").strip()
    if not text:
        return ""
    text = re.sub(r"(?is)<think>.*?</think>", "", text).strip()
    text = re.sub(r"(?is)<system-reminder>.*?</system-reminder>", "", text).strip()
    text = re.sub(r"(?m)^#{1,6}\s*([一二三四五六七八九十0-9]+[、.)．]\s*)", "", text)
    text = re.sub(
        r"(?m)^#{1,6}\s*(结论|依据|行动建议|建议|补充说明)\s*$", r"**\1**", text
    )
    text = re.sub(
        r"(?m)^(处理摘要|系统策略|内部模板|内部分类|处理步骤)\s*[:：].*$", "", text
    )
    if re.search(r"(?i)thinking process\s*:", text):
        public_match = re.search(
            r"(?m)^(?:\*\*)?(结论|直接回答|答案|依据|行动建议|补充说明)(?:\*\*)?\s*$",
            text,
        )
        if public_match:
            text = text[public_match.start() :].strip()
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
    normalized_question: str,
    is_complex_query: bool = True,
) -> str:
    if not is_complex_query:
        return (
            f"关于“{question_focus}”，当前可用信息还不够完整。"
            f"先给你一个直接建议：{no_answer_text}。"
        )
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
    if style == "course_selection":
        return _render_course_selection_fallback(question_focus)
    if style == "guidance":
        if _looks_like_club_overload_query(normalized_question):
            return _render_club_overload_fallback(question_focus)
        if _looks_like_doctoral_extension_query(normalized_question):
            return _render_doctoral_extension_fallback(question_focus)
        if _looks_like_learning_support_query(normalized_question):
            return _render_learning_support_fallback(question_focus)
        if _looks_like_service_process_query(normalized_question):
            return _render_service_process_fallback(question_focus)
        if _looks_like_stress_conflict_query(normalized_question):
            return _render_stress_conflict_fallback(question_focus)
        if _looks_like_decision_guidance_query(normalized_question):
            return _render_decision_guidance_fallback(question_focus)
        return (
            f"关于“{question_focus}”，当前可引用资料不足，先给你一个可执行起步方案：\n"
            "1) 先确定本周唯一主目标（例如通过期末或完成关键项目）；\n"
            "2) 把目标拆成每天可完成的最小任务，并设置截止时间；\n"
            "3) 每天结束前复盘一次完成率，次日优先处理未完成项；\n"
            f"4) 同步补充事实信息（如时间冲突、资源约束），我再给你精确方案。\n"
            f"建议：{no_answer_text}。"
        )
    if style == "analysis":
        return (
            f"目前缺少支撑“{question_focus}”的关键证据，无法做可靠原因判断。\n"
            f"建议：{no_answer_text}。"
        )
    return (
        f"当前资料不足，暂时无法直接回答“{question_focus}”。\n建议：{no_answer_text}。"
    )


def _render_general_knowledge_fallback(
    style: str,
    question_focus: str,
    is_complex_query: bool = True,
) -> str:
    if not is_complex_query:
        return (
            f"关于“{question_focus}”，我先给你一个直接、可执行的通用回答。"
            "如果你补充具体场景，我可以再细化到更贴合你的版本。"
        )
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
    if style == "course_selection":
        return _render_course_selection_fallback(question_focus)
    if style == "guidance":
        if _looks_like_club_overload_query(question_focus):
            return _render_club_overload_fallback(question_focus)
        if _looks_like_doctoral_extension_query(question_focus):
            return _render_doctoral_extension_fallback(question_focus)
        if _looks_like_learning_support_query(question_focus):
            return _render_learning_support_fallback(question_focus)
        if _looks_like_service_process_query(question_focus):
            return _render_service_process_fallback(question_focus)
        if _looks_like_stress_conflict_query(question_focus):
            return _render_stress_conflict_fallback(question_focus)
        if _looks_like_decision_guidance_query(question_focus):
            return _render_decision_guidance_fallback(question_focus)
        return (
            f"关于“{question_focus}”，可先按“目标-执行-校正”三阶段推进：\n"
            "第一阶段（今天）：明确核心目标与不可妥协约束；\n"
            "第二阶段（本周）：将任务拆成每日动作并安排固定执行时段；\n"
            "第三阶段（每2天）：检查结果与偏差，必要时压缩低优先级事项。"
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
    is_complex_query: bool = True,
) -> str:
    evidence_hint = (evidence_lines[0] or "").strip()[:52] if evidence_lines else ""
    if not is_complex_query:
        if evidence_hint:
            return (
                f"先直接回答：围绕“{question_focus}”，目前更稳妥的做法是先按现有信息执行。"
                f"可参考线索：{evidence_hint}。"
            )
        return (
            f"先直接回答：围绕“{question_focus}”，目前信息有限，建议先做小范围验证，"
            "再根据结果细化方案。"
        )
    if style == "comparison":
        return (
            f"基于当前命中内容，可先对“{question_focus}”做初步对比：\n"
            f"- 可参考线索：{evidence_hint or '当前命中资料'}\n"
            "- 若需最终结论，请补充明确的比较对象和决策优先级。"
        )
    if style == "summary":
        bullets = "\n".join(f"- {item}" for item in evidence_lines[:4])
        return f"本轮可提炼的重点如下：\n{bullets}"
    if style == "guidance":
        if _looks_like_club_overload_query(normalized_question):
            return _render_club_overload_fallback(question_focus, evidence_hint)
        if _looks_like_doctoral_extension_query(normalized_question):
            return _render_doctoral_extension_fallback(question_focus, evidence_hint)
        if _looks_like_learning_support_query(normalized_question):
            return _render_learning_support_fallback(question_focus, evidence_hint)
        if _looks_like_service_process_query(normalized_question):
            return _render_service_process_fallback(question_focus, evidence_hint)
        if _looks_like_stress_conflict_query(normalized_question):
            answer = _render_stress_conflict_fallback(question_focus)
            if evidence_hint:
                answer += f"\n可参考线索：{evidence_hint}。"
            return answer
        if _looks_like_decision_guidance_query(normalized_question):
            answer = _render_decision_guidance_fallback(question_focus)
            if evidence_hint:
                answer += f"\n可参考线索：{evidence_hint}。"
            return answer
        return (
            f"结合现有资料，围绕“{question_focus}”建议执行以下计划：\n"
            "1) 先锁定本周唯一主目标；\n"
            "2) 将本周任务拆成“每日必做 + 可选优化”两层，先保必做；\n"
            "3) 每天设置一个固定复盘点，记录完成率与阻塞原因；\n"
            "4) 按复盘结果滚动调整计划，并优先处理高收益低成本事项。\n"
            f"可参考线索：{evidence_hint or '当前命中资料'}。"
        )
    if style == "course_selection":
        return _render_course_selection_fallback(question_focus, evidence_hint)
    if style == "analysis" or any(
        token in normalized_question for token in ["原因", "分析", "风险"]
    ):
        return (
            f"初步判断：该问题可以从已有资料中部分解释。\n"
            f"可参考线索：{evidence_hint or '当前命中资料'}。\n"
            "如果你希望更深入，我可以继续拆解成“成因-影响-优先级”三层分析。"
        )
    return (
        f"目前可以先给你一个方向性答复：围绕“{question_focus}”，建议先做短周期验证，再做长期承诺。\n"
        f"可参考线索：{evidence_hint or '当前命中资料'}。\n"
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
