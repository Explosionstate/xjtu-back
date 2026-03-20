from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any

from langchain_openai import ChatOpenAI

from app.core.config import settings


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
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,
    )


def answer_with_llm(
    question: str,
    contexts: list[str],
    llm_enabled: bool | None = None,
    system_instruction: str | None = None,
    timeout_seconds: int | None = None,
) -> LLMAnswerResult:
    enabled = settings.llm_enabled if llm_enabled is None else llm_enabled
    if not enabled or not settings.api_key:
        return LLMAnswerResult(answer="\n\n".join(contexts), mode="disabled")

    style = _detect_answer_style(question)
    profile = STYLE_PROFILES.get(style, STYLE_PROFILES["direct"])
    compact_contexts = _compact_contexts(
        contexts=contexts,
        limit=profile.context_limit,
        max_chars=profile.context_chars,
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

    prompt = (
        f"{(system_instruction or '').strip()}\n"
        f"{academic_prompt}"
        "你是知识库问答助手。必须基于给定资料作答，不得编造事实。\n"
        f"回答类型：{profile.style}\n"
        f"组织建议：{profile.organization_hint}\n"
        "写作要求：\n"
        "1) 优先回答用户当前问题，再补充必要依据。\n"
        "2) 不要机械套用固定模板，按问题自然组织段落。\n"
        "3) 资料不足时，明确不确定边界，并给出下一步可执行建议。\n\n"
        f"问题：{question}\n\n"
        f"资料：\n{chr(10).join(compact_contexts) if compact_contexts else '（当前未检索到高置信资料）'}"
    )

    effective_timeout = max(
        2,
        int(
            timeout_seconds
            if timeout_seconds is not None
            else settings.llm_timeout_seconds
        ),
    )
    effective_temperature = max(settings.llm_temperature, profile.temperature_floor)
    effective_tokens = max(120, min(LLM_MAX_OUTPUT_TOKENS, profile.max_tokens))

    try:
        executor = ThreadPoolExecutor(max_workers=1)
        llm = get_chat_llm(
            temperature=round(effective_temperature, 2),
            top_p=round(profile.top_p, 2),
            frequency_penalty=round(profile.frequency_penalty, 2),
            presence_penalty=round(profile.presence_penalty, 2),
        )
        future = executor.submit(
            llm.invoke,
            prompt,
            max_tokens=effective_tokens,
        )
        try:
            response = future.result(timeout=effective_timeout)
        finally:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        answer = _normalize_answer_text(_extract_text_content(getattr(response, "content", "")))
        if not answer or len(answer) < 18:
            return LLMAnswerResult(
                answer=_fallback_natural_answer(question, compact_contexts),
                mode="empty_fallback",
            )
        return LLMAnswerResult(
            answer=answer,
            mode="llm",
            reasoning=_extract_reasoning_text(response),
        )
    except FuturesTimeoutError:
        return LLMAnswerResult(
            answer=_fallback_natural_answer(question, compact_contexts),
            mode="timeout_fallback",
        )
    except Exception:
        # Keep endpoint responsive and still return readable text when LLM is unavailable.
        return LLMAnswerResult(
            answer=_fallback_natural_answer(question, compact_contexts),
            mode="error_fallback",
        )


def _fallback_natural_answer(question: str, contexts: list[str]) -> str:
    style = _detect_answer_style(question)
    q = (question or "").strip().lower()
    question_focus = (question or "").strip()[:80] or "当前问题"

    if not contexts:
        return _render_no_context_fallback(style, question_focus)

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
        return _render_no_context_fallback(style, question_focus)

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


def _render_no_context_fallback(style: str, question_focus: str) -> str:
    if style == "comparison":
        return (
            f"当前还缺少可比依据，暂时不能对“{question_focus}”给出可靠对比结论。\n"
            "你可以补充：比较对象、评价维度（如成本/效果/风险）和时间范围。"
        )
    if style == "summary":
        return (
            f"当前资料不足，暂时无法对“{question_focus}”做有效摘要。\n"
            "建议先补充目标文档范围或主题关键词，我再按重点清单快速整理。"
        )
    if style == "guidance":
        return (
            f"关于“{question_focus}”，当前依据不足，但可以先按这三步推进：\n"
            "1) 明确目标与约束；2) 给出已知事实；3) 我据此输出可执行步骤和风险检查点。"
        )
    if style == "analysis":
        return (
            f"目前缺少支撑“{question_focus}”的关键证据，无法做可靠原因判断。\n"
            "建议补充事件背景、时间线和关键指标，我再给出有证据链的分析。"
        )
    return (
        f"当前资料不足，暂时无法直接回答“{question_focus}”。\n"
        "你可以补充对象、时间范围和已知事实，我会给出更贴题的结论。"
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
