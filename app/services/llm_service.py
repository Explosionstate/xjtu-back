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


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=settings.api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
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

    compact_contexts = [ctx.strip()[:1200] for ctx in contexts[:3] if ctx.strip()]

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
        "你是知识库问答助手。请严格依据给定资料作答，不编造事实。"
        "回答必须严格按以下结构输出：\n"
        "结论：...\n"
        "依据：...\n"
        "建议：...\n"
        "请优先使用自然语言总结，先给结论再给简洁步骤。"
        "除非用户明确要求，不要原样大段复制文档或连续命令列表。"
        "如果资料不足，请明确说明资料不足。\n\n"
        f"问题：{question}\n\n"
        f"资料：\n{chr(10).join(compact_contexts)}"
    )
    effective_timeout = max(
        2,
        int(
            timeout_seconds
            if timeout_seconds is not None
            else settings.llm_timeout_seconds
        ),
    )

    try:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(get_chat_llm().invoke, prompt)
        try:
            response = future.result(timeout=effective_timeout)
        finally:
            future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        answer = _extract_text_content(getattr(response, "content", ""))
        if not answer:
            return LLMAnswerResult(
                answer=_fallback_natural_answer(question, compact_contexts),
                mode="empty_fallback",
            )
        answer = _ensure_three_section(answer)
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
    q = (question or "").strip().lower()
    if not contexts:
        return "结论：未在知识库中检索到可用于回答的资料。\n依据：当前检索结果为空。\n建议：请补充更具体问题或增加知识库资料后重试。"

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
        ]
        return any(marker in t for marker in noise_markers)

    sentences = re.split(r"[\n。！？!?]", merged)
    cleaned: list[str] = []
    for item in sentences:
        text = " ".join(item.strip().split())
        if not text or text in cleaned or _is_noise_line(text):
            continue
        cleaned.append(text)
        if len(cleaned) >= 5:
            break

    if not cleaned:
        return (
            "结论：已完成检索，但可直接引用的高质量片段不足。\n"
            "依据：命中内容以命令/配置片段为主，不适合作为自然语言回答依据。\n"
            "建议：请补充更贴近业务的文档内容，或降低该类技术脚本文档在当前问答场景中的优先级。"
        )

    summary = "；".join(cleaned[:3])

    if any(token in q for token in ["总结", "概述", "重点", "新增文档", "指南"]):
        return (
            "结论：根据当前检索内容，可提炼出本次资料的主要关注点。\n"
            f"依据：{summary}。\n"
            "建议：如需更可执行的版本，请指定对象（学生/教师/管理员）和输出格式（三点清单/一周行动表）。"
        )

    return (
        f"结论：根据当前检索结果，问题可从资料中部分回答。\n"
        f"依据：{summary}。\n"
        "建议：如需更精准结论，请补充上下文或限定具体场景。"
    )


def _is_academic_analysis_query(question: str) -> bool:
    q = (question or "").strip().lower()
    flags = ["学业分析", "学习分析", "成绩分析", "学情分析", "学习情况"]
    return any(flag in q for flag in flags)


def _ensure_three_section(answer: str) -> str:
    text = (answer or "").strip()
    if not text:
        return "结论：暂无有效回答。\n依据：模型未返回内容。\n建议：请调整问题后重试。"

    if all(tag in text for tag in ("结论", "依据", "建议")):
        return text

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    head = lines[0] if lines else text
    rest = "；".join(lines[1:3]) if len(lines) > 1 else text[:160]
    tail = lines[-1] if len(lines) > 2 else "建议结合更多上下文继续追问。"
    return f"结论：{head}\n依据：{rest}\n建议：{tail}"


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
