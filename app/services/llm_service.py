from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from functools import lru_cache
import re

from langchain_openai import ChatOpenAI

from app.core.config import settings


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
) -> str:
    enabled = settings.llm_enabled if llm_enabled is None else llm_enabled
    if not enabled or not settings.api_key:
        return "\n\n".join(contexts)

    compact_contexts = [ctx.strip()[:1200] for ctx in contexts[:3] if ctx.strip()]

    prompt = (
        "你是知识库问答助手。请严格根据给定资料作答，不编造事实。"
        "请优先使用自然语言总结，先给结论再给简洁步骤。"
        "除非用户明确要求，不要原样大段复制文档或连续命令列表。"
        "若资料不足，请明确说资料不足。\n\n"
        f"问题：{question}\n\n"
        f"资料：\n{chr(10).join(compact_contexts)}"
    )
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(get_chat_llm().invoke, prompt)
            response = future.result(timeout=settings.llm_timeout_seconds + 1)
        content = response.content
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()
    except FuturesTimeoutError:
        return _fallback_natural_answer(question, compact_contexts)
    except Exception:
        # Keep endpoint responsive and still return readable text when LLM is unavailable.
        return _fallback_natural_answer(question, compact_contexts)


def _fallback_natural_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "未在知识库中检索到可用于回答的资料。"

    merged = "\n".join(contexts)
    # Normalize noisy line breaks for a short, readable summary.
    sentences = re.split(r"[\n。！？!?]", merged)
    cleaned: list[str] = []
    for item in sentences:
        text = " ".join(item.strip().split())
        if not text:
            continue
        if text in cleaned:
            continue
        cleaned.append(text)
        if len(cleaned) >= 4:
            break

    summary = "；".join(cleaned)
    return f"根据知识库检索结果，{summary}。"
