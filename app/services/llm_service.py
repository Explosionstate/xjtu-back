from __future__ import annotations

from functools import lru_cache

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
    )


def answer_with_llm(question: str, contexts: list[str]) -> str:
    if not settings.llm_enabled or not settings.api_key:
        return "\n\n".join(contexts)

    prompt = (
        "你是知识库问答助手，请严格根据给定资料回答。"
        "若资料无法回答，请明确说明。\n\n"
        f"问题：{question}\n\n"
        f"资料：\n{chr(10).join(contexts)}"
    )
    try:
        response = get_chat_llm().invoke(prompt)
        return (response.content or "").strip()
    except Exception:
        # Keep chat endpoint responsive when upstream model is slow/unavailable.
        return "\n\n".join(contexts)
