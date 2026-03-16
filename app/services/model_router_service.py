from __future__ import annotations

from time import perf_counter

from app.core.errors import BusinessError
from app.services.llm_service import answer_with_llm
from app.services.local_transformer_service import (
    generate_answer_with_local_transformer,
)


def generate_answer_by_provider(
    provider: str,
    question: str,
    contexts: list[str],
    model: str | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
) -> tuple[str, str, dict[str, int | str | bool]]:
    provider_name = (provider or "").strip().lower() or "local_transformer"
    if provider_name == "qwen":
        start = perf_counter()
        answer = answer_with_llm(question=question, contexts=contexts, llm_enabled=True)
        return (
            answer,
            model or "qwen3.5-plus",
            {
                "provider": "qwen",
                "generation_ms": int((perf_counter() - start) * 1000),
                "fallback_cpu": False,
            },
        )

    if provider_name in {"local_transformer", "transformer", "local"}:
        answer, model_reference, metrics = generate_answer_with_local_transformer(
            question=question,
            contexts=contexts,
            model_name=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return answer, model_reference, metrics

    raise BusinessError(
        f"不支持的provider: {provider}. 可选值: qwen, local_transformer",
        status_code=400,
    )
