from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter

from app.core.config import settings
from app.core.errors import BusinessError
from app.services.embedding_service import resolve_model_reference

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


_GENERATION_SEMAPHORE = BoundedSemaphore(
    value=max(1, settings.local_transformer_max_concurrency)
)


def _resolve_transformer_model_reference(model_name: str | None) -> str:
    candidate = (model_name or "").strip() or settings.local_transformer_model
    reference = resolve_model_reference(candidate)
    if not Path(reference).exists():
        raise BusinessError(
            f"未找到本地Transformer模型: {candidate}，请先下载到本地模型目录。",
            status_code=500,
        )
    return reference


@lru_cache(maxsize=2)
def _get_local_model(model_reference: str):
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise BusinessError(
            "未安装 transformers，无法加载本地Transformer模型。",
            status_code=500,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_reference, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if (
        settings.transformer_device == "cuda"
        and torch is not None
        and torch.cuda.is_available()
    ):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_reference,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            model = model.to("cuda")
        except Exception:
            # Fallback to CPU when CUDA is unavailable at runtime or VRAM is insufficient.
            model = AutoModelForCausalLM.from_pretrained(
                model_reference,
                trust_remote_code=True,
            )
            model = model.to("cpu")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_reference,
            trust_remote_code=True,
        )
        model = model.to("cpu")
    return tokenizer, model


def _build_prompt(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(
        item.strip()[:1200] for item in contexts[:4] if item.strip()
    )
    return (
        "你是西交AI助手，请基于检索资料回答。"
        "若资料不足请明确指出，不要编造。"
        "输出结构：结论、依据、建议。"
        "回答不少于220字，建议至少5条且可执行。\n\n"
        f"问题：{question}\n\n"
        f"资料：\n{context_block}"
    )


def generate_answer_with_local_transformer(
    question: str,
    contexts: list[str],
    model_name: str | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
) -> tuple[str, str, dict[str, int | str | bool]]:
    if not settings.local_transformer_enabled:
        raise BusinessError("本地Transformer模型已禁用", status_code=400)

    model_reference = _resolve_transformer_model_reference(model_name)
    acquired = _GENERATION_SEMAPHORE.acquire(
        timeout=max(1, settings.local_transformer_queue_timeout_seconds)
    )
    if not acquired:
        raise BusinessError(
            "本地模型当前繁忙，请稍后重试（队列已满）。",
            status_code=429,
        )

    tokenizer, model = _get_local_model(model_reference)
    initial_device = str(getattr(model, "device", "unknown"))
    start = perf_counter()
    fallback_cpu = False

    try:
        prompt = _build_prompt(question=question, contexts=contexts)
        model_inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens
            or settings.local_transformer_max_new_tokens,
            "temperature": temperature
            if temperature is not None
            else settings.local_transformer_temperature,
            "do_sample": True,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        try:
            output = _generate(
                model=model, model_inputs=model_inputs, kwargs=generation_kwargs
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or torch is None:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model = model.to("cpu")
            fallback_cpu = True
            model_inputs = {k: v.to("cpu") for k, v in model_inputs.items()}
            reduced_kwargs = dict(generation_kwargs)
            reduced_kwargs["max_new_tokens"] = min(
                128, int(reduced_kwargs.get("max_new_tokens", 128))
            )
            output = _generate(
                model=model, model_inputs=model_inputs, kwargs=reduced_kwargs
            )

        prompt_tokens = model_inputs["input_ids"].shape[1]
        generated = output[0][prompt_tokens:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not answer:
            answer = "模型未生成有效结果，请调整问题后重试。"
        elapsed_ms = int((perf_counter() - start) * 1000)
        metrics: dict[str, int | str | bool] = {
            "provider": "local_transformer",
            "generation_ms": elapsed_ms,
            "queue_wait_ms": 0,
            "initial_device": initial_device,
            "final_device": str(getattr(model, "device", "unknown")),
            "fallback_cpu": fallback_cpu,
        }
        return answer, model_reference, metrics
    finally:
        _GENERATION_SEMAPHORE.release()


def _generate(model, model_inputs: dict, kwargs: dict):
    if torch is not None:
        with torch.inference_mode():
            return model.generate(**model_inputs, **kwargs)
    return model.generate(**model_inputs, **kwargs)


def local_transformer_runtime() -> dict[str, int | str | bool]:
    active_device = settings.transformer_device
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if settings.transformer_device == "cuda" and not cuda_available:
        active_device = "cpu"
    return {
        "local_transformer_enabled": settings.local_transformer_enabled,
        "local_model": settings.local_transformer_model,
        "transformer_device": settings.transformer_device,
        "active_device": active_device,
        "cuda_available": cuda_available,
        "max_concurrency": settings.local_transformer_max_concurrency,
        "queue_timeout_seconds": settings.local_transformer_queue_timeout_seconds,
    }
