from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import BoundedSemaphore
from time import perf_counter

from app.core.config import settings
from app.core.errors import BusinessError
from app.services.embedding_service import resolve_model_reference

_torch = None
_AutoModelForCausalLM = None
_AutoTokenizer = None
_runtime_import_attempted = False


_GENERATION_SEMAPHORE = BoundedSemaphore(
    value=max(1, settings.local_transformer_max_concurrency)
)


def _load_runtime_modules():
    global _torch, _AutoModelForCausalLM, _AutoTokenizer, _runtime_import_attempted
    if _runtime_import_attempted:
        return _torch, _AutoModelForCausalLM, _AutoTokenizer

    _runtime_import_attempted = True
    try:
        import torch as torch_module
    except ImportError:  # pragma: no cover
        torch_module = None
    _torch = torch_module

    try:
        from transformers import AutoModelForCausalLM as model_cls
        from transformers import AutoTokenizer as tokenizer_cls
    except ImportError:  # pragma: no cover
        model_cls = None
        tokenizer_cls = None
    _AutoModelForCausalLM = model_cls
    _AutoTokenizer = tokenizer_cls
    return _torch, _AutoModelForCausalLM, _AutoTokenizer


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
    torch_module, model_cls, tokenizer_cls = _load_runtime_modules()
    if tokenizer_cls is None or model_cls is None:
        raise BusinessError(
            "未安装 transformers，无法加载本地Transformer模型。",
            status_code=500,
        )

    tokenizer = tokenizer_cls.from_pretrained(model_reference, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if (
        settings.transformer_device == "cuda"
        and torch_module is not None
        and torch_module.cuda.is_available()
    ):
        try:
            model = model_cls.from_pretrained(
                model_reference,
                trust_remote_code=True,
                torch_dtype=torch_module.float16,
            )
            model = model.to("cuda")
        except Exception:
            # Fallback to CPU when CUDA is unavailable at runtime or VRAM is insufficient.
            model = model_cls.from_pretrained(
                model_reference,
                trust_remote_code=True,
            )
            model = model.to("cpu")
    else:
        model = model_cls.from_pretrained(
            model_reference,
            trust_remote_code=True,
        )
        model = model.to("cpu")
    return tokenizer, model


def _build_prompt(question: str, contexts: list[str]) -> str:
    style = _detect_answer_style(question)
    context_block = "\n\n".join(
        item.strip()[:1200] for item in contexts[:4] if item.strip()
    )
    style_hint = {
        "direct": "先直接回答，再补充必要依据。",
        "analysis": "先给判断，再解释原因与影响。",
        "comparison": "按差异维度并列对比后给建议。",
        "guidance": "按步骤给出可执行方案。",
        "summary": "提炼 3-5 条重点信息。",
    }.get(style, "按问题自然组织回答。")
    return (
        "你是西交AI助手，请基于检索资料回答。"
        "若资料不足请明确指出，不要编造。"
        "不要机械套用固定模板。"
        f"{style_hint}\n\n"
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
    torch_module, _, _ = _load_runtime_modules()
    initial_device = str(getattr(model, "device", "unknown"))
    start = perf_counter()
    fallback_cpu = False

    try:
        style = _detect_answer_style(question)
        prompt = _build_prompt(question=question, contexts=contexts)
        model_inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(model, "device"):
            model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        target_tokens, temperature_floor = _local_generation_profile(style)
        chosen_max_tokens = min(
            max_new_tokens or settings.local_transformer_max_new_tokens,
            target_tokens,
        )
        chosen_temperature = max(
            temperature
            if temperature is not None
            else settings.local_transformer_temperature,
            temperature_floor,
        )
        generation_kwargs = {
            "max_new_tokens": max(120, int(chosen_max_tokens)),
            "temperature": round(float(chosen_temperature), 2),
            "do_sample": True,
            "top_p": 0.88,
            "repetition_penalty": 1.05,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        try:
            output = _generate(
                model=model,
                model_inputs=model_inputs,
                kwargs=generation_kwargs,
                torch_module=torch_module,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or torch_module is None:
                raise
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()
            model = model.to("cpu")
            fallback_cpu = True
            model_inputs = {k: v.to("cpu") for k, v in model_inputs.items()}
            reduced_kwargs = dict(generation_kwargs)
            reduced_kwargs["max_new_tokens"] = min(
                128, int(reduced_kwargs.get("max_new_tokens", 128))
            )
            output = _generate(
                model=model,
                model_inputs=model_inputs,
                kwargs=reduced_kwargs,
                torch_module=torch_module,
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


def _local_generation_profile(style: str) -> tuple[int, float]:
    profile = {
        "direct": (220, 0.2),
        "analysis": (280, 0.22),
        "comparison": (280, 0.24),
        "guidance": (320, 0.25),
        "summary": (240, 0.2),
    }
    return profile.get(style, (240, 0.2))


def _generate(model, model_inputs: dict, kwargs: dict, torch_module=None):
    if torch_module is not None:
        with torch_module.inference_mode():
            return model.generate(**model_inputs, **kwargs)
    return model.generate(**model_inputs, **kwargs)


def local_transformer_runtime() -> dict[str, int | str | bool]:
    torch_module, _, _ = _load_runtime_modules()
    active_device = settings.transformer_device
    cuda_available = bool(
        torch_module is not None and torch_module.cuda.is_available()
    )
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
