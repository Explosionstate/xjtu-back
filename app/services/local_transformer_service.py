from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys
from threading import BoundedSemaphore, Lock
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
_WORKER_STATE_LOCK = Lock()
_WORKER_COOLDOWN_SECONDS = 90
_WORKER_TIMEOUT_SECONDS = 24
_worker_disabled_until = 0.0
_worker_last_error = ""


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
            f"未找到本地 Transformer 模型: {candidate}，请先下载到本地模型目录。",
            status_code=500,
        )
    return reference


@lru_cache(maxsize=2)
def _get_local_model(model_reference: str):
    torch_module, model_cls, tokenizer_cls = _load_runtime_modules()
    if tokenizer_cls is None or model_cls is None:
        raise BusinessError(
            "未安装 transformers，无法加载本地 Transformer 模型。",
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
            # Fallback to CPU when CUDA is unavailable or VRAM is insufficient.
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


def _build_prompt(
    question: str,
    contexts: list[str],
    system_instruction: str | None = None,
    kb_hit: bool | None = None,
) -> str:
    style = _detect_answer_style(question)
    open_answer_mode = _looks_like_open_guidance_question(question)
    evidence_contexts: list[str] = []
    background_contexts: list[str] = []
    for item in contexts[:5]:
        text = item.strip()
        if not text:
            continue
        lowered = text.lower()
        if "[证据" in text or "来源:" in text or "相关度:" in text:
            evidence_contexts.append(text[:1200])
            continue
        if "[会话背景]" in text or "[用户画像]" in text or "背景" in lowered:
            background_contexts.append(text[:600])
            continue
        if kb_hit is True:
            evidence_contexts.append(text[:1200])
        else:
            background_contexts.append(text[:600])

    evidence_block = "\n".join(f"- {item}" for item in evidence_contexts[:4])
    background_block = "\n".join(f"- {item}" for item in background_contexts[:2])

    style_hint = {
        "direct": "先直接回答，再补充必要依据。",
        "analysis": "先给判断，再解释主要原因与影响。",
        "comparison": "按差异维度并列对比后给建议。",
        "guidance": "按步骤给出可执行方案。",
        "summary": "提炼 3-5 条重点信息。",
    }.get(style, "按问题自然组织回答。")

    instruction_block = ""
    if system_instruction and system_instruction.strip():
        instruction_block = f"系统指令:\n{system_instruction.strip()}\n\n"

    if open_answer_mode:
        kb_hint = "开放问题：请先独立回答，再把资料当作校验线索，不要复述原文。"
    elif kb_hit is True:
        kb_hint = "知识库已命中：请先给结论，再给1条关键证据，不要泛聊。"
    elif kb_hit is False:
        kb_hint = "知识库未命中：请自然说明边界并给最小可执行建议，不要编造事实。"
    else:
        kb_hint = "优先基于知识库回答，证据不足时明确边界。"

    return (
        f"{instruction_block}"
        "你是西交AI助手，请优先回答用户真正的问题。"
        f"{kb_hint}"
        "不要机械套用固定模板。"
        f"{style_hint}\n\n"
        f"问题: {question}\n\n"
        f"知识库线索:\n{evidence_block or '（暂无高置信证据）'}\n\n"
        f"背景补充:\n{background_block or '（无额外背景）'}"
    )


def generate_answer_with_local_transformer(
    question: str,
    contexts: list[str],
    model_name: str | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    system_instruction: str | None = None,
    kb_hit: bool | None = None,
) -> tuple[str, str, dict[str, int | str | bool]]:
    if not settings.local_transformer_enabled:
        raise BusinessError("本地 Transformer 模型已禁用", status_code=400)

    return _generate_answer_via_worker(
        question=question,
        contexts=contexts,
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        system_instruction=system_instruction,
        kb_hit=kb_hit,
    )


def _generate_answer_via_worker(
    *,
    question: str,
    contexts: list[str],
    model_name: str | None,
    temperature: float | None,
    max_new_tokens: int | None,
    system_instruction: str | None,
    kb_hit: bool | None,
) -> tuple[str, str, dict[str, int | str | bool]]:
    global _worker_disabled_until, _worker_last_error

    now = perf_counter()
    with _WORKER_STATE_LOCK:
        if _worker_disabled_until > now:
            remaining = int(max(1, _worker_disabled_until - now))
            detail = f"本地模型运行时暂时不可用，请 {remaining}s 后重试。"
            if _worker_last_error:
                detail = f"{detail} 最近错误：{_worker_last_error}"
            raise BusinessError(detail, status_code=503)

    payload = {
        "question": question,
        "contexts": contexts,
        "model_name": model_name,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "system_instruction": system_instruction,
        "kb_hit": kb_hit,
    }
    cmd = [sys.executable, "-m", "app.services.local_transformer_worker"]
    try:
        result = subprocess.run(
            cmd,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_WORKER_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        with _WORKER_STATE_LOCK:
            _worker_disabled_until = perf_counter() + _WORKER_COOLDOWN_SECONDS
            _worker_last_error = "worker_timeout"
        raise BusinessError(
            "本地模型执行超时，已自动进入保护冷却。", status_code=503
        ) from exc
    except Exception as exc:
        with _WORKER_STATE_LOCK:
            _worker_disabled_until = perf_counter() + _WORKER_COOLDOWN_SECONDS
            _worker_last_error = exc.__class__.__name__
        raise BusinessError(
            "本地模型运行时启动失败，已切换保护模式。", status_code=503
        ) from exc

    if result.returncode != 0:
        stderr = (result.stderr or "").strip().splitlines()
        last_error = stderr[-1][:120] if stderr else f"exit_{result.returncode}"
        with _WORKER_STATE_LOCK:
            _worker_disabled_until = perf_counter() + _WORKER_COOLDOWN_SECONDS
            _worker_last_error = last_error
        raise BusinessError("本地模型运行异常，已自动进入保护冷却。", status_code=503)

    try:
        output = json.loads((result.stdout or "").strip() or "{}")
    except json.JSONDecodeError as exc:
        with _WORKER_STATE_LOCK:
            _worker_disabled_until = perf_counter() + _WORKER_COOLDOWN_SECONDS
            _worker_last_error = "invalid_worker_json"
        raise BusinessError(
            "本地模型返回结果异常，已切换保护模式。", status_code=503
        ) from exc

    with _WORKER_STATE_LOCK:
        _worker_disabled_until = 0.0
        _worker_last_error = ""

    answer = str(output.get("answer") or "").strip()
    model_reference = str(output.get("model_reference") or model_name or "")
    metrics = output.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    return answer, model_reference, metrics


def _generate_answer_in_process(
    question: str,
    contexts: list[str],
    model_name: str | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    system_instruction: str | None = None,
    kb_hit: bool | None = None,
) -> tuple[str, str, dict[str, int | str | bool]]:
    if not settings.local_transformer_enabled:
        raise BusinessError("本地 Transformer 模型已禁用", status_code=400)

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
        prompt = _build_prompt(
            question=question,
            contexts=contexts,
            system_instruction=system_instruction,
            kb_hit=kb_hit,
        )
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
    if _looks_like_open_guidance_question(q):
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


def _looks_like_open_guidance_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    decision_markers = ["还是", "要不要", "选择", "纠结", "不知道"]
    open_markers = [
        "跨考",
        "考研",
        "读研",
        "实习",
        "就业",
        "方向",
        "大厂",
        "焦虑",
        "迷茫",
        "压力",
        "痛苦",
        "怎么办",
        "思路",
    ]
    return any(marker in q for marker in open_markers) or (
        any(marker in q for marker in decision_markers)
        and any(marker in q for marker in ["跨考", "实习", "就业", "方向"])
    )


def _local_generation_profile(style: str) -> tuple[int, float]:
    profile = {
        "direct": (220, 0.2),
        "analysis": (240, 0.22),
        "comparison": (240, 0.24),
        "guidance": (260, 0.23),
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
    cuda_available = bool(torch_module is not None and torch_module.cuda.is_available())
    if settings.transformer_device == "cuda" and not cuda_available:
        active_device = "cpu"
    with _WORKER_STATE_LOCK:
        worker_available = _worker_disabled_until <= perf_counter()
        worker_last_error = _worker_last_error
    return {
        "local_transformer_enabled": settings.local_transformer_enabled,
        "local_model": settings.local_transformer_model,
        "transformer_device": settings.transformer_device,
        "active_device": active_device,
        "cuda_available": cuda_available,
        "max_concurrency": settings.local_transformer_max_concurrency,
        "queue_timeout_seconds": settings.local_transformer_queue_timeout_seconds,
        "worker_isolated": True,
        "worker_available": worker_available,
        "worker_last_error": worker_last_error,
    }


def local_transformer_backup_available() -> bool:
    runtime = local_transformer_runtime()
    if not bool(runtime.get("local_transformer_enabled")):
        return False
    if not bool(runtime.get("worker_available", True)):
        return False
    if settings.transformer_device == "cuda" and runtime.get("active_device") != "cuda":
        return False
    return True
