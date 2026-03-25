from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from time import perf_counter
import logging
import re
from typing import Any, Callable
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import BusinessError
from app.models.chat import ChatLog, ChatPerfLog, Conversation, Message
from app.models.knowledge_base import KnowledgeBase
from app.models.rbac import User
from app.schemas.chat import ChatCompletionRequest, ChatThinking, SourceItem
from app.services.agent_profile_service import (
    build_agent_cloud_instruction,
    build_agent_system_instruction,
    get_agent_bound_kb_name,
    get_agent_no_answer_strategy,
    needs_profile_context,
    normalize_agent_key,
)
from app.services.llm_service import LLMAnswerResult, answer_with_llm
from app.services.local_transformer_service import (
    generate_answer_with_local_transformer,
    local_transformer_backup_available,
    local_transformer_runtime,
)
from app.services.external_profile_service import load_user_profile_context
from app.services.academic_service import get_my_academic_analysis
from app.services.langgraph_service import run_chat_workflow_graph
from app.services.retrieval_config_service import get_effective_retrieval_config
from app.services.retrieval_service import hybrid_retrieve
from app.services.sensitive_service import (
    detect_sensitive_text,
    get_sensitive_words,
    log_sensitive_block,
    mask_sensitive_text,
)
from app.services.system_config_service import (
    DEFAULT_CONTEXT_MAX_ROUNDS,
    DEFAULT_CONTEXT_MAX_TOKENS,
    get_config_values,
)
from app.services.kb_service import resolve_agent_bound_kb_scope


CHAT_TOTAL_TIMEOUT_SECONDS = 120
WORKFLOW_TIMEOUT_SECONDS = 112
GENERATION_STAGE_TIMEOUT_SECONDS = 96
LOCAL_GENERATION_TIMEOUT_SECONDS = 20
WORKFLOW_POLL_INTERVAL_SECONDS = 0.35
NON_ACADEMIC_CLOUD_CHAT_TIMEOUT_SECONDS = 120
NON_ACADEMIC_CLOUD_WORKFLOW_TIMEOUT_SECONDS = 112
NON_ACADEMIC_CLOUD_GENERATION_TIMEOUT_SECONDS = 96
ACADEMIC_CHAT_TOTAL_TIMEOUT_SECONDS = 120
ACADEMIC_WORKFLOW_TIMEOUT_SECONDS = 112
ACADEMIC_GENERATION_STAGE_TIMEOUT_SECONDS = 96
ACADEMIC_PROFILE_TIMEOUT_SECONDS = 8.0
DEFAULT_PROFILE_TIMEOUT_SECONDS = 1.5
MIN_RELAXED_THRESHOLD = 0.12
LONG_TERM_MEMORY_ROLE = "system"
LONG_TERM_MEMORY_PREFIX = "[CONTEXT_MEMORY]"
LONG_TERM_MEMORY_MAX_CHARS = 1200
LONG_TERM_MEMORY_MAX_LINES = 14
logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ChatCompletionResult:
    conversation_id: str
    answer: str
    sources: list[SourceItem]
    thinking: ChatThinking


MODEL_ROUTE_CLOUD_ONLY = "cloud_only"
MODEL_ROUTE_HYBRID = "hybrid"
MODEL_ROUTE_LOCAL_ONLY = "local_only"
MODEL_ROUTE_RETRIEVAL_ONLY = "retrieval_only"

QUESTION_MODE_OPEN = "open_question"
QUESTION_MODE_FACT = "fact_lookup"
QUESTION_MODE_ACADEMIC = "academic_analysis"
QUESTION_MODE_CRISIS = "crisis"


def _resolve_model_route_mode(
    payload: ChatCompletionRequest,
) -> tuple[str, bool, bool]:
    cloud_enabled = (
        settings.llm_enabled
        if payload.llm_enabled is None
        else bool(payload.llm_enabled)
    )
    local_enabled = bool(payload.local_transformer_enabled)

    # Keep routing deterministic: cloud and local are treated as two distinct modes.
    # When both switches are on, prioritize cloud mode but preserve local fallback ability.
    if cloud_enabled and local_enabled:
        return MODEL_ROUTE_CLOUD_ONLY, True, True
    if cloud_enabled:
        return MODEL_ROUTE_CLOUD_ONLY, cloud_enabled, local_enabled
    if local_enabled:
        return MODEL_ROUTE_LOCAL_ONLY, cloud_enabled, local_enabled
    return MODEL_ROUTE_RETRIEVAL_ONLY, cloud_enabled, local_enabled


def _model_route_desc(route_mode: str) -> str:
    return {
        MODEL_ROUTE_CLOUD_ONLY: "云端模型主答（Qwen 主答，可结合知识库）",
        MODEL_ROUTE_HYBRID: "云端+本地协同（Qwen 主答，知识库轻量补充）",
        MODEL_ROUTE_LOCAL_ONLY: "仅本地模型（知识库增强回答）",
        MODEL_ROUTE_RETRIEVAL_ONLY: "检索整理模式（未启用模型）",
    }.get(route_mode, route_mode)


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(payload)
    except Exception:
        logger.debug("progress callback failed", exc_info=True)


def _compact_text(text: str, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", (text or "").strip())
    if not compact:
        return ""
    return compact[:max_chars]


def _clean_retrieval_snippet(text: str, max_chars: int = 240) -> str:
    cleaned = str(text or "").replace("\r\n", "\n")
    cleaned = re.sub(r"(?m)^#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\|?\s*[-:|]{3,}\s*\|?$", "", cleaned)
    cleaned = re.sub(r"(?m)^[-*•]\s*", "", cleaned)
    cleaned = re.sub(r"(?m)^\d+[.)、]\s*", "", cleaned)
    cleaned = cleaned.replace("\\.", ".").replace("\\-", "-").replace("\\+", "+")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip(" ：:;；,，。-")
    if len(cleaned) < 10:
        return ""
    return cleaned[:max_chars]


_INTERNAL_TEMPLATE_MARKERS = (
    "回答类型",
    "复杂度模式",
    "组织建议",
    "写作要求",
    "知识库证据",
    "背景补充",
    "可参考线索（只在最后一句提及）",
    "以下几个方面",
    "按以下模板",
    "内部模板",
    "内部分类",
    "处理步骤",
    "系统策略",
    "处理摘要",
    "思考过程",
)


def _clean_answer_fragment(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^[\-\*\d\.\)\(、\s]+", "", cleaned)
    cleaned = re.sub(
        r"^(结论|建议|依据|缺少的依据|缺失依据|需补充信息|需要补充信息)\s*[:：]\s*",
        "",
        cleaned,
    )
    return cleaned.strip(" ：:;-")


def _strip_internal_template_text(raw_answer: str) -> str:
    lines: list[str] = []
    for raw_line in (raw_answer or "").replace("\r\n", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(marker in line for marker in _INTERNAL_TEMPLATE_MARKERS):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _polish_final_answer_text(raw_answer: str) -> str:
    text = _strip_internal_template_text(raw_answer)
    if not text:
        return ""
    text = re.sub(r"[；;]{2,}", "；", text)
    text = re.sub(r"(?m)^(先给结论|先给判断|先直接回答|先给答复)\s*[:：]\s*", "", text)
    text = text.replace("可参考线索：", "依据线索：")
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        current = line.strip()
        if not current:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue
        if cleaned_lines and current == cleaned_lines[-1]:
            continue
        cleaned_lines.append(current)
    return "\n".join(cleaned_lines).strip()


_PRIVATE_REASONING_MARKERS = (
    "Thinking Process:",
    "Analyze the Request:",
    "Determine the Conclusion:",
    "Drafting Content",
    "Review and Refine",
    "Character Count Check",
)


def _looks_like_private_reasoning_text(text: str | None) -> bool:
    content = (text or "").strip()
    if not content:
        return False
    lowered = content.lower()
    if any(marker.lower() in lowered for marker in _PRIVATE_REASONING_MARKERS):
        return True
    return bool(re.search(r"(?m)^\d+\.\s+\*\*.*\*\*", content))


def _looks_like_low_quality_local_answer(answer: str | None) -> bool:
    text = _compact_text(str(answer or ""), 240).lower()
    if not text or len(text) < 18:
        return True
    refusal_markers = (
        "我无法提供",
        "无法提供您所请求",
        "我不能提供",
        "对不起，我无法",
        "抱歉，我无法",
        "无法获取",
        "请提供更多信息",
        "如果您能提供更多",
        "作为ai",
        "没有足够信息",
    )
    if any(marker in text for marker in refusal_markers):
        return True
    if text.count("human:") >= 1 or text.count("assistant:") >= 2:
        return True
    return False


def _extract_answer_sections(
    cleaned_answer: str,
) -> tuple[dict[str, str], list[str]]:
    sections: dict[str, list[str]] = {
        "conclusion": [],
        "evidence": [],
        "advice": [],
        "missing": [],
    }
    fallback_lines: list[str] = []
    current_key: str | None = None
    label_patterns = {
        "conclusion": [
            "结论",
            "直接回答",
            "答复",
            "先给结论",
            "先给判断",
        ],
        "advice": [
            "建议",
            "行动建议",
            "下一步",
            "可执行方案",
        ],
        "evidence": [
            "依据",
            "证据",
            "关键依据",
            "支撑点",
            "理由",
            "参考依据",
        ],
        "missing": [
            "缺少的依据",
            "缺失依据",
            "缺少依据",
            "缺少信息",
            "需补充信息",
            "需要补充信息",
            "需补充",
            "信息不足",
        ],
    }

    for raw_line in (cleaned_answer or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        matched_key: str | None = None
        matched_content = ""
        for key, aliases in label_patterns.items():
            for alias in aliases:
                match = re.match(rf"^{re.escape(alias)}\s*[:：]\s*(.*)$", line)
                if match:
                    matched_key = key
                    matched_content = _clean_answer_fragment(match.group(1))
                    break
            if matched_key:
                break

        if matched_key:
            current_key = matched_key
            if matched_content:
                sections[matched_key].append(matched_content)
            continue

        cleaned_line = _clean_answer_fragment(line)
        if not cleaned_line:
            continue
        if current_key:
            sections[current_key].append(cleaned_line)
        else:
            fallback_lines.append(cleaned_line)

    flattened = {
        key: "；".join(item for item in values if item)
        for key, values in sections.items()
    }
    return flattened, fallback_lines


def _split_brief_sentences(text: str) -> list[str]:
    parts = re.split(r"[。！？!?;\n]+", text or "")
    output: list[str] = []
    for part in parts:
        cleaned = _clean_answer_fragment(part)
        if cleaned and cleaned not in output:
            output.append(cleaned)
    return output


def _infer_missing_evidence(
    *,
    question: str,
    question_mode: str,
    retrieved: list[dict],
    agent_key: str | None = None,
) -> str:
    lowered = (question or "").strip().lower()
    points: list[str] = []
    normalized_agent = normalize_agent_key(agent_key)

    role_points: dict[str, tuple[str, ...]] = {
        "student-growth": (
            "最近一学期成绩明细与薄弱课程清单",
            "每周可投入学习时间与阶段目标",
            "当前学习阻碍（时间冲突、习惯、压力）",
        ),
        "teacher-assistant": (
            "课程目标、授课对象与课时安排",
            "班级基础水平与当前课堂问题",
            "可用教学资源和评估约束",
        ),
        "counselor-ideology": (
            "学生基本情况、事件背景与时间线",
            "风险等级线索与已沟通记录",
            "可协同部门或可用支持资源",
        ),
        "risk-warning": (
            "预警对象、时间窗口与异常指标",
            "历史风险记录与近期变化趋势",
            "干预资源和责任人安排",
        ),
        "report-assistant": (
            "统计口径、数据时间范围与样本范围",
            "关键指标明细与对比对象",
            "报告用途和读者对象",
        ),
        "policy-qa": (
            "政策适用对象与具体场景",
            "时间范围和版本口径",
            "需核验的条款来源或制度名称",
        ),
    }
    points.extend(role_points.get(normalized_agent, ()))

    if question_mode == QUESTION_MODE_ACADEMIC or any(
        token in lowered for token in ["学业", "成绩", "学分", "课程", "挂科", "重修"]
    ):
        points.extend(
            [
                "最近一学期的成绩明细（课程名、分数、学分）",
                "课程完成情况与挂科/重修记录",
                "当前目标方向（保研、就业或升学）",
            ]
        )
    elif question_mode == QUESTION_MODE_FACT or any(
        token in lowered for token in ["政策", "规定", "流程", "办理", "申请", "材料"]
    ):
        points.extend(
            [
                "明确的对象或事项名称（课程/政策/系统模块）",
                "适用身份与范围（年级、学院、角色）",
                "具体时间范围（例如本学期、近30天）",
            ]
        )
    elif any(token in lowered for token in ["对比", "比较", "区别", "vs", "哪个"]):
        points.extend(
            [
                "明确的对比对象 A/B",
                "你的决策优先级（时间、成本、风险）",
                "可量化结果指标（成功标准）",
            ]
        )
    elif question_mode == QUESTION_MODE_CRISIS:
        points.extend(
            [
                "你当前是否存在立即危险",
                "你所在位置与是否有人陪同",
                "可联系的家人/同学/老师信息",
            ]
        )
    else:
        points.extend(
            [
                "具体场景和目标对象",
                "时间范围（例如本周、近一学期）",
                "当前约束条件（可投入时间、资源限制）",
            ]
        )

    if not retrieved:
        points.append("可核验的现状事实或关键数据")

    deduped: list[str] = []
    for item in points:
        normalized = _clean_answer_fragment(item)
        if not normalized:
            continue
        if normalized in deduped:
            continue
        deduped.append(normalized)
        if len(deduped) >= 3:
            break
    return "；".join(deduped)


def _format_final_user_answer(
    *,
    answer: str,
    question: str,
    question_mode: str,
    retrieved: list[dict],
    agent_key: str | None = None,
) -> str:
    cleaned_answer = _strip_internal_template_text(answer)
    extracted, fallback_lines = _extract_answer_sections(cleaned_answer)
    sentence_pool = _split_brief_sentences(cleaned_answer)
    is_complex = _is_complex_question(question, question_mode)

    conclusion = _clean_answer_fragment(extracted.get("conclusion", ""))
    if not conclusion:
        if fallback_lines:
            conclusion = fallback_lines[0]
        elif sentence_pool:
            conclusion = sentence_pool[0]
        else:
            focus = _compact_text(question, 56) or "当前问题"
            conclusion = f"围绕“{focus}”，先按现有信息给出可执行结论。"

    advice = _clean_answer_fragment(extracted.get("advice", ""))
    if not advice:
        if len(fallback_lines) > 1:
            advice = fallback_lines[1]
        else:
            candidate = next(
                (
                    item
                    for item in sentence_pool
                    if any(
                        token in item
                        for token in ["建议", "可以", "优先", "下一步", "先"]
                    )
                    and item != conclusion
                ),
                "",
            )
            advice = candidate or "先执行最小可行步骤，再根据补充信息细化方案。"

    evidence = _clean_answer_fragment(extracted.get("evidence", ""))
    if not evidence:
        candidate = next(
            (
                item
                for item in sentence_pool
                if any(
                    token in item
                    for token in ["依据", "证据", "基于", "因为", "来自", "显示"]
                )
                and item not in {conclusion, advice}
            ),
            "",
        )
        evidence = candidate
    if not evidence and retrieved:
        snippets: list[str] = []
        for item in retrieved[:2]:
            text = _clean_retrieval_snippet(str(item.get("content") or ""), 100)
            if not text:
                continue
            if text in snippets:
                continue
            snippets.append(text)
        evidence = "；".join(snippets)
    if not evidence:
        evidence = "当前知识库可直接引用的高置信证据有限，已按保守原则整理回答。"

    missing = _clean_answer_fragment(extracted.get("missing", ""))
    if not missing:
        missing = _infer_missing_evidence(
            question=question,
            question_mode=question_mode,
            retrieved=retrieved,
            agent_key=agent_key,
        )

    conclusion = _compact_text(conclusion, 170 if is_complex else 130)
    evidence = _compact_text(evidence, 260 if is_complex else 180)
    advice = _compact_text(advice, 220 if is_complex else 160)
    missing = _compact_text(missing, 220 if is_complex else 170)
    if missing and missing[-1] not in "。！？":
        missing = f"{missing}。"

    def _split_points(text: str, *, max_items: int, max_chars: int) -> list[str]:
        output: list[str] = []
        for part in re.split(r"[；;。！？!?]\s*|\n+", text or ""):
            cleaned = _clean_answer_fragment(part)
            if not cleaned:
                continue
            compact = _compact_text(cleaned, max_chars)
            if compact in output:
                continue
            output.append(compact)
            if len(output) >= max_items:
                break
        return output

    evidence_points = _split_points(
        evidence,
        max_items=3 if is_complex else 1,
        max_chars=100 if is_complex else 140,
    ) or [evidence]
    advice_points = _split_points(
        advice,
        max_items=3 if is_complex else 2,
        max_chars=100 if is_complex else 120,
    ) or [advice]

    lines = [f"结论：{conclusion}", "", "依据："]
    if len(evidence_points) == 1:
        lines.append(evidence_points[0])
    else:
        lines.extend(
            f"{index}. {item}" for index, item in enumerate(evidence_points, start=1)
        )
    lines.append("")
    lines.append("建议：")
    if len(advice_points) == 1:
        lines.append(advice_points[0])
    else:
        lines.extend(
            f"{index}. {item}" for index, item in enumerate(advice_points, start=1)
        )
    lines.append("")
    lines.append(f"缺少的依据：{missing}")
    return "\n".join(lines).strip()


def _is_long_term_memory_message(message: Message) -> bool:
    return (message.role or "").strip().lower() == LONG_TERM_MEMORY_ROLE and (
        message.content or ""
    ).startswith(LONG_TERM_MEMORY_PREFIX)


def _extract_long_term_memory(history: list[Message]) -> str:
    for message in reversed(history):
        if not _is_long_term_memory_message(message):
            continue
        return (message.content or "").replace(LONG_TERM_MEMORY_PREFIX, "", 1).strip()
    return ""


def _format_long_term_memory(memory: str) -> str:
    compact = _compact_text(memory, LONG_TERM_MEMORY_MAX_CHARS)
    if not compact:
        return ""
    return f"{LONG_TERM_MEMORY_PREFIX}\n{compact}"


def _looks_like_key_user_context(text: str) -> bool:
    lowered = (text or "").lower()
    if not lowered:
        return False
    if len(lowered) >= 24:
        return True
    key_tokens = [
        "目标",
        "背景",
        "约束",
        "范围",
        "时间",
        "课程",
        "学生",
        "老师",
        "管理员",
        "学分",
        "risk",
        "deadline",
        "port",
        "环境",
    ]
    return any(token in lowered for token in key_tokens)


def _merge_memory_lines(existing_memory: str, new_lines: list[str]) -> str:
    merged_lines: list[str] = []
    for raw in [
        line for line in existing_memory.split("\n") if line.strip()
    ] + new_lines:
        line = raw.strip()
        if not line or line in merged_lines:
            continue
        merged_lines.append(line)

    if len(merged_lines) > LONG_TERM_MEMORY_MAX_LINES:
        merged_lines = merged_lines[-LONG_TERM_MEMORY_MAX_LINES:]
    merged = "\n".join(merged_lines)
    return merged[:LONG_TERM_MEMORY_MAX_CHARS]


def _build_long_term_memory_update(
    existing_memory: str,
    dropped_messages: list[Message],
) -> str:
    memory_lines: list[str] = []
    for message in dropped_messages:
        text = _compact_text(message.content, 120 if message.role == "user" else 90)
        if not text:
            continue
        if message.role == "user":
            if _looks_like_key_user_context(text):
                memory_lines.append(f"用户背景: {text}")
        elif message.role == "assistant":
            if any(flag in text for flag in ["结论", "建议", "方案", "风险", "计划"]):
                memory_lines.append(f"助手结论: {text}")
        if len(memory_lines) >= 10:
            break
    return _merge_memory_lines(existing_memory, memory_lines)


def _build_history_context_brief(
    history: list[Message],
    question: str,
    long_term_memory: str = "",
    max_chars: int = 980,
) -> str:
    if not history:
        base = _compact_text(long_term_memory, 320)
        return f"长期记忆:\n{base}" if base else ""

    recent = history[-10:]
    user_lines = [
        _compact_text(item.content, 150)
        for item in recent
        if item.role == "user" and item.content.strip()
    ]
    assistant_lines = [
        _compact_text(item.content, 120)
        for item in recent
        if item.role == "assistant" and item.content.strip()
    ]

    sections: list[str] = []
    memory_block = _compact_text(long_term_memory, 320)
    if memory_block:
        sections.append(f"长期记忆:\n- {memory_block}")
    if user_lines:
        sections.append(
            "近期用户问题:\n"
            + "\n".join(f"- {line}" for line in user_lines[-6:] if line)
        )
    if assistant_lines:
        sections.append(
            "近期助手回答:\n"
            + "\n".join(f"- {line}" for line in assistant_lines[-3:] if line)
        )

    current = _compact_text(question, 140)
    if current:
        sections.append(f"当前问题:\n- {current}")

    merged = "\n".join(section for section in sections if section).strip()
    if not merged:
        return ""
    if len(merged) > max_chars:
        merged = merged[:max_chars]
    return f"对话上下文摘要:\n{merged}"


def _build_retrieval_shortfall_answer(
    question: str,
    trimmed_history: list[Message],
    agent_key: str | None = None,
) -> str:
    style = _detect_answer_style(question)
    short_question = _compact_text(question, 80) or "当前问题"
    no_answer_rules = get_agent_no_answer_strategy(agent_key)
    no_answer_hint = (
        "；".join(no_answer_rules[:2])
        if no_answer_rules
        else "补充目标对象、时间范围和关键事实后继续"
    )
    user_history = [
        _compact_text(item.content, 50)
        for item in trimmed_history
        if item.role == "user" and item.content.strip()
    ]
    history_hint = (
        "；".join(item for item in user_history[-2:] if item) or "暂无稳定历史事实"
    )
    if style == "comparison":
        return (
            f"当前资料不足，暂时无法对“{short_question}”给出可靠对比结论。\n"
            f"已可复用的历史信息：{history_hint}。\n"
            f"建议：{no_answer_hint}。"
        )
    if style == "summary":
        return (
            f"关于“{short_question}”，当前命中资料不足以形成有效摘要。\n"
            f"可参考的历史线索：{history_hint}。\n"
            f"建议：{no_answer_hint}。"
        )
    if style == "analysis":
        return (
            f"目前还缺少支撑“{short_question}”的关键证据，无法做可靠原因判断。\n"
            f"现有可复用上下文：{history_hint}。\n"
            f"建议：{no_answer_hint}。"
        )
    if style == "guidance":
        return (
            f"当前资料不足以直接回答“{short_question}”，但可以先启动可执行方案。\n"
            f"当前可复用上下文：{history_hint}。\n"
            f"建议：{no_answer_hint}。"
        )
    return (
        f"当前资料不足，暂时无法直接回答“{short_question}”。\n"
        f"可复用对话信息：{history_hint}。\n"
        f"建议：{no_answer_hint}。"
    )


def _build_kb_bounded_shortfall_answer(
    question: str,
    trimmed_history: list[Message],
    route_mode: str,
    agent_key: str | None = None,
) -> str:
    base = _build_retrieval_shortfall_answer(
        question=question,
        trimmed_history=trimmed_history,
        agent_key=agent_key,
    )
    if route_mode == MODEL_ROUTE_LOCAL_ONLY:
        return (
            "当前为“仅本地 Qwen + 本地知识库”模式，回答必须受知识库约束。\n"
            f"{base}\n"
            "如需继续回答，请补充或上传更相关的本地知识库资料。"
        )
    if route_mode == MODEL_ROUTE_HYBRID:
        return (
            "当前为“云端 + 本地知识库增强”模式，本轮未检索到足够知识库证据。\n"
            f"{base}\n"
            "建议先补充知识库内容或缩小问题范围后再问。"
        )
    return base


def _build_cloud_timeout_degraded_answer(
    question: str,
    trimmed_history: list[Message],
    long_term_memory: str = "",
) -> str:
    focus = _compact_text(question, 120) or "当前问题"
    hints: list[str] = []
    if trimmed_history:
        for item in reversed(trimmed_history):
            if item.role != "user":
                continue
            text = _compact_text(item.content, 60)
            if not text or text == focus:
                continue
            hints.append(text)
            if len(hints) >= 2:
                break
    if not hints and long_term_memory:
        memory = _compact_text(long_term_memory, 60)
        if memory:
            hints.append(memory)

    guidance = (
        "\n".join(f"- {item}" for item in hints)
        if hints
        else "- 明确你的目标、约束和预期输出格式"
    )
    return (
        "云端模型本轮响应超时，已返回快速可执行回答（不依赖知识库命中）。\n"
        f"问题：{focus}\n"
        "建议你按以下结构继续提问，以便下轮更快得到完整答案：\n"
        f"{guidance}\n"
        "- 将问题拆成 1~2 个子问题后再次提问"
    )


def _build_cloud_timeout_answer(
    question: str,
    retrieved: list[dict],
    trimmed_history: list[Message],
    long_term_memory: str,
    agent_key: str | None,
) -> str:
    if retrieved:
        retrieval_based = _fast_retrieval_answer(
            question,
            retrieved,
            agent_key=agent_key,
        )
        return (
            f"{retrieval_based}\n"
            "（说明：云端模型本轮响应超时，以上为基于已命中知识库证据的快速回答。）"
        )
    return _build_cloud_timeout_degraded_answer(
        question=question,
        trimmed_history=trimmed_history,
        long_term_memory=long_term_memory,
    )


def _format_retrieval_contexts_for_generation(
    retrieved: list[dict],
    limit: int = 4,
) -> list[str]:
    formatted: list[str] = []
    seen: set[str] = set()
    for item in retrieved:
        content = _clean_retrieval_snippet(str(item.get("content") or ""), 280)
        if not content:
            continue
        key = content[:120].lower()
        if key in seen:
            continue
        seen.add(key)
        formatted.append(content)
        if len(formatted) >= limit:
            break
    return formatted


def _build_unique_sources(
    retrieved: list[dict],
    limit: int = 5,
) -> list[SourceItem]:
    deduped: dict[str, dict[str, object]] = {}
    order: list[str] = []
    for item in retrieved:
        source_location = str(item.get("source_location") or "").strip()
        content = _clean_retrieval_snippet(str(item.get("content") or ""), 240)
        if not source_location or not content:
            continue
        score_raw = item.get("score")
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        key = source_location.lower()
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = {
                "source_location": source_location,
                "content": content,
                "score": score,
            }
            order.append(key)
            continue
        if score > float(existing.get("score", 0.0)):
            existing["score"] = score
            existing["content"] = content

    output: list[SourceItem] = []
    for key in order:
        row = deduped.get(key) or {}
        output.append(
            SourceItem(
                source_location=str(row.get("source_location") or ""),
                content=str(row.get("content") or ""),
                score=round(float(row.get("score") or 0.0), 4),
            )
        )
        if len(output) >= limit:
            break
    return output


def _fast_retrieval_answer(
    question: str,
    retrieved: list[dict],
    agent_key: str | None = None,
) -> str:
    style = _detect_answer_style(question)
    lowered = (question or "").lower()
    no_answer_rules = get_agent_no_answer_strategy(agent_key)
    no_answer_hint = (
        "；".join(no_answer_rules[:2]) if no_answer_rules else "补充关键事实后继续"
    )
    agent_focus = {
        "student-growth": "学习成长",
        "teacher-assistant": "课堂教学",
        "counselor-ideology": "学生事务与思政",
        "risk-warning": "学业风险预警",
        "report-assistant": "学情报告",
        "policy-qa": "政策制度问答",
    }.get((agent_key or "").strip().lower(), "知识库问答")
    if "xjtu-back" in lowered and any(
        token in question for token in ["启动", "运行", "start"]
    ):
        return (
            "结论：建议在 xjtu-back 根目录使用脚本启动，最稳妥。\n"
            "依据：仓库已提供 scripts/ops.py，可自动处理端口占用、重启与健康检查。\n"
            "建议：先执行 `python scripts/ops.py start --reload --force-stop`，"
            "再执行 `python scripts/ops.py check --probe` 验证 /health。"
        )

    if not retrieved:
        if style == "summary":
            return (
                f"当前未检索到足够依据，暂时无法输出可靠摘要（{agent_focus}）。\n"
                f"建议：{no_answer_hint}。"
            )
        if style == "comparison":
            return (
                f"当前缺少可比较资料，暂时无法给出对比结论（{agent_focus}）。\n"
                f"建议：{no_answer_hint}。"
            )
        if style == "analysis":
            return (
                f"当前证据不足，暂时无法给出可靠分析判断（{agent_focus}）。\n"
                f"建议：{no_answer_hint}。"
            )
        if style == "guidance":
            return (
                f"当前资料不足，暂时无法给出稳妥执行方案（{agent_focus}）。\n"
                f"建议：{no_answer_hint}。"
            )
        return f"当前未检索到足够依据（{agent_focus}）。\n建议：{no_answer_hint}。"

    summarize_keywords = ["总结", "重点", "概述", "本周", "新增文档"]
    if any(token in question for token in summarize_keywords):
        source_names: list[str] = []
        for item in retrieved:
            name = str(item.get("source_location") or "").strip()
            if name and name not in source_names:
                source_names.append(name)
            if len(source_names) >= 4:
                break
        refs = "、".join(source_names) if source_names else "当前检索命中文档"
        highlights: list[str] = []
        for item in retrieved[:4]:
            text = _clean_retrieval_snippet(str(item.get("content") or ""), 80)
            if text:
                highlights.append(text)
        if not highlights:
            return f"已完成资料汇总（来源：{refs}），但高质量摘要片段仍不足。"
        bullets = "\n".join(f"- {item}" for item in highlights[:4])
        return f"已基于命中资料提炼重点（来源：{refs}）：\n{bullets}"

    snippets: list[str] = []
    for item in retrieved[:3]:
        content = _clean_retrieval_snippet(str(item.get("content") or ""), 140)
        if any(
            token in content.lower()
            for token in ["create table", "insert into", " key `", "idx_"]
        ):
            continue
        if content:
            snippets.append(content[:140])
    if not snippets:
        snippets = ["已检索到相关片段，但可直接引用的高置信文本较少。"]
    focus = _compact_text(question, 42) or "当前问题"
    evidence_hint = _compact_text(snippets[0], 60)
    if _looks_like_course_selection_query(question):
        return (
            f"针对“{focus}”，选课前先核对培养方案要求，再确认学分限制、先修条件和选课时间窗口（{agent_focus}）。\n"
            "1) 先看培养方案：确认必修、限选、任选及本学期建议修读顺序；\n"
            "2) 再查系统限制：核对学分上下限、先修课程、时间冲突和课程容量；\n"
            "3) 提交前复核：确认退补改时间、是否需要学院审批、是否影响后续课程修读。\n"
            f"依据线索：{evidence_hint or '已命中相关资料'}。"
        )
    if _looks_like_service_process_query(question):
        return (
            f"针对“{focus}”，建议先做止损，再按学校流程办理（{agent_focus}）。\n"
            "1) 先挂失/冻结，避免继续损失；\n"
            "2) 再确认办理入口与材料要求；\n"
            "3) 记录办理进度，必要时联系辅导员或服务窗口协助。\n"
            f"当前依据线索：{evidence_hint or '已命中相关资料'}。"
        )
    if style == "comparison":
        return (
            f"围绕“{focus}”，更稳妥的做法是先做对比再决策（{agent_focus}）。\n"
            "建议从投入产出、可持续性和机会成本三项打分，先试运行高分方案 1-2 周，再按真实反馈调整。\n"
            f"依据线索：{evidence_hint or '已命中相关资料'}。"
        )
    if style == "analysis":
        return (
            f"关于“{focus}”，当前更关键的是先建立验证闭环，而不是一次性做最终选择（{agent_focus}）。\n"
            "可以按“目标-行动-反馈”推进：先定短周期目标，再做每周动作，最后根据结果迭代。\n"
            f"依据线索：{evidence_hint or '已命中相关资料'}。"
        )
    if style == "guidance":
        return (
            f"针对“{focus}”，建议先做短周期验证，再做二选一决策（{agent_focus}）。\n"
            "1) 每周固定时间验证不同方案；\n"
            "2) 记录投入时长、完成度和真实反馈；\n"
            "3) 到第 4 周按结果决策，不按情绪决策。\n"
            f"依据线索：{evidence_hint or '已命中相关资料'}。"
        )
    return (
        f"围绕“{focus}”，建议先做 2-4 周小步验证，再决定长期路径（{agent_focus}）。\n"
        "先用小范围实践获取反馈，再把时间投入到效果更稳定的方向上。\n"
        f"依据线索：{evidence_hint or '已命中相关资料'}。"
    )


def _detect_answer_style(question: str) -> str:
    q = (question or "").strip().lower()
    if not q:
        return "direct"
    if _looks_like_course_selection_query(q):
        return "course_selection"
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


def _is_complex_question(question: str, question_mode: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if question_mode in {QUESTION_MODE_ACADEMIC, QUESTION_MODE_CRISIS}:
        return True
    style = _detect_answer_style(question)
    if style in {"analysis", "comparison"}:
        return True
    punctuation_score = q.count("，") + q.count(",") + q.count("；") + q.count(";")
    if style == "guidance" and (len(q) >= 26 or punctuation_score >= 2):
        return True
    if style == "summary" and len(q) >= 38:
        return True
    if len(q) >= 58 or punctuation_score >= 3:
        return True
    return False


def _looks_like_decision_guidance_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    decision_markers = ["还是", "要不要", "选择", "纠结", "不知道"]
    path_markers = ["跨考", "考研", "读研", "实习", "就业", "方向", "大厂"]
    return any(marker in q for marker in decision_markers) and any(
        marker in q for marker in path_markers
    )


def _get_latest_user_question(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _ensure_conversation(
    db: Session, conversation_id: str, user_id: int | None
) -> Conversation:
    conv = db.get(Conversation, conversation_id)
    if conv is None:
        conv = Conversation(id=conversation_id, user_id=user_id, title="")
        db.add(conv)
        db.commit()
        db.refresh(conv)
    return conv


def _is_simple_greeting(question: str) -> bool:
    raw = (question or "").strip().lower()
    if not raw:
        return False
    normalized = re.sub(r"[\s\.,!?;:，。！？；：~～`'\"()\[\]{}]+", "", raw)
    if not normalized:
        return False
    greetings = {"hi", "hello", "hey", "yo", "你好", "您好", "嗨", "在吗", "在嘛"}
    if normalized in greetings:
        return True
    return len(normalized) <= 6 and normalized.startswith(("hi", "hey"))


def _format_rule_answer(question: str) -> str | None:
    lowered = question.lower()
    if _looks_like_crisis_query(question):
        return (
            "我听到你现在非常痛苦，这很重要：你不需要一个人扛。\n"
            "先做两件事：\n"
            "1) 立刻联系你信任的人（室友/同学/辅导员/家人），明确告诉对方“我现在状态很危险，需要你陪我”；\n"
            "2) 如果你有伤害自己或结束生命的冲动，请马上拨打 120 或当地24小时心理危机热线，优先保证人身安全。\n"
            "如果你愿意，我可以继续陪你，把今晚到明早的安全计划一步步写出来。"
        )
    if _is_simple_greeting(question):
        return (
            "结论：你好，我是西交 AI 智能体。\n"
            "依据：当前消息属于问候场景，不需要知识库检索。\n"
            "建议：你可以继续提问，例如“请分析我的学情风险并给建议”。"
        )
    if "帮助" in question or "help" in lowered:
        return (
            "结论：你可以直接发起知识问答。\n"
            "依据：系统支持知识库检索、文档范围限定与模型增强。\n"
            "建议：可指定问题场景并选择助手类型以提升回答质量。"
        )
    if "版本" in question or "version" in lowered:
        return (
            "结论：当前对话服务版本为 0.1.0。\n"
            "依据：系统内置版本信息。\n"
            "建议：如需新特性，请关注后续版本发布说明。"
        )
    if "xjtu-back" in lowered and any(
        token in question for token in ["启动", "运行", "start"]
    ):
        return (
            "结论：请在 xjtu-back 根目录用运维脚本启动后端。\n"
            "依据：`scripts/ops.py` 已封装 start/stop/restart/check，并处理端口冲突。\n"
            "建议：依次执行 `python scripts/ops.py start --reload --force-stop`、"
            "`python scripts/ops.py check --probe`；如需重启用 `python scripts/ops.py restart --reload`。"
        )
    return None


def _looks_like_crisis_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    markers = [
        "离开这个世界",
        "不想活",
        "活着好累",
        "想死",
        "自杀",
        "结束生命",
        "轻生",
    ]
    return any(marker in q for marker in markers)


def _estimate_tokens(text: str) -> int:
    # Lightweight token estimation for mixed Chinese and English text.
    return max(1, len(text) // 2)


def _truncate_history(
    history: list[Message], max_rounds: int, max_tokens: int
) -> list[Message]:
    if not history:
        return []
    chat_history = [
        msg
        for msg in history
        if msg.role in {"user", "assistant"} and not _is_long_term_memory_message(msg)
    ]
    if not chat_history:
        return []
    kept: list[Message] = []
    token_sum = 0
    max_messages = max_rounds * 2
    for msg in reversed(chat_history):
        tokens = _estimate_tokens(msg.content)
        if kept and (len(kept) >= max_messages or token_sum + tokens > max_tokens):
            break
        kept.append(msg)
        token_sum += tokens
    return list(reversed(kept))


def _build_retrieval_query(
    history: list[Message], question: str, long_term_memory: str = ""
) -> str:
    summary_keywords = ["总结", "概述", "重点", "本周新增文档", "学生指南"]
    if any(keyword in question for keyword in summary_keywords):
        # For summary-style requests, avoid dragging unrelated chat history.
        return question

    if not history:
        memory = _compact_text(long_term_memory, 220)
        if memory:
            return f"长期记忆: {memory}\n{_compact_text(question, 200)}"
        return question
    user_snippets = [
        _compact_text(item.content, 120)
        for item in history
        if item.role == "user" and item.content.strip()
    ]
    snippets = [item for item in user_snippets[-3:] if item]
    memory = _compact_text(long_term_memory, 220)
    if memory:
        snippets.insert(0, f"长期记忆: {memory}")
    if not snippets:
        return question
    retrieval_query = "\n".join(snippets + [_compact_text(question, 200)])
    return retrieval_query[:880]


def _expand_academic_query(agent_key: str | None, question: str) -> str:
    if not _is_student_growth_agent(agent_key):
        return question
    q = (question or "").strip()
    if not q:
        return q
    if any(token in q for token in ["学业分析", "学习分析", "成绩分析", "学情分析"]):
        return (
            f"{q}\n"
            "请重点检索：学习成绩、课堂互动、学习时长、课程完成度、薄弱课程、改进建议。"
        )
    return q


def _expand_student_growth_query(agent_key: str | None, question: str) -> str:
    if not _is_student_growth_agent(agent_key):
        return question
    q = (question or "").strip()
    if not q:
        return q

    stress_tokens = [
        "期末",
        "考试",
        "焦虑",
        "熬夜",
        "效率",
        "换届",
        "选举",
        "冲突",
        "任务太多",
        "看不进",
    ]
    if any(token in q for token in stress_tokens):
        return (
            f"{q}\n"
            "请重点检索：期末复习、时间冲突、任务优先级、社团事务协调、焦虑缓解、"
            "睡眠恢复、短周期执行计划、每日复盘。"
        )
    return q


def _is_student_growth_agent(agent_key: str | None) -> bool:
    return normalize_agent_key(agent_key) == "student-growth"


def _is_academic_analysis_query(agent_key: str | None, question: str) -> bool:
    if not _is_student_growth_agent(agent_key):
        return False
    q = (question or "").strip().lower()
    return any(
        token in q
        for token in [
            "学业分析",
            "学习分析",
            "成绩分析",
            "学情分析",
            "学业",
            "学情",
            "学习情况",
            "成绩情况",
        ]
    )


def _format_academic_analysis_context(login_name: str) -> str:
    data = get_my_academic_analysis(login_name=login_name)
    metrics = data.metrics
    course_lines = []
    for item in data.course_scores[:4]:
        course_lines.append(f"- {item.course_name}: {item.final_score:.1f}")
    warning_open = [w for w in data.warnings if (w.status or "").lower() == "open"]

    return "\n".join(
        [
            "学业分析结构化数据：",
            f"- 学生: {data.student.student_name} ({data.student.login_name})",
            f"- 学期: {data.term.term_name}",
            f"- 平均分: {metrics.avg_score if metrics.avg_score is not None else 'N/A'}",
            f"- GPA: {metrics.gpa if metrics.gpa is not None else 'N/A'}",
            f"- 风险等级: {data.risk_level}",
            f"- 未处理预警数: {len(warning_open)}",
            "- 课程成绩(前4):",
            *(course_lines or ["- 无课程成绩数据"]),
            "- 关键发现:",
            *([f"- {x}" for x in data.key_findings[:4]] or ["- 无"]),
            "- 建议:",
            *([f"- {x}" for x in data.recommendations[:4]] or ["- 无"]),
        ]
    )


def _is_guidance_query(question: str, agent_key: str | None = None) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    common_flags = ["建议", "指南", "总结", "重点", "规划", "方案", "怎么做", "如何做"]
    if any(flag in q for flag in common_flags):
        return True

    normalized_agent = normalize_agent_key(agent_key)
    agent_flags: dict[str, tuple[str, ...]] = {
        "student-growth": ("学生", "学习", "生活", "成长", "学业"),
        "teacher-assistant": ("教学", "课堂", "备课", "作业", "教案", "授课", "评价"),
        "counselor-ideology": (
            "辅导员",
            "思政",
            "谈心",
            "班级",
            "学生管理",
            "活动组织",
        ),
        "risk-warning": ("预警", "风险", "异常", "干预", "监测", "识别"),
        "report-assistant": ("报告", "汇总", "趋势", "分析", "归纳", "简报"),
    }
    return any(flag in q for flag in agent_flags.get(normalized_agent, ()))


def _looks_like_fact_lookup_query(agent_key: str | None, question: str) -> bool:
    normalized_agent = normalize_agent_key(agent_key)
    if normalized_agent == "policy-qa":
        return True
    q = (question or "").strip().lower()
    if not q:
        return False
    if normalized_agent in {"teacher-assistant", "risk-warning"} and any(
        token in q
        for token in [
            "焦虑",
            "迷茫",
            "纠结",
            "压力",
            "不知道",
            "怎么办",
            "怎么选",
            "要不要",
        ]
    ):
        return False
    fact_tokens = [
        "政策",
        "规定",
        "流程",
        "步骤",
        "材料",
        "申请",
        "办理",
        "挂失",
        "补办",
        "丢了",
        "丢失",
        "博士",
        "修业年限",
        "最长",
        "几年",
        "延期",
        "毕业要求",
        "校园卡",
        "一卡通",
        "饭卡",
        "时间",
        "地点",
        "要求",
        "学分",
        "绩点",
        "选课",
        "课程规则",
        "怎么办理",
        "是什么",
        "在哪里",
    ]
    open_tokens = [
        "焦虑",
        "纠结",
        "迷茫",
        "痛苦",
        "职业",
        "方向",
        "跨考",
        "实习",
        "思路",
    ]
    fact_hit = any(token in q for token in fact_tokens)
    if not fact_hit:
        return False
    if _looks_like_service_process_query(question):
        return True
    open_hit_count = sum(1 for token in open_tokens if token in q)
    if open_hit_count >= 2:
        return False
    if any(token in q for token in ["怎么办", "怎么处理", "如何处理"]) and any(
        token in q for token in ["办理", "流程", "步骤", "申请", "材料"]
    ):
        return True
    return True


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


def _looks_like_kb_grounded_request(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    kb_markers = ["根据知识库", "结合知识库", "按知识库", "基于知识库", "结合资料"]
    return any(marker in q for marker in kb_markers)


def _should_force_cloud_retrieval_generation(
    agent_key: str | None,
    *,
    question_mode: str,
    question: str,
    cloud_enabled: bool,
) -> bool:
    if not cloud_enabled:
        return False
    if normalize_agent_key(agent_key) != "student-growth":
        return False
    if question_mode == QUESTION_MODE_FACT:
        return True
    if question_mode == QUESTION_MODE_OPEN and _looks_like_kb_grounded_request(
        question
    ):
        return True
    return False


def _should_use_local_timeout_backup(
    *,
    agent_key: str | None,
    question_mode: str,
    question: str,
) -> bool:
    if normalize_agent_key(agent_key) != "student-growth":
        return False
    if question_mode != QUESTION_MODE_FACT:
        return False
    return _looks_like_course_selection_query(question)


def _looks_like_open_guidance_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    if _looks_like_service_process_query(question):
        return False
    open_tokens = [
        "焦虑",
        "迷茫",
        "纠结",
        "压力",
        "痛苦",
        "方向",
        "选择",
        "职业",
        "还是",
        "要不要",
        "不知道",
        "怎么办",
        "怎么选",
        "怎么做",
        "思路",
        "建议",
    ]
    fact_tokens = [
        "政策",
        "规定",
        "流程",
        "步骤",
        "材料",
        "申请",
        "办理",
        "要求",
        "时间",
        "地点",
        "学分",
        "绩点",
        "是什么",
        "多少",
    ]
    open_hits = sum(1 for token in open_tokens if token in q)
    fact_hits = sum(1 for token in fact_tokens if token in q)
    return open_hits >= 2 or (open_hits >= 1 and fact_hits == 0)


def _resolve_question_mode(agent_key: str | None, question: str) -> str:
    if _looks_like_crisis_query(question):
        return QUESTION_MODE_CRISIS
    if _is_academic_analysis_query(agent_key, question):
        return QUESTION_MODE_ACADEMIC
    if _looks_like_open_guidance_query(question):
        return QUESTION_MODE_OPEN
    if _looks_like_fact_lookup_query(agent_key, question):
        return QUESTION_MODE_FACT
    if _is_student_growth_agent(agent_key) or _is_guidance_query(question, agent_key):
        return QUESTION_MODE_OPEN
    lowered = (question or "").lower()
    return (
        QUESTION_MODE_FACT
        if any(token in lowered for token in ["什么", "what"])
        else QUESTION_MODE_OPEN
    )


def _should_use_generation_first_mode(
    *,
    question_mode: str,
    route_mode: str,
    agent_key: str | None,
    question: str,
) -> bool:
    if question_mode != QUESTION_MODE_OPEN:
        return False
    if route_mode == MODEL_ROUTE_RETRIEVAL_ONLY:
        return False
    if _is_academic_analysis_query(agent_key, question):
        return False
    if _looks_like_fact_lookup_query(agent_key, question):
        return False
    if _looks_like_service_process_query(question):
        return False
    q = (question or "").strip().lower()
    if not q:
        return False
    if _looks_like_open_guidance_query(question):
        return True
    if any(token in q for token in ["总结", "概述", "梳理", "报告", "对比", "分析"]):
        return False
    return len(q) <= 120


def _build_open_question_fallback(
    question: str,
    trimmed_history: list[Message],
    agent_key: str | None = None,
    long_term_memory: str = "",
    profile_context: str = "",
) -> str:
    normalized_agent = normalize_agent_key(agent_key)
    if normalized_agent != "student-growth":
        focus = _compact_text(question, 60) or "当前问题"
        role_plan: dict[str, tuple[str, str, str]] = {
            "teacher-assistant": (
                "先对齐本节课目标与学生现状，再组织可执行的课堂动作。",
                "先明确课堂目标和时间约束；",
                "再设计“讲解-练习-反馈”闭环，并在课后用1个指标复盘。",
            ),
            "counselor-ideology": (
                "先做风险分级，再用低冲突方式开展沟通与跟进。",
                "先确认事实和风险等级；",
                "再安排“谈心-协同-复盘”三步处置，避免先入为主定性。",
            ),
            "risk-warning": (
                "先确定风险等级与触发证据，再安排本周干预优先级。",
                "先补齐时间区间和对象范围；",
                "再按“高风险先干预、低风险先监测”推进。",
            ),
            "report-assistant": (
                "先明确报告口径，再输出可确认结论与待确认项。",
                "先锁定统计范围和时间窗口；",
                "再按“结论-依据-建议”组织报告，缺数据处显式标注。",
            ),
            "policy-qa": (
                "先给可核验的制度解释，再说明适用范围与边界。",
                "先确认政策对象、时间和适用场景；",
                "再按“结论-依据-执行建议”答复，不编造条款细节。",
            ),
        }
        lead, step1, step2 = role_plan.get(
            normalized_agent,
            (
                "先聚焦问题本身给出可执行建议，再根据补充信息细化。",
                "先明确目标和约束条件；",
                "再按最小可执行步骤推进并复盘结果。",
            ),
        )
        return (
            f"先直接回答：围绕“{focus}”，{lead}\n"
            f"1) {step1}\n"
            f"2) {step2}\n"
            "3) 你补充关键事实后，我可以给出更精确的下一步方案。"
        )

    focus = _compact_text(question, 60) or "当前问题"
    if _looks_like_club_overload_query(question):
        return (
            f"先直接回答：像“{focus}”这种情况，真正要优先保的是学业，你现在不是“得罪人”，而是在给自己恢复边界。\n"
            "1) 先做取舍：学生会和两个大社团不可能长期同时高投入，至少要砍掉一项核心承诺；\n"
            "2) 不要突然失联，直接和负责人说“开学初判断失误，学业已经明显受影响，需要从高频事务中退出或降频”；\n"
            "3) 先提出替代方案：把手头任务交接清楚、给出过渡时间，这样比硬拖到彻底崩掉更负责；\n"
            "4) 从这周开始固定晚间两到三个时段只留给上课、自习和作业，社团活动只能占剩余时间。"
        )
    if _looks_like_doctoral_extension_query(question):
        return (
            f"先直接回答：像“{focus}”这种博士一年级就担心延期和毕业的情况，很常见，但现在最重要的不是提前判自己毕不了业，而是先把“最长修业年限 + 导师预期 + 接下来3个月里程碑”这三件事尽快确认清楚。\n"
            "1) 先问学院研究生秘书或培养办：确认博士最长修业年限、延期条件和学位申请基本要求；\n"
            "2) 再和导师单独谈一次：不要泛泛说焦虑，直接把“目前卡点、需要的支持、三个月目标”摆出来；\n"
            "3) 把目标从“几年内发够文章”改成“本学期先解决一个核心研究问题，产出一个稳定小结果”；\n"
            "4) 如果已经焦虑到明显影响睡眠和身体状态，就不要硬扛，尽快找校内心理咨询或辅导员做支持。"
        )
    if _looks_like_learning_support_query(question):
        return (
            f"先直接回答：像“{focus}”这种已经出现明显挂科风险的情况，不建议你继续一个人死磕，最好马上把“老师答疑 + 学校辅导资源 + 同伴帮扶”三条线同时拉起来。\n"
            "1) 先找任课教师或助教，尽快问清楚期中失分点、期末重点和补救顺序；\n"
            "2) 再找学院或辅导员，直接问有没有官方学业帮扶、答疑安排、朋辈辅导或补习资源；\n"
            "3) 同步找一位学得好的同学或学长学姐，先带你补最基础的章节和题型；\n"
            "4) 接下来两周只抓最可能决定及格的核心知识点和高频题型。"
        )
    if _looks_like_stress_conflict_query(question):
        return (
            f"先直接回答：像“{focus}”这种考试和社团事务撞车的情况，你现在最需要的是先止损，而不是继续硬扛。\n"
            "1) 今晚只保留一个最关键学习任务，其他任务全部顺延，不再继续熬夜；\n"
            "2) 明天把社团事务分成“必须亲自做、可以委托、可以延后”三类，能交接的立刻交接；\n"
            "3) 复习时只抓最影响期末结果的课程和题型，每次 45 分钟，中间强制休息；\n"
            "4) 如果连续两天效率仍然很差，就先压缩社团投入，优先保考试和睡眠。"
        )
    if _looks_like_decision_guidance_query(question):
        return (
            f"先直接回答：对于“{focus}”，先不要急着二选一，建议用 2-4 周做双轨验证。\n"
            "1) 给长期路线和短期路线各留一个固定时间块；\n"
            "2) 一条路线看你能不能持续投入，另一条路线看外部反馈是否更好；\n"
            "3) 每周末只看三个指标：投入时长、完成度、反馈强弱；\n"
            "4) 第4周再决定把主要精力压到哪一条路。"
        )
    hints: list[str] = []
    recent_user_questions = [
        _compact_text(item.content, 44)
        for item in trimmed_history
        if item.role == "user" and item.content.strip()
    ]
    if recent_user_questions:
        hints.append(f"你最近最关心的是：{recent_user_questions[-1]}")
    if long_term_memory:
        hints.append(f"长期约束：{_compact_text(long_term_memory, 60)}")
    if profile_context:
        hints.append(f"可用背景：{_compact_text(profile_context, 60)}")
    hint_line = f"\n可用背景：{'；'.join(hints[:2])}。" if hints else ""
    return (
        f"先直接回答你这个问题：围绕“{focus}”，现在更重要的不是立刻找到唯一正确答案，而是先确定接下来 7 天最值得验证的一步。\n"
        "建议你按这个顺序行动：\n"
        "1) 先写下你最在意的 2 个决策标准；\n"
        "2) 本周只验证 1 个最关键假设，不同时摊开太多选择；\n"
        "3) 用真实反馈而不是当下情绪做下一轮决定。"
        f"{hint_line}"
    )


def _looks_like_stress_conflict_query(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    stress_markers = ["焦虑", "熬夜", "效率低", "看不进", "很累", "崩溃"]
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
    return any(marker in q for marker in org_markers) and any(
        marker in q for marker in overload_markers
    )


def _build_scope_text(kb_ids: list[str], document_ids: list[str] | None) -> str:
    if document_ids:
        return f"当前勾选的 {len(document_ids)} 份文档"
    if kb_ids:
        return f"{len(kb_ids)} 个知识库"
    return "默认检索范围"


def _build_summary_thinking(
    trimmed_history: list[Message],
    kb_ids: list[str],
    document_ids: list[str] | None,
    route_mode: str,
    question_mode: str,
    retrieved_count: int,
    top_k: int,
    score_threshold: float,
    rule_answer: bool,
    llm_result: LLMAnswerResult | None,
    profile_ms: int,
    retrieval_ms: int,
    llm_ms: int,
    workflow_wait_ms: int,
    workflow_stage: str,
    total_ms: int,
) -> ChatThinking:
    steps: list[str] = []
    rounds = max(1, (len(trimmed_history) + 1) // 2) if trimmed_history else 0
    cloud_direct_mode = route_mode == MODEL_ROUTE_CLOUD_ONLY

    if rule_answer:
        steps.append("已识别当前问题属于规则类场景，并直接给出可执行答复。")
    elif cloud_direct_mode:
        if rounds > 0:
            steps.append(
                f"已理解你的问题重点，并结合最近 {rounds} 轮对话补充必要上下文。"
            )
        else:
            steps.append("已理解你的问题重点，并聚焦当前提问生成回答。")
        if llm_result is not None and llm_result.mode in {"llm", "llm_retry"}:
            if retrieved_count > 0:
                steps.append("已结合知识库证据组织回答，并完成云端表达优化。")
            else:
                steps.append("已完成云端模型回答生成，并对表述做了可读性整理。")
        elif llm_result is not None and "fallback" in llm_result.mode:
            steps.append("云端响应出现波动，已先给出稳妥的可执行答复，并可继续细化。")
        else:
            steps.append("已完成问题处理并返回当前最佳回答。")
    else:
        scope_text = _build_scope_text(kb_ids, document_ids)
        if retrieved_count > 0:
            steps.append(
                f"已在{scope_text}中检索到 {retrieved_count} 条相关资料，并提炼关键线索。"
            )
            steps.append("已结合知识库证据组织回答，确保内容与知识库信息一致。")
        else:
            steps.append(
                f"已在{scope_text}完成检索，但当前问题缺少可直接支撑的高置信资料。"
            )
            steps.append("已基于现有资料范围给出保守回答，并提示补充所需信息。")
        if llm_result is not None and llm_result.mode.startswith("local_transformer"):
            steps.append("已由本地模型基于检索资料生成分段回答。")
        elif llm_result is not None and "fallback" in llm_result.mode:
            steps.append("生成阶段出现波动，已切换为稳定的知识库约束答复。")

    total_seconds = max(0.1, round(total_ms / 1000, 1))
    steps.append(f"本次处理约耗时 {total_seconds} 秒。")
    if workflow_stage and workflow_stage not in {"init", "done"}:
        steps.append(f"当前进度：{workflow_stage} 阶段已完成。")
    steps.append("以上为可公开展示的处理过程，不包含模型私有推理内容。")
    return ChatThinking(
        title="处理摘要",
        content="\n".join(f"- {step}" for step in steps),
        kind="summary",
        is_real=False,
        collapsed=True,
    )


def _trim_public_reasoning(reasoning: str, max_chars: int = 900) -> str:
    text = (reasoning or "").replace("\r\n", "\n").strip()
    if not text:
        return ""
    text = _strip_internal_template_text(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rstrip()
    return f"{trimmed}\n\n（思考内容较长，已截断展示）"


def _build_thinking_payload(
    trimmed_history: list[Message],
    kb_ids: list[str],
    document_ids: list[str] | None,
    route_mode: str,
    question_mode: str,
    retrieved_count: int,
    top_k: int,
    score_threshold: float,
    rule_answer: bool,
    llm_result: LLMAnswerResult | None,
    profile_ms: int,
    retrieval_ms: int,
    llm_ms: int,
    workflow_wait_ms: int,
    workflow_stage: str,
    total_ms: int,
) -> ChatThinking:
    if (
        llm_result
        and llm_result.reasoning
        and not _looks_like_private_reasoning_text(llm_result.reasoning)
    ):
        public_reasoning = _trim_public_reasoning(llm_result.reasoning)
        return ChatThinking(
            title="思考过程",
            content=public_reasoning,
            kind="reasoning",
            is_real=True,
            collapsed=True,
        )
    return _build_summary_thinking(
        trimmed_history=trimmed_history,
        kb_ids=kb_ids,
        document_ids=document_ids,
        route_mode=route_mode,
        question_mode=question_mode,
        retrieved_count=retrieved_count,
        top_k=top_k,
        score_threshold=score_threshold,
        rule_answer=rule_answer,
        llm_result=llm_result,
        profile_ms=profile_ms,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        workflow_wait_ms=workflow_wait_ms,
        workflow_stage=workflow_stage,
        total_ms=total_ms,
    )


def chat_completion(
    db: Session,
    payload: ChatCompletionRequest,
    current_user: User | None,
    progress_callback: ProgressCallback | None = None,
) -> ChatCompletionResult:
    raw_question = _get_latest_user_question([m.model_dump() for m in payload.messages])
    sensitive_words = get_sensitive_words(db)
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    blocked_word = detect_sensitive_text(raw_question, sensitive_words)
    if blocked_word:
        log_sensitive_block(
            db=db,
            user_id=current_user.id if current_user else None,
            conversation_id=conversation_id,
            agent_key=payload.agent_key,
            direction="input",
            blocked_word=blocked_word,
            content=raw_question,
        )
        raise BusinessError(
            f"输入内容命中敏感词，已拦截（{blocked_word}）。",
            status_code=400,
        )

    question = raw_question.strip()
    route_mode, cloud_enabled, local_enabled = _resolve_model_route_mode(payload)
    if route_mode == MODEL_ROUTE_CLOUD_ONLY:
        # Cloud direct mode should not depend on bound knowledge-base availability.
        kb_ids = payload.kb_ids or []
        document_ids = payload.document_ids
    else:
        resolved_scope = resolve_agent_bound_kb_scope(
            db,
            agent_key=payload.agent_key,
            document_ids=payload.document_ids,
        )
        if resolved_scope is not None:
            kb_ids, document_ids = resolved_scope
        else:
            kb_ids = payload.kb_ids or [
                item.id
                for item in db.scalars(
                    select(KnowledgeBase).where(KnowledgeBase.status == "active")
                ).all()
            ]
            document_ids = payload.document_ids
    retrieval_config = get_effective_retrieval_config(
        db=db,
        conversation_id=conversation_id,
        payload_top_k=payload.top_k,
        payload_score_threshold=payload.score_threshold,
        payload_fusion_mode=payload.fusion_mode,
        payload_alpha=payload.alpha,
    )
    top_k = int(retrieval_config["retrieval_top_k"])
    score_threshold = float(retrieval_config["score_threshold"])
    fusion_mode = str(retrieval_config["fusion_mode"])
    alpha = float(retrieval_config["alpha"])

    if _is_guidance_query(question, payload.agent_key):
        # Guidance tasks need enough recall coverage; avoid over-aggressive truncation.
        top_k = min(max(top_k, 4), 12)
        score_threshold = min(score_threshold, 0.24)

    context_cfg = get_config_values(
        db,
        ["context_max_rounds", "context_max_tokens"],
    )
    try:
        configured_max_rounds = int(
            context_cfg.get("context_max_rounds", DEFAULT_CONTEXT_MAX_ROUNDS)
        )
    except (TypeError, ValueError):
        configured_max_rounds = DEFAULT_CONTEXT_MAX_ROUNDS
    try:
        configured_max_tokens = int(
            context_cfg.get("context_max_tokens", DEFAULT_CONTEXT_MAX_TOKENS)
        )
    except (TypeError, ValueError):
        configured_max_tokens = DEFAULT_CONTEXT_MAX_TOKENS
    max_rounds = payload.context_max_rounds or configured_max_rounds
    max_tokens = payload.context_max_tokens or configured_max_tokens

    _ensure_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id if current_user else None,
    )
    history = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        ).all()
    )
    long_term_memory = _extract_long_term_memory(history)
    trimmed_history = _truncate_history(
        history=history, max_rounds=max_rounds, max_tokens=max_tokens
    )
    retrieval_query_cache: dict[str, str] = {}
    retrieval_result_cache: dict[tuple[str, int, float], list[dict]] = {}
    history_context_cache: dict[str, str] = {}

    def _resolve_history_context(runtime_question: str) -> str:
        key = (runtime_question or "").strip()
        if not key:
            return ""
        cached_summary = history_context_cache.get(key)
        if cached_summary is not None:
            return cached_summary
        summary = _build_history_context_brief(
            trimmed_history,
            key,
            long_term_memory=long_term_memory,
        )
        history_context_cache[key] = summary
        return summary

    def _resolve_retrieval_query(runtime_question: str) -> str:
        key = (runtime_question or "").strip()
        if not key:
            return ""
        cached_query = retrieval_query_cache.get(key)
        if cached_query is not None:
            return cached_query
        if (
            route_mode == MODEL_ROUTE_CLOUD_ONLY
            and not force_cloud_retrieval_generation
        ):
            retrieval_query_cache[key] = key
            return key
        if not is_complex_query and len(key) <= 26:
            retrieval_query_cache[key] = key
            return key
        if _looks_like_service_process_query(key) or _looks_like_fact_lookup_query(
            payload.agent_key, key
        ):
            retrieval_query_cache[key] = key
            return key
        built_query = _expand_academic_query(
            payload.agent_key,
            _build_retrieval_query(trimmed_history, key, long_term_memory),
        )
        built_query = _expand_student_growth_query(payload.agent_key, built_query)
        retrieval_query_cache[key] = built_query
        return built_query

    def _run_retrieval_query(
        runtime_query: str,
        runtime_top_k: int,
        runtime_threshold: float,
    ) -> list[dict]:
        cache_key = (
            runtime_query,
            runtime_top_k,
            round(runtime_threshold, 4),
        )
        cached_result = retrieval_result_cache.get(cache_key)
        if cached_result is not None:
            return list(cached_result)
        output = hybrid_retrieve(
            db=db,
            query=runtime_query,
            kb_ids=kb_ids,
            document_ids=document_ids,
            top_k=runtime_top_k,
            score_threshold=runtime_threshold,
            fusion_mode=fusion_mode,
            alpha=alpha,
            agent_key=payload.agent_key,
        )
        retrieval_result_cache[cache_key] = list(output)
        return output

    question_mode = _resolve_question_mode(payload.agent_key, question)
    is_academic_analysis = question_mode == QUESTION_MODE_ACADEMIC
    is_complex_query = _is_complex_question(question, question_mode)
    force_cloud_retrieval_generation = _should_force_cloud_retrieval_generation(
        payload.agent_key,
        question_mode=question_mode,
        question=question,
        cloud_enabled=cloud_enabled,
    )

    if question_mode == QUESTION_MODE_FACT:
        top_k = min(max(top_k, 2), 4 if is_complex_query else 3)
        score_threshold = max(score_threshold, 0.24 if is_complex_query else 0.28)
    elif question_mode == QUESTION_MODE_OPEN and route_mode != MODEL_ROUTE_CLOUD_ONLY:
        if is_complex_query:
            top_k = min(max(top_k, 4), 10)
            score_threshold = min(score_threshold, 0.24)
        else:
            top_k = min(max(top_k, 3), 6)
            score_threshold = max(score_threshold, 0.22)
    elif question_mode == QUESTION_MODE_CRISIS:
        top_k = min(max(top_k, 3), 6)
        score_threshold = min(score_threshold, 0.24)

    generation_first_mode = _should_use_generation_first_mode(
        question_mode=question_mode,
        route_mode=route_mode,
        agent_key=payload.agent_key,
        question=question,
    )
    # Explicit mode split:
    # - Cloud mode: direct LLM answer (KB retrieval is optional, not mandatory).
    # - Local mode: retrieval-enhanced answering only.
    if route_mode == MODEL_ROUTE_CLOUD_ONLY and cloud_enabled:
        generation_first_mode = not force_cloud_retrieval_generation
    elif route_mode == MODEL_ROUTE_LOCAL_ONLY:
        generation_first_mode = False
    chat_total_timeout = CHAT_TOTAL_TIMEOUT_SECONDS
    workflow_timeout_cap = WORKFLOW_TIMEOUT_SECONDS
    generation_timeout = GENERATION_STAGE_TIMEOUT_SECONDS
    profile_timeout_seconds = (
        ACADEMIC_PROFILE_TIMEOUT_SECONDS
        if is_academic_analysis
        else DEFAULT_PROFILE_TIMEOUT_SECONDS
    )

    logger.info(
        "chat time budget: conversation=%s agent=%s route=%s mode=%s cloud=%s local=%s total=%ss workflow_cap=%ss generation=%ss top_k=%s threshold=%.3f",
        conversation_id,
        payload.agent_key,
        route_mode,
        "complex" if is_complex_query else "simple",
        cloud_enabled,
        local_enabled,
        chat_total_timeout,
        workflow_timeout_cap,
        generation_timeout,
        top_k,
        score_threshold,
    )

    start = perf_counter()
    llm_result: LLMAnswerResult | None = None
    retrieved: list[dict] = []
    profile_context_text = ""
    profile_ms = 0
    retrieval_ms = 0
    llm_ms = 0
    workflow_wait_ms = 0
    workflow_stage = "init"
    rule_answer_text = _format_rule_answer(question)
    if rule_answer_text:
        answer = rule_answer_text
        sources: list[SourceItem] = []
        system_instruction = ""
        _emit_progress(
            progress_callback,
            type="stage",
            stage="rule",
            status="done",
            detail="命中内置规则回答，已直接返回结果。",
        )
    else:

        def _load_profile(_: str) -> str:
            nonlocal profile_ms, workflow_stage, profile_context_text
            workflow_stage = "profile"
            if current_user is None:
                profile_context_text = ""
                return ""
            if not is_academic_analysis and not needs_profile_context(
                payload.agent_key
            ):
                profile_context_text = ""
                return ""
            stage_start = perf_counter()
            _emit_progress(
                progress_callback,
                type="stage",
                stage="profile",
                status="start",
                detail="正在加载用户画像上下文...",
            )
            try:
                if route_mode == MODEL_ROUTE_CLOUD_ONLY:
                    profile_context_text = ""
                    return ""
                profile_executor = ThreadPoolExecutor(max_workers=1)
                profile_future = (
                    profile_executor.submit(
                        _format_academic_analysis_context,
                        current_user.login_name,
                    )
                    if is_academic_analysis
                    else profile_executor.submit(
                        load_user_profile_context,
                        current_user.login_name,
                    )
                )
                try:
                    profile_context_text = profile_future.result(
                        timeout=profile_timeout_seconds
                    )
                    return profile_context_text
                except FuturesTimeoutError:
                    logger.warning(
                        "profile timeout: conversation=%s agent=%s academic=%s",
                        conversation_id,
                        payload.agent_key,
                        is_academic_analysis,
                    )
                    _emit_progress(
                        progress_callback,
                        type="stage",
                        stage="profile",
                        status="timeout",
                        detail="用户画像加载超时，已继续后续流程。",
                    )
                    profile_context_text = ""
                    return ""
                finally:
                    profile_future.cancel()
                    profile_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                _emit_progress(
                    progress_callback,
                    type="stage",
                    stage="profile",
                    status="error",
                    detail="用户画像加载失败，已继续后续流程。",
                )
                profile_context_text = ""
                return ""
            finally:
                stage_ms = int((perf_counter() - stage_start) * 1000)
                profile_ms += stage_ms
                _emit_progress(
                    progress_callback,
                    type="stage",
                    stage="profile",
                    status="done",
                    stage_ms=stage_ms,
                )

        def _retrieve(
            runtime_question: str,
            *,
            runtime_top_k: int | None = None,
            runtime_threshold: float | None = None,
            allow_relaxed_retry: bool = True,
            detail: str = "正在检索知识库资料...",
        ) -> tuple[list[str], list[dict]]:
            nonlocal retrieved, retrieval_ms, workflow_stage
            workflow_stage = "retrieval"
            stage_start = perf_counter()
            retrieval_skipped = False
            effective_top_k = runtime_top_k or top_k
            effective_threshold = (
                runtime_threshold if runtime_threshold is not None else score_threshold
            )
            _emit_progress(
                progress_callback,
                type="stage",
                stage="retrieval",
                status="start",
                detail=detail,
            )
            try:
                if (
                    route_mode == MODEL_ROUTE_CLOUD_ONLY
                    and cloud_enabled
                    and not force_cloud_retrieval_generation
                ):
                    retrieval_skipped = True
                    retrieved = []
                else:
                    query = _resolve_retrieval_query(runtime_question)
                    retrieved = _run_retrieval_query(
                        query,
                        effective_top_k,
                        effective_threshold,
                    )
                    if allow_relaxed_retry and not retrieved:
                        relaxed_top_k = min(8, max(effective_top_k + 2, 3))
                        relaxed_threshold = max(
                            MIN_RELAXED_THRESHOLD,
                            min(0.24, effective_threshold - 0.06),
                        )
                        if (
                            relaxed_top_k > effective_top_k
                            or relaxed_threshold < effective_threshold
                        ):
                            retrieved = _run_retrieval_query(
                                query,
                                relaxed_top_k,
                                relaxed_threshold,
                            )
            except Exception:
                logger.exception(
                    "retrieval failed: conversation=%s agent=%s",
                    conversation_id,
                    payload.agent_key,
                )
                retrieved = []
            finally:
                stage_ms = int((perf_counter() - stage_start) * 1000)
                retrieval_ms += stage_ms
                preview = ""
                if retrieved:
                    first_hit = _clean_retrieval_snippet(
                        str(retrieved[0].get("content") or ""), 80
                    )
                    preview = f"已命中 {len(retrieved)} 条相关资料，正在组织答案。" + (
                        f"\n参考片段：{first_hit}" if first_hit else ""
                    )
                else:
                    if retrieval_skipped:
                        preview = "已完成问题理解与上下文整理，正在生成回答。"
                    elif route_mode == MODEL_ROUTE_CLOUD_ONLY:
                        preview = "已完成信息准备，正在生成回答。"
                    elif route_mode in {MODEL_ROUTE_HYBRID, MODEL_ROUTE_LOCAL_ONLY}:
                        preview = "高置信检索命中不足，将按知识库约束策略返回。"
                    else:
                        preview = "高置信检索命中不足，正在返回检索兜底结果。"
                _emit_progress(
                    progress_callback,
                    type="stage",
                    stage="retrieval",
                    status="done",
                    count=len(retrieved),
                    stage_ms=stage_ms,
                    preview=preview,
                )
            return _format_retrieval_contexts_for_generation(retrieved), retrieved

        def _attempt_local_timeout_backup_for_question(
            runtime_question: str,
        ) -> LLMAnswerResult | None:
            if not local_enabled:
                return None
            if not _should_use_local_timeout_backup(
                agent_key=payload.agent_key,
                question_mode=question_mode,
                question=runtime_question,
            ):
                return None
            local_timeout = min(18, max(10, generation_timeout // 4))
            local_course_instruction = (
                "围绕选课问题输出：先给选课要求，再写学分限制、培养方案、先修条件、"
                "选课时间窗口与注意事项；禁止输出挂失/补办/冻结等通用事务流程。"
            )
            local_contexts = _format_retrieval_contexts_for_generation(retrieved)[:3]
            local_executor = ThreadPoolExecutor(max_workers=1)
            local_future = local_executor.submit(
                generate_answer_with_local_transformer,
                runtime_question,
                local_contexts,
                settings.local_transformer_model,
                None,
                360,
                local_course_instruction,
                bool(retrieved),
            )
            try:
                answer, model_reference, _metrics = local_future.result(
                    timeout=local_timeout
                )
                if _looks_like_low_quality_local_answer(answer):
                    return None
                return LLMAnswerResult(
                    answer=answer,
                    mode=f"local_transformer:timeout_backup:{model_reference}",
                )
            except Exception:
                logger.warning(
                    "local timeout backup failed: conversation=%s agent=%s",
                    conversation_id,
                    payload.agent_key,
                    exc_info=True,
                )
                return None
            finally:
                local_future.cancel()
                local_executor.shutdown(wait=False, cancel_futures=True)

        def _generate(
            runtime_question: str,
            contexts: list[str],
            system_instruction: str,
        ) -> str:
            nonlocal llm_result, llm_ms, workflow_stage
            workflow_stage = "generation"
            stage_start = perf_counter()
            _emit_progress(
                progress_callback,
                type="stage",
                stage="generation",
                status="start",
                detail="正在生成最终回答...",
            )
            retrieval_contexts = (
                contexts[:4]
                if contexts
                else _format_retrieval_contexts_for_generation(retrieved)
            )
            force_retrieval_generation = _should_force_cloud_retrieval_generation(
                payload.agent_key,
                question_mode=question_mode,
                question=runtime_question,
                cloud_enabled=cloud_enabled,
            )
            cloud_direct_mode = (
                route_mode == MODEL_ROUTE_CLOUD_ONLY
                and cloud_enabled
                and not force_retrieval_generation
            )
            history_context = (
                _resolve_history_context(runtime_question)
                if (
                    question_mode != QUESTION_MODE_FACT
                    and (not cloud_direct_mode or is_complex_query)
                )
                else ""
            )
            if cloud_direct_mode:
                retrieval_contexts = []
            background_contexts: list[str] = []
            if question_mode != QUESTION_MODE_FACT and history_context:
                history_max_chars = (
                    220
                    if (cloud_direct_mode and is_complex_query)
                    else 140
                    if cloud_direct_mode
                    else 360
                )
                background_contexts.append(
                    f"[会话背景] {_compact_text(history_context, history_max_chars)}"
                )
            if question_mode != QUESTION_MODE_FACT and profile_context_text:
                background_contexts.append(
                    f"[用户画像] {_compact_text(profile_context_text, 320)}"
                )
            if not background_contexts and contexts:
                for item in contexts[:2]:
                    compact = _compact_text(item, 280)
                    if compact:
                        background_contexts.append(compact)
            prefer_model_answer = cloud_direct_mode or (
                generation_first_mode
                and not is_academic_analysis
                and not bool(retrieved)
            )
            if prefer_model_answer:
                generation_contexts = background_contexts[
                    : (3 if is_complex_query else 1)
                ]
                llm_retrieval_contexts: list[str] = []
            elif (
                route_mode in {MODEL_ROUTE_CLOUD_ONLY, MODEL_ROUTE_HYBRID}
                and cloud_enabled
            ):
                evidence_limit = 3 if force_retrieval_generation else 1
                llm_retrieval_contexts = retrieval_contexts[:evidence_limit]
                generation_contexts = (
                    llm_retrieval_contexts + background_contexts[:1]
                    if force_retrieval_generation
                    else (background_contexts + llm_retrieval_contexts)
                )[:4]
            else:
                generation_contexts = (
                    (retrieval_contexts + background_contexts)[:4]
                    if (retrieval_contexts or background_contexts)
                    else []
                )
                llm_retrieval_contexts = retrieval_contexts

            def _attempt_local_timeout_backup() -> LLMAnswerResult | None:
                return _attempt_local_timeout_backup_for_question(runtime_question)

            def _attempt_local_open_backup() -> LLMAnswerResult | None:
                if not local_transformer_backup_available():
                    return None
                local_timeout = min(6, max(4, generation_timeout // 5))
                local_executor = ThreadPoolExecutor(max_workers=1)
                local_future = local_executor.submit(
                    generate_answer_with_local_transformer,
                    runtime_question,
                    background_contexts[:2],
                    settings.local_transformer_model,
                    None,
                    220,
                    system_instruction,
                    False,
                )
                try:
                    answer, model_reference, _metrics = local_future.result(
                        timeout=local_timeout
                    )
                    return LLMAnswerResult(
                        answer=answer,
                        mode=f"open_cloud_timeout_local_backup:{model_reference}",
                    )
                except Exception:
                    logger.warning(
                        "open local backup failed: conversation=%s agent=%s",
                        conversation_id,
                        payload.agent_key,
                        exc_info=True,
                    )
                    return None
                finally:
                    local_future.cancel()
                    local_executor.shutdown(wait=False, cancel_futures=True)

            try:
                if (
                    route_mode == MODEL_ROUTE_LOCAL_ONLY
                    and not retrieved
                    and not prefer_model_answer
                ):
                    llm_result = LLMAnswerResult(
                        answer=_build_kb_bounded_shortfall_answer(
                            runtime_question,
                            trimmed_history,
                            route_mode=route_mode,
                            agent_key=payload.agent_key,
                        ),
                        mode=f"{route_mode}:kb_shortfall",
                    )
                elif (
                    route_mode == MODEL_ROUTE_HYBRID
                    and not retrieved
                    and cloud_enabled
                    and not prefer_model_answer
                ):
                    # Hybrid mode fallback: no KB hit -> let cloud model answer naturally.
                    llm_result = answer_with_llm(
                        question=runtime_question,
                        contexts=background_contexts,
                        llm_enabled=True,
                        system_instruction="",
                        agent_key=payload.agent_key,
                        timeout_seconds=generation_timeout,
                        allow_general_knowledge=True,
                        kb_hit=False,
                        retrieval_contexts=[],
                        background_contexts=background_contexts,
                    )
                elif (
                    route_mode == MODEL_ROUTE_HYBRID
                    and not retrieved
                    and not prefer_model_answer
                ):
                    # Defensive branch (normally hybrid always has cloud enabled).
                    llm_result = LLMAnswerResult(
                        answer=_build_kb_bounded_shortfall_answer(
                            runtime_question,
                            trimmed_history,
                            route_mode=route_mode,
                            agent_key=payload.agent_key,
                        ),
                        mode=f"{route_mode}:kb_shortfall",
                    )
                elif route_mode == MODEL_ROUTE_HYBRID and cloud_enabled:
                    llm_result = answer_with_llm(
                        question=runtime_question,
                        contexts=generation_contexts,
                        llm_enabled=True,
                        system_instruction=system_instruction,
                        agent_key=payload.agent_key,
                        timeout_seconds=generation_timeout,
                        allow_general_knowledge=prefer_model_answer,
                        kb_hit=False if prefer_model_answer else bool(retrieved),
                        retrieval_contexts=llm_retrieval_contexts,
                        background_contexts=background_contexts,
                    )
                    if (
                        route_mode == MODEL_ROUTE_HYBRID
                        and prefer_model_answer
                        and llm_result.mode
                        in {
                            "timeout_fallback",
                            "error_fallback",
                        }
                    ):
                        backup_result = _attempt_local_open_backup()
                        if backup_result is not None:
                            llm_result = backup_result
                elif local_enabled and route_mode == MODEL_ROUTE_LOCAL_ONLY:
                    local_timeout = max(
                        6,
                        min(LOCAL_GENERATION_TIMEOUT_SECONDS, generation_timeout - 2),
                    )
                    try:
                        runtime = local_transformer_runtime()
                        if str(runtime.get("active_device") or "").lower() != "cuda":
                            local_timeout = min(
                                max(local_timeout, 16),
                                24 if is_complex_query else 20,
                            )
                    except Exception:
                        logger.debug("local runtime check failed", exc_info=True)
                    local_executor = ThreadPoolExecutor(max_workers=1)
                    local_future = local_executor.submit(
                        generate_answer_with_local_transformer,
                        runtime_question,
                        generation_contexts[:4],
                        settings.local_transformer_model,
                        None,
                        320 if prefer_model_answer else 520,
                        system_instruction,
                        False if prefer_model_answer else bool(retrieved),
                    )
                    try:
                        answer, model_reference, _metrics = local_future.result(
                            timeout=local_timeout
                        )
                        if _looks_like_low_quality_local_answer(answer):
                            recovered_answer = (
                                _fast_retrieval_answer(
                                    runtime_question,
                                    retrieved,
                                    agent_key=payload.agent_key,
                                )
                                if retrieved
                                else _build_kb_bounded_shortfall_answer(
                                    runtime_question,
                                    trimmed_history,
                                    route_mode=route_mode,
                                    agent_key=payload.agent_key,
                                )
                            )
                            llm_result = LLMAnswerResult(
                                answer=recovered_answer,
                                mode=f"local_transformer:low_quality_rewrite:{model_reference}",
                            )
                        else:
                            llm_result = LLMAnswerResult(
                                answer=answer,
                                mode=f"local_transformer:{model_reference}",
                            )
                    except FuturesTimeoutError:
                        logger.warning(
                            "local transformer timeout: conversation=%s agent=%s",
                            conversation_id,
                            payload.agent_key,
                        )
                        if route_mode == MODEL_ROUTE_LOCAL_ONLY:
                            llm_result = LLMAnswerResult(
                                answer=(
                                    _build_open_question_fallback(
                                        runtime_question,
                                        trimmed_history,
                                        agent_key=payload.agent_key,
                                        long_term_memory=long_term_memory,
                                        profile_context=profile_context_text,
                                    )
                                    if prefer_model_answer
                                    else _fast_retrieval_answer(
                                        runtime_question,
                                        retrieved,
                                        agent_key=payload.agent_key,
                                    )
                                ),
                                mode=(
                                    "local_transformer:open_timeout_fallback"
                                    if prefer_model_answer
                                    else "local_transformer:timeout_degraded"
                                ),
                            )
                        else:
                            llm_result = answer_with_llm(
                                question=runtime_question,
                                contexts=generation_contexts,
                                llm_enabled=True,
                                system_instruction=system_instruction,
                                agent_key=payload.agent_key,
                                timeout_seconds=generation_timeout,
                                allow_general_knowledge=prefer_model_answer,
                                kb_hit=False
                                if prefer_model_answer
                                else bool(retrieved),
                                retrieval_contexts=llm_retrieval_contexts,
                                background_contexts=background_contexts,
                            )
                    except Exception:
                        logger.exception(
                            "local transformer failed: conversation=%s agent=%s",
                            conversation_id,
                            payload.agent_key,
                        )
                        if route_mode == MODEL_ROUTE_LOCAL_ONLY:
                            llm_result = LLMAnswerResult(
                                answer=(
                                    _build_open_question_fallback(
                                        runtime_question,
                                        trimmed_history,
                                        agent_key=payload.agent_key,
                                        long_term_memory=long_term_memory,
                                        profile_context=profile_context_text,
                                    )
                                    if prefer_model_answer
                                    else _fast_retrieval_answer(
                                        runtime_question,
                                        retrieved,
                                        agent_key=payload.agent_key,
                                    )
                                ),
                                mode=(
                                    "local_transformer:open_error_fallback"
                                    if prefer_model_answer
                                    else "local_transformer:error_degraded"
                                ),
                            )
                        else:
                            llm_result = answer_with_llm(
                                question=runtime_question,
                                contexts=generation_contexts,
                                llm_enabled=True,
                                system_instruction=system_instruction,
                                agent_key=payload.agent_key,
                                timeout_seconds=generation_timeout,
                                allow_general_knowledge=prefer_model_answer,
                                kb_hit=False
                                if prefer_model_answer
                                else bool(retrieved),
                                retrieval_contexts=llm_retrieval_contexts,
                                background_contexts=background_contexts,
                            )
                    finally:
                        local_future.cancel()
                        local_executor.shutdown(wait=False, cancel_futures=True)
                elif cloud_enabled:
                    allow_general_mode = (
                        True if cloud_direct_mode else prefer_model_answer
                    )
                    kb_hit_flag = (
                        False
                        if (prefer_model_answer or cloud_direct_mode)
                        else bool(retrieved)
                    )
                    cloud_system_instruction = (
                        build_agent_cloud_instruction(payload.agent_key)
                        if cloud_direct_mode
                        else (
                            ""
                            if (
                                allow_general_mode
                                and not kb_hit_flag
                                and not prefer_model_answer
                            )
                            else system_instruction
                        )
                    )
                    llm_result = answer_with_llm(
                        question=runtime_question,
                        contexts=generation_contexts,
                        llm_enabled=cloud_enabled,
                        system_instruction=cloud_system_instruction,
                        agent_key=payload.agent_key,
                        timeout_seconds=generation_timeout,
                        allow_general_knowledge=allow_general_mode,
                        kb_hit=kb_hit_flag,
                        retrieval_contexts=llm_retrieval_contexts,
                        background_contexts=background_contexts,
                    )
                    if (
                        route_mode == MODEL_ROUTE_HYBRID
                        and prefer_model_answer
                        and llm_result.mode in {"timeout_fallback", "error_fallback"}
                    ):
                        backup_result = _attempt_local_open_backup()
                        if backup_result is not None:
                            llm_result = backup_result
                    elif force_retrieval_generation and llm_result.mode in {
                        "timeout_fallback",
                        "error_fallback",
                    }:
                        backup_result = _attempt_local_timeout_backup_for_question(
                            question
                        )
                        if backup_result is not None:
                            llm_result = backup_result
                else:
                    llm_result = LLMAnswerResult(
                        answer=(
                            _fast_retrieval_answer(
                                runtime_question,
                                retrieved,
                                agent_key=payload.agent_key,
                            )
                            if retrieved
                            else _build_kb_bounded_shortfall_answer(
                                runtime_question,
                                trimmed_history,
                                route_mode=MODEL_ROUTE_RETRIEVAL_ONLY,
                                agent_key=payload.agent_key,
                            )
                        ),
                        mode=MODEL_ROUTE_RETRIEVAL_ONLY,
                    )
            except Exception:
                logger.exception(
                    "generation failed: conversation=%s agent=%s",
                    conversation_id,
                    payload.agent_key,
                )
                llm_result = LLMAnswerResult(
                    answer=(
                        _build_kb_bounded_shortfall_answer(
                            runtime_question,
                            trimmed_history,
                            route_mode=route_mode,
                            agent_key=payload.agent_key,
                        )
                        if route_mode == MODEL_ROUTE_LOCAL_ONLY
                        else "当前生成失败，已切换到简化回答。请稍后重试，或缩小问题范围后再提问。"
                    ),
                    mode="error_fallback",
                )
            stage_ms = int((perf_counter() - stage_start) * 1000)
            llm_ms += stage_ms
            _emit_progress(
                progress_callback,
                type="stage",
                stage="generation",
                status="done",
                mode=llm_result.mode if llm_result else "unknown",
                stage_ms=stage_ms,
            )
            return llm_result.answer

        if generation_first_mode:
            system_instruction = (
                build_agent_cloud_instruction((payload.agent_key or "").strip().lower())
                if route_mode == MODEL_ROUTE_CLOUD_ONLY and cloud_enabled
                else build_agent_system_instruction(
                    (payload.agent_key or "").strip().lower()
                )
            )
            _load_profile(question)
            if (
                route_mode != MODEL_ROUTE_CLOUD_ONLY
                and get_agent_bound_kb_name(payload.agent_key)
                and not retrieved
            ):
                try:
                    _retrieve(
                        question,
                        runtime_top_k=min(4, max(top_k, 2)),
                        runtime_threshold=max(
                            MIN_RELAXED_THRESHOLD,
                            min(score_threshold, 0.2),
                        ),
                        allow_relaxed_retry=True,
                        detail="正在优先检索专属知识库...",
                    )
                except Exception:
                    logger.debug("pre-answer retrieval skipped", exc_info=True)
            answer = _generate(question, [], system_instruction)
            graph_result = {
                "error": "",
                "system_instruction": system_instruction,
                "answer": answer,
                "blocked_stage": "",
                "blocked_word": "",
            }
            if route_mode != MODEL_ROUTE_CLOUD_ONLY:
                try:
                    _retrieve(
                        question,
                        runtime_top_k=min(3, max(top_k, 2)),
                        runtime_threshold=max(score_threshold, 0.2),
                        allow_relaxed_retry=False,
                        detail="正在补充可参考线索...",
                    )
                except Exception:
                    logger.debug("post-answer retrieval skipped", exc_info=True)
            workflow_stage = "done"
        else:
            workflow_budget_reserved = 2 if cloud_enabled else 6
            workflow_timeout = max(
                8,
                min(
                    workflow_timeout_cap, chat_total_timeout - workflow_budget_reserved
                ),
            )
            if cloud_enabled:
                workflow_timeout = max(
                    workflow_timeout,
                    min(workflow_timeout_cap, max(12, generation_timeout + 2)),
                )
                workflow_timeout = min(workflow_timeout, max(8, chat_total_timeout - 1))

            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                run_chat_workflow_graph,
                question=question,
                agent_key=payload.agent_key,
                sensitive_words=sensitive_words,
                detect_fn=detect_sensitive_text,
                load_profile_fn=_load_profile,
                retrieve_fn=_retrieve,
                generate_fn=_generate,
            )
            wait_start = perf_counter()
            graph_result: dict[str, Any] = {}
            timed_out = False
            if (
                route_mode == MODEL_ROUTE_LOCAL_ONLY
                and local_enabled
                and not cloud_enabled
            ):
                generation_guard_timeout = max(
                    12,
                    min(generation_timeout - 4, 26 if is_academic_analysis else 20),
                )
            else:
                generation_guard_timeout = max(
                    16,
                    min(
                        generation_timeout + 10,
                        78
                        if is_academic_analysis
                        else (
                            108
                            if route_mode == MODEL_ROUTE_CLOUD_ONLY
                            else (96 if cloud_enabled else 42)
                        ),
                    ),
                )
            try:
                while True:
                    elapsed_wait = perf_counter() - wait_start
                    remaining_wait = workflow_timeout - elapsed_wait
                    if remaining_wait <= 0:
                        timed_out = True
                        break
                    try:
                        graph_result = future.result(
                            timeout=min(WORKFLOW_POLL_INTERVAL_SECONDS, remaining_wait)
                        )
                        workflow_stage = "done"
                        break
                    except FuturesTimeoutError:
                        if (
                            workflow_stage == "generation"
                            and elapsed_wait >= generation_guard_timeout
                        ):
                            timed_out = True
                            break
                        continue
                workflow_wait_ms += int((perf_counter() - wait_start) * 1000)
                if timed_out:
                    if workflow_stage == "generation" and not future.done():
                        remaining_total = chat_total_timeout - (perf_counter() - start)
                        grace_timeout = min(3.0, max(0.0, remaining_total - 0.4))
                        if grace_timeout >= 0.8:
                            grace_start = perf_counter()
                            try:
                                graph_result = future.result(timeout=grace_timeout)
                                workflow_stage = "done"
                                timed_out = False
                            except FuturesTimeoutError:
                                pass
                            finally:
                                workflow_wait_ms += int(
                                    (perf_counter() - grace_start) * 1000
                                )
                if timed_out:
                    logger.warning(
                        "workflow timeout: conversation=%s agent=%s stage=%s elapsed_ms=%s",
                        conversation_id,
                        payload.agent_key,
                        workflow_stage,
                        workflow_wait_ms,
                    )
                    _emit_progress(
                        progress_callback,
                        type="stage",
                        stage=workflow_stage,
                        status="timeout",
                        detail="生成阶段等待过长，已切换快速兜底策略。",
                        elapsed_ms=workflow_wait_ms,
                    )
                    if generation_first_mode:
                        timeout_answer = _build_open_question_fallback(
                            question,
                            trimmed_history,
                            agent_key=payload.agent_key,
                            long_term_memory=long_term_memory,
                            profile_context=profile_context_text,
                        )
                        llm_result = LLMAnswerResult(
                            answer=timeout_answer,
                            mode="open_question_timeout_fallback",
                        )
                    elif route_mode == MODEL_ROUTE_CLOUD_ONLY and cloud_enabled:
                        backup_result = _attempt_local_timeout_backup_for_question(
                            question
                        )
                        if backup_result is not None:
                            llm_result = backup_result
                            timeout_answer = backup_result.answer
                        else:
                            timeout_answer = _build_cloud_timeout_answer(
                                question=question,
                                retrieved=retrieved,
                                trimmed_history=trimmed_history,
                                long_term_memory=long_term_memory,
                                agent_key=payload.agent_key,
                            )
                            llm_result = LLMAnswerResult(
                                answer=timeout_answer,
                                mode="cloud_timeout_fast_fallback",
                            )
                    elif route_mode == MODEL_ROUTE_HYBRID and cloud_enabled:
                        backup_result = _attempt_local_timeout_backup_for_question(
                            question
                        )
                        if backup_result is not None:
                            llm_result = backup_result
                            timeout_answer = backup_result.answer
                        else:
                            timeout_answer = _build_cloud_timeout_answer(
                                question=question,
                                retrieved=retrieved,
                                trimmed_history=trimmed_history,
                                long_term_memory=long_term_memory,
                                agent_key=payload.agent_key,
                            )
                            llm_result = LLMAnswerResult(
                                answer=timeout_answer,
                                mode="hybrid_cloud_timeout_fast_fallback",
                            )
                    else:
                        try:
                            stage_start = perf_counter()
                            fallback_top_k = min(8, max(top_k + 1, 3))
                            fallback_threshold = max(
                                MIN_RELAXED_THRESHOLD,
                                min(0.24, score_threshold - 0.08),
                            )
                            fallback_query = _resolve_retrieval_query(question)
                            retrieved = _run_retrieval_query(
                                fallback_query,
                                fallback_top_k,
                                fallback_threshold,
                            )
                            retrieval_ms += int((perf_counter() - stage_start) * 1000)
                        except Exception:
                            retrieved = []
                        timeout_answer = (
                            _fast_retrieval_answer(
                                question,
                                retrieved,
                                agent_key=payload.agent_key,
                            )
                            if retrieved
                            else _build_kb_bounded_shortfall_answer(
                                question,
                                trimmed_history,
                                route_mode=route_mode,
                                agent_key=payload.agent_key,
                            )
                        )
                    graph_result = {
                        "error": "WORKFLOW_TIMEOUT",
                        "system_instruction": "",
                        "answer": timeout_answer,
                        "blocked_stage": "",
                        "blocked_word": "",
                    }
            finally:
                future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)

        blocked_stage = str(graph_result.get("blocked_stage") or "")
        blocked_word = str(graph_result.get("blocked_word") or "")
        if blocked_stage and blocked_word:
            direction = "output" if blocked_stage == "output" else "input"
            log_sensitive_block(
                db=db,
                user_id=current_user.id if current_user else None,
                conversation_id=conversation_id,
                agent_key=payload.agent_key,
                direction=direction,
                blocked_word=blocked_word,
                content=(
                    str(graph_result.get("answer") or "")
                    if direction == "output"
                    else question
                ),
            )
            raise BusinessError(
                f"{('回答' if direction == 'output' else '输入')}命中敏感词，已拦截（{blocked_word}）。",
                status_code=400,
            )

        system_instruction = str(graph_result.get("system_instruction") or "")
        answer = str(graph_result.get("answer") or "").strip()
        if not answer:
            if generation_first_mode:
                answer = _build_open_question_fallback(
                    question,
                    trimmed_history,
                    agent_key=payload.agent_key,
                    long_term_memory=long_term_memory,
                    profile_context=profile_context_text,
                )
            elif (route_mode == MODEL_ROUTE_CLOUD_ONLY and cloud_enabled) or (
                route_mode == MODEL_ROUTE_HYBRID and cloud_enabled and not retrieved
            ):
                fallback_history = _resolve_history_context(question)
                fallback_retrieval_contexts = _format_retrieval_contexts_for_generation(
                    retrieved
                )
                fallback_background_contexts: list[str] = []
                if fallback_history:
                    fallback_background_contexts.append(
                        f"[会话背景] {_compact_text(fallback_history, 360)}"
                    )
                llm_result = answer_with_llm(
                    question=question,
                    contexts=fallback_retrieval_contexts + fallback_background_contexts,
                    llm_enabled=True,
                    system_instruction=build_agent_cloud_instruction(payload.agent_key),
                    agent_key=payload.agent_key,
                    timeout_seconds=max(4, min(8, generation_timeout)),
                    allow_general_knowledge=True,
                    retry_on_failure=False,
                    kb_hit=False,
                    retrieval_contexts=fallback_retrieval_contexts,
                    background_contexts=fallback_background_contexts,
                )
                answer = llm_result.answer
            else:
                answer = (
                    _fast_retrieval_answer(
                        question,
                        retrieved,
                        agent_key=payload.agent_key,
                    )
                    if retrieved
                    else _build_kb_bounded_shortfall_answer(
                        question,
                        trimmed_history,
                        route_mode=route_mode,
                        agent_key=payload.agent_key,
                    )
                )
        if not answer and (
            (route_mode == MODEL_ROUTE_CLOUD_ONLY and cloud_enabled)
            or (route_mode == MODEL_ROUTE_HYBRID and cloud_enabled)
        ):
            answer = (
                _build_open_question_fallback(
                    question,
                    trimmed_history,
                    agent_key=payload.agent_key,
                    long_term_memory=long_term_memory,
                    profile_context=profile_context_text,
                )
                if generation_first_mode
                else _build_cloud_timeout_answer(
                    question=question,
                    retrieved=retrieved,
                    trimmed_history=trimmed_history,
                    long_term_memory=long_term_memory,
                    agent_key=payload.agent_key,
                )
            )

        if perf_counter() - start > chat_total_timeout and not answer:
            if (route_mode == MODEL_ROUTE_CLOUD_ONLY and cloud_enabled) or (
                route_mode == MODEL_ROUTE_HYBRID and cloud_enabled
            ):
                answer = (
                    _build_open_question_fallback(
                        question,
                        trimmed_history,
                        agent_key=payload.agent_key,
                        long_term_memory=long_term_memory,
                        profile_context=profile_context_text,
                    )
                    if generation_first_mode
                    else _build_cloud_timeout_answer(
                        question=question,
                        retrieved=retrieved,
                        trimmed_history=trimmed_history,
                        long_term_memory=long_term_memory,
                        agent_key=payload.agent_key,
                    )
                )
            else:
                answer = (
                    _build_open_question_fallback(
                        question,
                        trimmed_history,
                        agent_key=payload.agent_key,
                        long_term_memory=long_term_memory,
                        profile_context=profile_context_text,
                    )
                    if generation_first_mode
                    else (
                        _fast_retrieval_answer(
                            question,
                            retrieved,
                            agent_key=payload.agent_key,
                        )
                        if retrieved
                        else _build_kb_bounded_shortfall_answer(
                            question,
                            trimmed_history,
                            route_mode=route_mode,
                            agent_key=payload.agent_key,
                        )
                    )
                )

        if (
            route_mode in {MODEL_ROUTE_LOCAL_ONLY, MODEL_ROUTE_RETRIEVAL_ONLY}
            and not retrieved
            and len(answer) < 160
            and not generation_first_mode
        ):
            answer = _build_kb_bounded_shortfall_answer(
                question,
                trimmed_history,
                route_mode=route_mode,
                agent_key=payload.agent_key,
            )

        show_sources = route_mode in {
            MODEL_ROUTE_LOCAL_ONLY,
            MODEL_ROUTE_RETRIEVAL_ONLY,
        }
        sources = (
            _build_unique_sources(retrieved) if (show_sources and retrieved) else []
        )

    if route_mode in {MODEL_ROUTE_LOCAL_ONLY, MODEL_ROUTE_RETRIEVAL_ONLY}:
        if route_mode == MODEL_ROUTE_RETRIEVAL_ONLY:
            answer = _format_final_user_answer(
                answer=answer,
                question=question,
                question_mode=question_mode,
                retrieved=retrieved,
                agent_key=payload.agent_key,
            )
        else:
            answer = _polish_final_answer_text(answer)

    blocked_output_word = detect_sensitive_text(answer, sensitive_words)
    if blocked_output_word:
        log_sensitive_block(
            db=db,
            user_id=current_user.id if current_user else None,
            conversation_id=conversation_id,
            agent_key=payload.agent_key,
            direction="output",
            blocked_word=blocked_output_word,
            content=answer,
        )
        raise BusinessError(
            f"回答命中敏感词，已拦截（{blocked_output_word}）。",
            status_code=400,
        )

    answer = mask_sensitive_text(answer, sensitive_words)
    if len(answer) > settings.max_answer_chars:
        answer = answer[: settings.max_answer_chars] + "..."

    elapsed_ms = int((perf_counter() - start) * 1000)

    thinking = _build_thinking_payload(
        trimmed_history=trimmed_history,
        kb_ids=kb_ids,
        document_ids=document_ids,
        route_mode=route_mode,
        question_mode=question_mode,
        retrieved_count=len(retrieved),
        top_k=top_k,
        score_threshold=score_threshold,
        rule_answer=bool(rule_answer_text),
        llm_result=llm_result,
        profile_ms=profile_ms,
        retrieval_ms=retrieval_ms,
        llm_ms=llm_ms,
        workflow_wait_ms=workflow_wait_ms,
        workflow_stage=workflow_stage,
        total_ms=elapsed_ms,
    )
    thinking_content = mask_sensitive_text(thinking.content, sensitive_words).strip()
    thinking = ChatThinking(
        title=thinking.title,
        content=thinking_content
        or "- 已完成当前问题处理。\n- 这里展示的是系统处理摘要，不包含模型私有思维链。",
        kind=thinking.kind,
        is_real=thinking.is_real,
        collapsed=thinking.collapsed,
    )

    db.add(Message(conversation_id=conversation_id, role="user", content=question))
    db.add(Message(conversation_id=conversation_id, role="assistant", content=answer))
    db.add(
        ChatLog(
            conversation_id=conversation_id,
            user_id=current_user.id if current_user else None,
            question=question,
            answer=answer,
            kb_ids=",".join(kb_ids),
            retrieval_top_k=top_k,
            score_threshold=score_threshold,
            elapsed_ms=elapsed_ms,
        )
    )
    db.add(
        ChatPerfLog(
            conversation_id=conversation_id,
            user_id=current_user.id if current_user else None,
            agent_key=(payload.agent_key or "").strip().lower() or None,
            question=question,
            retrieved_count=len(retrieved),
            llm_mode=llm_result.mode
            if llm_result
            else ("rule" if rule_answer_text else "none"),
            profile_ms=profile_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            workflow_wait_ms=workflow_wait_ms,
            total_ms=elapsed_ms,
            workflow_stage=workflow_stage,
        )
    )
    db.commit()

    refreshed_history = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        ).all()
    )
    existing_memory = _extract_long_term_memory(refreshed_history)
    memory_messages = [
        item for item in refreshed_history if _is_long_term_memory_message(item)
    ]
    latest_memory_message = memory_messages[-1] if memory_messages else None
    non_memory_history = [
        item for item in refreshed_history if not _is_long_term_memory_message(item)
    ]
    trimmed_after = _truncate_history(
        history=non_memory_history, max_rounds=max_rounds, max_tokens=max_tokens
    )
    keep_ids = {item.id for item in trimmed_after}
    dropped_messages = [item for item in non_memory_history if item.id not in keep_ids]
    updated_memory = _build_long_term_memory_update(existing_memory, dropped_messages)

    if updated_memory:
        memory_payload = _format_long_term_memory(updated_memory)
        if latest_memory_message is None:
            latest_memory_message = Message(
                conversation_id=conversation_id,
                role=LONG_TERM_MEMORY_ROLE,
                content=memory_payload,
            )
            db.add(latest_memory_message)
            db.flush()
        elif latest_memory_message.content != memory_payload:
            latest_memory_message.content = memory_payload
        keep_ids.add(latest_memory_message.id)
    elif latest_memory_message is not None:
        keep_ids.add(latest_memory_message.id)

    if keep_ids:
        db.query(Message).filter(
            Message.conversation_id == conversation_id,
            ~Message.id.in_(keep_ids),
        ).delete(synchronize_session=False)
        db.commit()

    _emit_progress(
        progress_callback,
        type="stage",
        stage="finalize",
        status="done",
        detail="回答已生成完成。",
        total_ms=elapsed_ms,
    )

    return ChatCompletionResult(
        conversation_id=conversation_id,
        answer=answer,
        sources=sources,
        thinking=thinking,
    )


def clear_conversation_context(db: Session, conversation_id: str) -> int:
    deleted = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .delete(synchronize_session=False)
    )
    db.commit()
    return int(deleted or 0)


def rollback_conversation_context(
    db: Session, conversation_id: str, keep_rounds: int
) -> int:
    history = list(
        db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        ).all()
    )
    if not history:
        return 0

    keep_messages = max(0, keep_rounds * 2)
    keep_ids = {item.id for item in history[:keep_messages]}
    query = db.query(Message).filter(Message.conversation_id == conversation_id)
    if keep_ids:
        query = query.filter(~Message.id.in_(keep_ids))
    deleted = query.delete(synchronize_session=False)
    db.commit()
    return int(deleted or 0)
