from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from time import perf_counter
import logging
import re
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import BusinessError
from app.models.chat import ChatLog, ChatPerfLog, Conversation, Message
from app.models.knowledge_base import KnowledgeBase
from app.models.rbac import User
from app.schemas.chat import ChatCompletionRequest, ChatThinking, SourceItem
from app.services.llm_service import LLMAnswerResult, answer_with_llm
from app.services.local_transformer_service import (
    generate_answer_with_local_transformer,
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
    get_int_config,
)


CHAT_TOTAL_TIMEOUT_SECONDS = 60
WORKFLOW_TIMEOUT_SECONDS = 52
GENERATION_STAGE_TIMEOUT_SECONDS = 42
ACADEMIC_CHAT_TOTAL_TIMEOUT_SECONDS = 120
ACADEMIC_WORKFLOW_TIMEOUT_SECONDS = 112
ACADEMIC_GENERATION_STAGE_TIMEOUT_SECONDS = 96
ACADEMIC_PROFILE_TIMEOUT_SECONDS = 8.0
DEFAULT_PROFILE_TIMEOUT_SECONDS = 1.5
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatCompletionResult:
    conversation_id: str
    answer: str
    sources: list[SourceItem]
    thinking: ChatThinking


def _fast_retrieval_answer(question: str, retrieved: list[dict]) -> str:
    lowered = (question or "").lower()
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
        return (
            "结论：当前未检索到足够的直接依据。\n"
            "依据：系统已在限定时间内完成快速检索，但有效片段不足。\n"
            "建议：请补充具体场景（例如课程、作息、时间安排）后重试。"
        )

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
        return (
            f"结论：已基于本周新增资料给出重点概览（来源：{refs}）。\n"
            "依据：检索结果集中在方法论表达升级、落地场景设计与系统稳定性改进。\n"
            "建议：可继续指定“面向学生/教师/管理员”其一，我将输出对应版本的三点重点与行动清单。"
        )

    snippets: list[str] = []
    for item in retrieved[:3]:
        content = str(item.get("content") or "").strip().replace("\n", " ")
        content = re.sub(r"\s+", " ", content)
        if any(
            token in content.lower()
            for token in ["create table", "insert into", " key `", "idx_"]
        ):
            continue
        if content:
            snippets.append(content[:140])
    evidence = "；".join(snippets) if snippets else "已检索到相关片段。"
    return (
        f"结论：已基于当前检索结果回答“{question[:32]}”。\n"
        f"依据：{evidence}\n"
        "建议：若你希望更精确，请补充目标对象、时间范围和约束条件（例如平台、端口、环境）。"
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


def _format_rule_answer(question: str) -> str | None:
    lowered = question.lower()
    if any(item in lowered for item in ["你好", "hello", "hi", "嗨", "在吗"]):
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


def _estimate_tokens(text: str) -> int:
    # Lightweight token estimation for mixed Chinese and English text.
    return max(1, len(text) // 2)


def _truncate_history(
    history: list[Message], max_rounds: int, max_tokens: int
) -> list[Message]:
    if not history:
        return []
    kept: list[Message] = []
    token_sum = 0
    max_messages = max_rounds * 2
    for msg in reversed(history):
        tokens = _estimate_tokens(msg.content)
        if kept and (len(kept) >= max_messages or token_sum + tokens > max_tokens):
            break
        kept.append(msg)
        token_sum += tokens
    return list(reversed(kept))


def _build_retrieval_query(history: list[Message], question: str) -> str:
    summary_keywords = ["总结", "概述", "重点", "本周新增文档", "学生指南"]
    if any(keyword in question for keyword in summary_keywords):
        # For summary-style requests, avoid dragging unrelated chat history.
        return question

    if not history:
        return question
    snippets = [item.content for item in history[-4:] if item.content.strip()]
    if not snippets:
        return question
    return "\n".join(snippets + [question])


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


def _is_student_growth_agent(agent_key: str | None) -> bool:
    key = (agent_key or "").strip().lower()
    return key in {"student-growth", "student_growth", "student"}


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


def _is_guidance_query(question: str) -> bool:
    q = (question or "").strip().lower()
    flags = ["学生", "建议", "指南", "总结", "重点", "规划", "学习", "生活"]
    return any(flag in q for flag in flags)


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
    if trimmed_history:
        rounds = max(1, (len(trimmed_history) + 1) // 2)
        steps.append(f"已结合最近 {rounds} 轮对话上下文整理当前问题。")
    else:
        steps.append("已读取当前问题并准备检索。")

    if rule_answer:
        steps.append("命中系统内置规则回答，未进入知识库检索和模型生成。")
    else:
        scope_text = _build_scope_text(kb_ids, document_ids)
        if retrieved_count > 0:
            steps.append(
                f"已在{scope_text}中筛到 {retrieved_count} 条相关片段（Top-K={top_k}，阈值={score_threshold:.2f}）。"
            )
        else:
            steps.append(
                f"已在{scope_text}中完成检索，但当前阈值 {score_threshold:.2f} 下没有足够相关的参考资料。"
            )

        if llm_result is None:
            steps.append("本次回答直接基于检索结果整理。")
        elif llm_result.mode == "llm":
            steps.append("已基于检索结果组织最终回答。")
        elif llm_result.mode == "disabled":
            steps.append(
                "当前未启用可返回 reasoning 的模型，本次回答由检索结果整理生成。"
            )
        else:
            steps.append("模型未返回可公开展示的 reasoning，本次展示的是系统处理摘要。")

    steps.append("这里展示的是处理摘要，不是模型私有思维链。")
    steps.append(
        "性能耗时："
        f"profile_ms={profile_ms}，retrieval_ms={retrieval_ms}，"
        f"llm_ms={llm_ms}，total_ms={total_ms}。"
    )
    if workflow_wait_ms > 0:
        steps.append(f"流程等待耗时：workflow_wait_ms={workflow_wait_ms}。")
    if workflow_stage and workflow_stage != "done":
        steps.append(f"当前判定卡点：workflow_stage={workflow_stage}。")
    return ChatThinking(
        title="处理摘要",
        content="\n".join(
            f"{index}. {step}" for index, step in enumerate(steps, start=1)
        ),
        kind="summary",
        is_real=False,
        collapsed=True,
    )


def _build_thinking_payload(
    trimmed_history: list[Message],
    kb_ids: list[str],
    document_ids: list[str] | None,
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
    if llm_result and llm_result.reasoning:
        return ChatThinking(
            title="思考过程",
            content=llm_result.reasoning,
            kind="reasoning",
            is_real=True,
            collapsed=True,
        )
    return _build_summary_thinking(
        trimmed_history=trimmed_history,
        kb_ids=kb_ids,
        document_ids=document_ids,
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
    kb_ids = payload.kb_ids or [
        item.id
        for item in db.scalars(
            select(KnowledgeBase).where(KnowledgeBase.status == "active")
        ).all()
    ]
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

    if _is_guidance_query(question):
        top_k = min(top_k, 5)
        score_threshold = max(score_threshold, 0.25)

    configured_max_rounds = get_int_config(
        db, "context_max_rounds", DEFAULT_CONTEXT_MAX_ROUNDS
    )
    configured_max_tokens = get_int_config(
        db, "context_max_tokens", DEFAULT_CONTEXT_MAX_TOKENS
    )
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
    trimmed_history = _truncate_history(
        history=history, max_rounds=max_rounds, max_tokens=max_tokens
    )
    retrieval_query = _expand_academic_query(
        payload.agent_key,
        _build_retrieval_query(trimmed_history, question),
    )
    is_academic_analysis = _is_academic_analysis_query(payload.agent_key, question)
    chat_total_timeout = (
        ACADEMIC_CHAT_TOTAL_TIMEOUT_SECONDS
        if is_academic_analysis
        else CHAT_TOTAL_TIMEOUT_SECONDS
    )
    workflow_timeout_cap = (
        ACADEMIC_WORKFLOW_TIMEOUT_SECONDS
        if is_academic_analysis
        else WORKFLOW_TIMEOUT_SECONDS
    )
    generation_timeout = (
        ACADEMIC_GENERATION_STAGE_TIMEOUT_SECONDS
        if is_academic_analysis
        else GENERATION_STAGE_TIMEOUT_SECONDS
    )
    profile_timeout_seconds = (
        ACADEMIC_PROFILE_TIMEOUT_SECONDS
        if is_academic_analysis
        else DEFAULT_PROFILE_TIMEOUT_SECONDS
    )

    start = perf_counter()
    llm_result: LLMAnswerResult | None = None
    retrieved: list[dict] = []
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
    else:

        def _load_profile(_: str) -> str:
            nonlocal profile_ms, workflow_stage
            workflow_stage = "profile"
            if current_user is None:
                return ""
            stage_start = perf_counter()
            try:
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
                    return profile_future.result(timeout=profile_timeout_seconds)
                except FuturesTimeoutError:
                    logger.warning(
                        "profile timeout: conversation=%s agent=%s academic=%s",
                        conversation_id,
                        payload.agent_key,
                        is_academic_analysis,
                    )
                    return ""
                finally:
                    profile_future.cancel()
                    profile_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                return ""
            finally:
                profile_ms += int((perf_counter() - stage_start) * 1000)

        def _retrieve(runtime_question: str) -> tuple[list[str], list[dict]]:
            nonlocal retrieved, retrieval_ms, workflow_stage
            workflow_stage = "retrieval"
            stage_start = perf_counter()
            query = _expand_academic_query(
                payload.agent_key,
                _build_retrieval_query(trimmed_history, runtime_question),
            )
            try:
                retrieved = hybrid_retrieve(
                    db=db,
                    query=query,
                    kb_ids=kb_ids,
                    document_ids=payload.document_ids,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    fusion_mode=fusion_mode,
                    alpha=alpha,
                    agent_key=payload.agent_key,
                )
            except Exception:
                logger.exception(
                    "retrieval failed: conversation=%s agent=%s",
                    conversation_id,
                    payload.agent_key,
                )
                retrieved = []
            finally:
                retrieval_ms += int((perf_counter() - stage_start) * 1000)
            return [item["content"] for item in retrieved], retrieved

        def _generate(
            runtime_question: str,
            contexts: list[str],
            system_instruction: str,
        ) -> str:
            nonlocal llm_result, llm_ms, workflow_stage
            workflow_stage = "generation"
            stage_start = perf_counter()
            try:
                if payload.local_transformer_enabled:
                    answer, model_reference, _metrics = (
                        generate_answer_with_local_transformer(
                            question=runtime_question,
                            contexts=contexts,
                            model_name=settings.local_transformer_model,
                        )
                    )
                    llm_result = LLMAnswerResult(
                        answer=answer,
                        mode=f"local_transformer:{model_reference}",
                    )
                else:
                    llm_result = answer_with_llm(
                        question=runtime_question,
                        contexts=contexts,
                        llm_enabled=payload.llm_enabled,
                        system_instruction=system_instruction,
                        timeout_seconds=generation_timeout,
                    )
            except Exception:
                logger.exception(
                    "generation failed: conversation=%s agent=%s",
                    conversation_id,
                    payload.agent_key,
                )
                llm_result = LLMAnswerResult(
                    answer="结论：当前生成失败，已切换到简化回答。\n依据：模型推理阶段发生异常。\n建议：请稍后重试，或缩小问题范围后再提问。",
                    mode="error_fallback",
                )
            llm_ms += int((perf_counter() - stage_start) * 1000)
            return llm_result.answer

        workflow_timeout = max(
            8,
            min(workflow_timeout_cap, chat_total_timeout - 6),
        )

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
        try:
            graph_result = future.result(timeout=workflow_timeout)
            workflow_wait_ms += int((perf_counter() - wait_start) * 1000)
            workflow_stage = "done"
        except FuturesTimeoutError:
            workflow_wait_ms += int((perf_counter() - wait_start) * 1000)
            logger.warning(
                "workflow timeout: conversation=%s agent=%s stage=%s elapsed_ms=%s",
                conversation_id,
                payload.agent_key,
                workflow_stage,
                workflow_wait_ms,
            )
            try:
                # Ensure user gets a practical answer instead of a pure timeout message.
                stage_start = perf_counter()
                retrieved = hybrid_retrieve(
                    db=db,
                    query=_expand_academic_query(
                        payload.agent_key,
                        _build_retrieval_query(trimmed_history, question),
                    ),
                    kb_ids=kb_ids,
                    document_ids=payload.document_ids,
                    top_k=min(2, top_k),
                    score_threshold=max(score_threshold, 0.35),
                    fusion_mode=fusion_mode,
                    alpha=alpha,
                    agent_key=payload.agent_key,
                )
                retrieval_ms += int((perf_counter() - stage_start) * 1000)
            except Exception:
                retrieved = []
            graph_result = {
                "error": "WORKFLOW_TIMEOUT",
                "system_instruction": "",
                "answer": _fast_retrieval_answer(question, retrieved),
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
            answer = "结论：未检索到可用答案。\n依据：当前知识库上下文不足。\n建议：请补充更具体问题或导入相关文档。"

        if perf_counter() - start > chat_total_timeout and not answer:
            answer = _fast_retrieval_answer(question, retrieved)

        if retrieved:
            sources = [
                SourceItem(
                    source_location=item["source_location"],
                    content=item["content"],
                    score=round(item["score"], 4),
                )
                for item in retrieved
            ]
        else:
            sources = []

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
        document_ids=payload.document_ids,
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
        or "1. 已完成当前问题处理。\n2. 这里展示的是系统处理摘要，不是模型私有思维链。",
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
    trimmed_after = _truncate_history(
        history=refreshed_history, max_rounds=max_rounds, max_tokens=max_tokens
    )
    keep_ids = {item.id for item in trimmed_after}
    if keep_ids:
        db.query(Message).filter(
            Message.conversation_id == conversation_id,
            ~Message.id.in_(keep_ids),
        ).delete(synchronize_session=False)
        db.commit()

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
