from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from time import perf_counter
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import BusinessError
from app.models.chat import ChatLog, Conversation, Message
from app.models.knowledge_base import KnowledgeBase
from app.models.rbac import User
from app.schemas.chat import ChatCompletionRequest, ChatThinking, SourceItem
from app.services.llm_service import LLMAnswerResult, answer_with_llm
from app.services.external_profile_service import load_user_profile_context
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


CHAT_TOTAL_TIMEOUT_SECONDS = 30
RETRIEVAL_TIMEOUT_SECONDS = 8
PROFILE_TIMEOUT_SECONDS = 2


@dataclass(frozen=True)
class ChatCompletionResult:
    conversation_id: str
    answer: str
    sources: list[SourceItem]
    thinking: ChatThinking


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
    if not history:
        return question
    snippets = [item.content for item in history[-4:] if item.content.strip()]
    if not snippets:
        return question
    return "\n".join(snippets + [question])


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
    retrieval_query = _build_retrieval_query(trimmed_history, question)

    start = perf_counter()
    llm_result: LLMAnswerResult | None = None
    retrieved: list[dict] = []
    rule_answer_text = _format_rule_answer(question)
    if rule_answer_text:
        answer = rule_answer_text
        sources: list[SourceItem] = []
        system_instruction = ""
    else:

        def _load_profile(_: str) -> str:
            if current_user is None:
                return ""
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        load_user_profile_context, current_user.login_name
                    )
                    return future.result(timeout=PROFILE_TIMEOUT_SECONDS)
            except Exception:
                return ""

        def _retrieve(runtime_question: str) -> tuple[list[str], list[dict]]:
            nonlocal retrieved
            query = _build_retrieval_query(trimmed_history, runtime_question)
            try:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        hybrid_retrieve,
                        db,
                        query,
                        kb_ids,
                        payload.document_ids,
                        top_k,
                        score_threshold,
                        fusion_mode,
                        alpha,
                    )
                    retrieved = future.result(timeout=RETRIEVAL_TIMEOUT_SECONDS)
            except FuturesTimeoutError:
                retrieved = []
            except Exception:
                retrieved = []
            return [item["content"] for item in retrieved], retrieved

        def _generate(
            runtime_question: str,
            contexts: list[str],
            system_instruction: str,
        ) -> str:
            nonlocal llm_result
            llm_result = answer_with_llm(
                question=runtime_question,
                contexts=contexts,
                llm_enabled=payload.llm_enabled,
                system_instruction=system_instruction,
            )
            return llm_result.answer

        graph_result = run_chat_workflow_graph(
            question=question,
            agent_key=payload.agent_key,
            sensitive_words=sensitive_words,
            detect_fn=detect_sensitive_text,
            load_profile_fn=_load_profile,
            retrieve_fn=_retrieve,
            generate_fn=_generate,
        )

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

        if perf_counter() - start > CHAT_TOTAL_TIMEOUT_SECONDS:
            answer = (
                "结论：当前请求处理超出实时窗口。\n"
                "依据：检索或生成耗时超过 30 秒。\n"
                "建议：请缩小检索范围（文档/知识库）或降低召回数量后重试。"
            )
            sources = []
            retrieved = []

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

    thinking = _build_thinking_payload(
        trimmed_history=trimmed_history,
        kb_ids=kb_ids,
        document_ids=payload.document_ids,
        retrieved_count=len(retrieved),
        top_k=top_k,
        score_threshold=score_threshold,
        rule_answer=bool(rule_answer_text),
        llm_result=llm_result,
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

    elapsed_ms = int((perf_counter() - start) * 1000)

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
