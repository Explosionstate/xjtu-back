from __future__ import annotations

import uuid
from time import perf_counter

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chat import ChatLog, Conversation, Message
from app.models.knowledge_base import KnowledgeBase
from app.models.rbac import User
from app.schemas.chat import ChatCompletionRequest, SourceItem
from app.services.llm_service import answer_with_llm
from app.services.retrieval_config_service import get_effective_retrieval_config
from app.services.retrieval_service import hybrid_retrieve
from app.services.sensitive_service import get_sensitive_words, mask_sensitive_text
from app.services.system_config_service import (
    DEFAULT_CONTEXT_MAX_ROUNDS,
    DEFAULT_CONTEXT_MAX_TOKENS,
    get_int_config,
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
    if "帮助" in question or "help" in lowered:
        return "你可以提问知识库内容，或指定 kb_ids 进行范围检索。"
    if "版本" in question or "version" in lowered:
        return "xjtu-back 对话服务版本：0.1.0"
    return None


def _estimate_tokens(text: str) -> int:
    # Lightweight token estimation for mixed Chinese/English text.
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


def chat_completion(
    db: Session,
    payload: ChatCompletionRequest,
    current_user: User | None,
) -> tuple[str, str, list[SourceItem]]:
    question = _get_latest_user_question([m.model_dump() for m in payload.messages])
    sensitive_words = get_sensitive_words(db)
    question = mask_sensitive_text(question, sensitive_words)
    conversation_id = payload.conversation_id or str(uuid.uuid4())
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
    rule_answer = _format_rule_answer(question)
    if rule_answer:
        answer = rule_answer
        sources: list[SourceItem] = []
    else:
        retrieved = hybrid_retrieve(
            db=db,
            query=retrieval_query,
            kb_ids=kb_ids,
            document_ids=payload.document_ids,
            top_k=top_k,
            score_threshold=score_threshold,
            fusion_mode=fusion_mode,
            alpha=alpha,
        )
        if not retrieved:
            answer = "未在知识库中找到相关答案，请换个问题试试。"
            sources = []
        else:
            contexts = [item["content"] for item in retrieved]
            answer = answer_with_llm(
                question=question,
                contexts=contexts,
                llm_enabled=payload.llm_enabled,
            )
            if not answer:
                answer = "\n\n".join(contexts)
            sources = [
                SourceItem(
                    source_location=item["source_location"],
                    content=item["content"],
                    score=round(item["score"], 4),
                )
                for item in retrieved
            ]
    answer = mask_sensitive_text(answer, sensitive_words)

    if len(answer) > settings.max_answer_chars:
        answer = answer[: settings.max_answer_chars] + "..."

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

    return conversation_id, answer, sources


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
