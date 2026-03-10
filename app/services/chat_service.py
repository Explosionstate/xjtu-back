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
from app.services.retrieval_service import hybrid_retrieve


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


def chat_completion(
    db: Session,
    payload: ChatCompletionRequest,
    current_user: User | None,
) -> tuple[str, str, list[SourceItem]]:
    question = _get_latest_user_question([m.model_dump() for m in payload.messages])
    conversation_id = payload.conversation_id or str(uuid.uuid4())
    kb_ids = payload.kb_ids or [
        item.id
        for item in db.scalars(
            select(KnowledgeBase).where(KnowledgeBase.status == "active")
        ).all()
    ]
    top_k = payload.top_k or settings.retrieval_top_k
    score_threshold = (
        payload.score_threshold
        if payload.score_threshold is not None
        else settings.retrieval_score_threshold
    )
    fusion_mode = payload.fusion_mode or settings.retrieval_fusion_mode
    alpha = payload.alpha if payload.alpha is not None else settings.retrieval_alpha

    _ensure_conversation(
        db=db,
        conversation_id=conversation_id,
        user_id=current_user.id if current_user else None,
    )

    start = perf_counter()
    rule_answer = _format_rule_answer(question)
    if rule_answer:
        answer = rule_answer
        sources: list[SourceItem] = []
    else:
        retrieved = hybrid_retrieve(
            db=db,
            query=question,
            kb_ids=kb_ids,
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
            answer = answer_with_llm(question=question, contexts=contexts)
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

    return conversation_id, answer, sources
