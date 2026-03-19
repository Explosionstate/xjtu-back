from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.security import decode_access_token
from app.db.session import SessionLocal, get_db
from app.models.chat import ChatPerfLog
from app.models.knowledge_base import KnowledgeBase
from app.models.rbac import User
from app.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
)
from app.schemas.retrieval_debug import (
    RetrievalDebugRequest,
    RetrievalDebugResponse,
    RetrievalDebugScoreItem,
)
from app.services.auth_service import get_active_user
from app.services.chat_service import (
    chat_completion,
    clear_conversation_context,
    rollback_conversation_context,
)
from app.services.retrieval_config_service import get_effective_retrieval_config
from app.services.retrieval_service import hybrid_retrieve_with_debug

router = APIRouter(tags=["chat"])


def _iter_thinking_chunks(text: str) -> list[str]:
    chunks = [item for item in text.splitlines(keepends=True) if item.strip()]
    return chunks or [text]


def _iter_answer_chunks(text: str, chunk_size: int = 24) -> list[str]:
    if not text:
        return []
    safe_size = max(1, chunk_size)
    return [text[index : index + safe_size] for index in range(0, len(text), safe_size)]


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(
    payload: ChatCompletionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> ChatCompletionResponse:
    result = chat_completion(db=db, payload=payload, current_user=current_user)
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:20]}",
        model=payload.model or "xjtu-hybrid-rag",
        conversation_id=result.conversation_id,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionResponseMessage(content=result.answer),
            )
        ],
        sources=result.sources,
        thinking=result.thinking,
    )


@router.delete("/chat/conversations/{conversation_id}/context")
def clear_context(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> dict[str, int]:
    _ = current_user
    deleted = clear_conversation_context(db=db, conversation_id=conversation_id)
    return {"deleted_messages": deleted}


@router.post("/chat/conversations/{conversation_id}/rollback")
def rollback_context(
    conversation_id: str,
    keep_rounds: int = Query(ge=0, default=0),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> dict[str, int]:
    _ = current_user
    deleted = rollback_conversation_context(
        db=db,
        conversation_id=conversation_id,
        keep_rounds=keep_rounds,
    )
    return {"deleted_messages": deleted, "keep_rounds": keep_rounds}


@router.post("/chat/retrieval-debug", response_model=RetrievalDebugResponse)
def retrieval_debug(
    payload: RetrievalDebugRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> RetrievalDebugResponse:
    _ = current_user
    kb_ids = payload.kb_ids or [
        item.id
        for item in db.scalars(
            select(KnowledgeBase).where(KnowledgeBase.status == "active")
        ).all()
    ]
    cfg = get_effective_retrieval_config(
        db=db,
        conversation_id=None,
        payload_top_k=payload.top_k,
        payload_score_threshold=payload.score_threshold,
        payload_fusion_mode=payload.fusion_mode,
        payload_alpha=payload.alpha,
    )
    results, debug_rows = hybrid_retrieve_with_debug(
        db=db,
        query=payload.query,
        kb_ids=kb_ids,
        document_ids=payload.document_ids,
        top_k=int(cfg["retrieval_top_k"]),
        score_threshold=float(cfg["score_threshold"]),
        fusion_mode=str(cfg["fusion_mode"]),
        alpha=float(cfg["alpha"]),
    )
    top_ids = {item["chunk_id"] for item in results}
    top_rows = [item for item in debug_rows if item["chunk_id"] in top_ids]
    return RetrievalDebugResponse(
        top_k_results=[
            RetrievalDebugScoreItem.model_validate(item) for item in top_rows
        ],
        all_candidates=[
            RetrievalDebugScoreItem.model_validate(item) for item in debug_rows
        ],
    )


@router.get("/chat/perf/slow-top")
def slow_requests_top(
    limit: int = Query(default=20, ge=1, le=200),
    agent_key: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> list[dict[str, int | str | None]]:
    _ = current_user
    stmt = select(ChatPerfLog)
    if agent_key:
        stmt = stmt.where(ChatPerfLog.agent_key == agent_key.strip().lower())
    rows = list(
        db.scalars(stmt.order_by(ChatPerfLog.total_ms.desc()).limit(limit)).all()
    )
    return [
        {
            "conversation_id": item.conversation_id,
            "agent_key": item.agent_key,
            "question": item.question[:80],
            "workflow_stage": item.workflow_stage,
            "llm_mode": item.llm_mode,
            "retrieved_count": int(item.retrieved_count),
            "profile_ms": int(item.profile_ms),
            "retrieval_ms": int(item.retrieval_ms),
            "llm_ms": int(item.llm_ms),
            "workflow_wait_ms": int(item.workflow_wait_ms),
            "total_ms": int(item.total_ms),
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item in rows
    ]


@router.websocket("/ws/chat/completions")
async def ws_chat_completions(
    websocket: WebSocket,
    token: str = Query(default=""),
) -> None:
    await websocket.accept()
    db = SessionLocal()
    try:
        user: User | None = None
        subject = decode_access_token(token)
        if subject and subject.isdigit():
            user = get_active_user(db, int(subject))
        else:
            await websocket.send_json({"type": "error", "detail": "token无效"})
            await websocket.close(code=1008)
            return

        while True:
            incoming = await websocket.receive_json()
            payload = ChatCompletionRequest.model_validate(incoming)

            await websocket.send_json(
                {
                    "type": "thinking",
                    "status": "start",
                    "title": "思考中",
                    "content": "正在分析问题并检索相关资料...",
                    "kind": "summary",
                    "is_real": False,
                    "done": False,
                }
            )

            result = chat_completion(db=db, payload=payload, current_user=user)

            await websocket.send_json(
                {
                    "type": "meta",
                    "conversation_id": result.conversation_id,
                    "model": payload.model or "xjtu-hybrid-rag",
                }
            )

            for chunk in _iter_thinking_chunks(result.thinking.content):
                await websocket.send_json(
                    {
                        "type": "thinking",
                        "status": "delta",
                        "title": result.thinking.title,
                        "content": chunk,
                        "kind": result.thinking.kind,
                        "is_real": result.thinking.is_real,
                        "done": False,
                    }
                )

            await websocket.send_json(
                {
                    "type": "thinking",
                    "status": "done",
                    "title": result.thinking.title,
                    "content": "",
                    "kind": result.thinking.kind,
                    "is_real": result.thinking.is_real,
                    "done": True,
                }
            )

            for chunk in _iter_answer_chunks(result.answer):
                await websocket.send_json({"type": "delta", "content": chunk})
                if settings.chat_stream_delay_ms > 0:
                    await asyncio.sleep(settings.chat_stream_delay_ms / 1000)

            await websocket.send_json(
                {
                    "type": "done",
                    "sources": [item.model_dump() for item in result.sources],
                }
            )
    except WebSocketDisconnect:
        return
    finally:
        db.close()
