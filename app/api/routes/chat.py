from __future__ import annotations

import uuid
import asyncio

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.security import decode_access_token
from app.db.session import get_db
from app.db.session import SessionLocal
from app.models.rbac import User
from app.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
)
from app.services.chat_service import chat_completion
from app.services.auth_service import get_active_user

router = APIRouter(tags=["chat"])


@router.post("/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(
    payload: ChatCompletionRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> ChatCompletionResponse:
    conversation_id, answer, sources = chat_completion(
        db=db, payload=payload, current_user=current_user
    )


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
            conversation_id, answer, sources = chat_completion(
                db=db, payload=payload, current_user=user
            )
            await websocket.send_json(
                {
                    "type": "meta",
                    "conversation_id": conversation_id,
                    "model": payload.model or "xjtu-hybrid-rag",
                }
            )
            for ch in answer:
                await websocket.send_json({"type": "delta", "content": ch})
                if settings.chat_stream_delay_ms > 0:
                    await asyncio.sleep(settings.chat_stream_delay_ms / 1000)
            await websocket.send_json(
                {
                    "type": "done",
                    "sources": [item.model_dump() for item in sources],
                }
            )
    except WebSocketDisconnect:
        return
    finally:
        db.close()
    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:20]}",
        model=payload.model or "xjtu-hybrid-rag",
        conversation_id=conversation_id,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionResponseMessage(content=answer),
            )
        ],
        sources=sources,
    )
