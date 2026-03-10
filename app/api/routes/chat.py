from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.chat import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
)
from app.services.chat_service import chat_completion

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
