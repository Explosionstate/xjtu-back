from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ChatLogItem(BaseModel):
    id: str
    conversation_id: str
    user_id: int | None = None
    question: str
    answer: str
    kb_ids: str
    retrieval_top_k: int
    score_threshold: float
    elapsed_ms: int
    created_at: datetime

    class Config:
        from_attributes = True


class ChatLogListResponse(BaseModel):
    total: int
    items: list[ChatLogItem]
