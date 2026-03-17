from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SensitiveBlockLog(Base):
    __tablename__ = "sensitive_block_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    conversation_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    agent_key: Mapped[str] = mapped_column(String(64), default="", index=True)
    direction: Mapped[str] = mapped_column(String(16), default="input")
    blocked_word: Mapped[str] = mapped_column(String(128), default="")
    content_preview: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
