from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class TransformerEvalSnapshot(Base):
    __tablename__ = "transformer_eval_snapshot"

    snapshot_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    batch_id: Mapped[str] = mapped_column(String(36), index=True)
    run_index: Mapped[int] = mapped_column(Integer, default=1)
    provider: Mapped[str] = mapped_column(String(64), default="local_transformer")
    model: Mapped[str] = mapped_column(String(255), default="")
    total_topics: Mapped[int] = mapped_column(Integer, default=0)
    pass_count: Mapped[int] = mapped_column(Integer, default=0)
    average_score: Mapped[float] = mapped_column(default=0.0)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
