from __future__ import annotations

from datetime import datetime
from io import StringIO
import csv

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.chat import ChatLog


def list_chat_logs(
    db: Session,
    user_id: int | None,
    conversation_id: str | None,
    kb_id: str | None,
    keyword: str | None,
    created_from: datetime | None,
    created_to: datetime | None,
    offset: int,
    limit: int,
) -> tuple[int, list[ChatLog]]:
    stmt = select(ChatLog)
    count_stmt = select(func.count()).select_from(ChatLog)

    if user_id is not None:
        stmt = stmt.where(ChatLog.user_id == user_id)
        count_stmt = count_stmt.where(ChatLog.user_id == user_id)
    if conversation_id:
        stmt = stmt.where(ChatLog.conversation_id == conversation_id)
        count_stmt = count_stmt.where(ChatLog.conversation_id == conversation_id)
    if kb_id:
        stmt = stmt.where(ChatLog.kb_ids.contains(kb_id))
        count_stmt = count_stmt.where(ChatLog.kb_ids.contains(kb_id))
    if keyword:
        stmt = stmt.where(
            ChatLog.question.contains(keyword) | ChatLog.answer.contains(keyword)
        )
        count_stmt = count_stmt.where(
            ChatLog.question.contains(keyword) | ChatLog.answer.contains(keyword)
        )
    if created_from:
        stmt = stmt.where(ChatLog.created_at >= created_from)
        count_stmt = count_stmt.where(ChatLog.created_at >= created_from)
    if created_to:
        stmt = stmt.where(ChatLog.created_at <= created_to)
        count_stmt = count_stmt.where(ChatLog.created_at <= created_to)

    stmt = stmt.order_by(ChatLog.created_at.desc()).offset(offset).limit(limit)
    total = db.scalar(count_stmt) or 0
    items = list(db.scalars(stmt).all())
    return total, items


def export_chat_logs_csv(
    db: Session,
    created_from: datetime | None,
    created_to: datetime | None,
) -> str:
    stmt = select(ChatLog)
    if created_from:
        stmt = stmt.where(ChatLog.created_at >= created_from)
    if created_to:
        stmt = stmt.where(ChatLog.created_at <= created_to)
    stmt = stmt.order_by(ChatLog.created_at.asc())
    rows = list(db.scalars(stmt).all())

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "conversation_id",
            "user_id",
            "question",
            "answer",
            "kb_ids",
            "retrieval_top_k",
            "score_threshold",
            "elapsed_ms",
            "created_at",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.id,
                row.conversation_id,
                row.user_id,
                row.question,
                row.answer,
                row.kb_ids,
                row.retrieval_top_k,
                row.score_threshold,
                row.elapsed_ms,
                row.created_at.isoformat(),
            ]
        )
    return output.getvalue()


def cleanup_chat_logs(db: Session, before: datetime) -> int:
    result = db.execute(delete(ChatLog).where(ChatLog.created_at < before))
    db.commit()
    return result.rowcount or 0
