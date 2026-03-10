from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.db.session import get_db
from app.schemas.chat_log import ChatLogItem, ChatLogListResponse
from app.services.chat_log_service import list_chat_logs
from app.services.chat_log_service import cleanup_chat_logs, export_chat_logs_csv
from app.services.system_config_service import get_int_config

router = APIRouter(prefix="/chat/logs", tags=["chat-logs"])


@router.get("", response_model=ChatLogListResponse)
def list_chat_logs_api(
    user_id: int | None = Query(default=None),
    conversation_id: str | None = Query(default=None),
    kb_id: str | None = Query(default=None),
    keyword: str | None = Query(default=None),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> ChatLogListResponse:
    total, items = list_chat_logs(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        kb_id=kb_id,
        keyword=keyword,
        created_from=created_from,
        created_to=created_to,
        offset=offset,
        limit=limit,
    )
    return ChatLogListResponse(
        total=total,
        items=[ChatLogItem.model_validate(item) for item in items],
    )


@router.get("/export")
def export_chat_logs_api(
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    csv_text = export_chat_logs_csv(
        db=db,
        created_from=created_from,
        created_to=created_to,
    )
    filename = f"chat_logs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([csv_text]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.delete("/cleanup")
def cleanup_chat_logs_api(
    before: datetime | None = Query(default=None),
    retention_days: int | None = Query(default=None, ge=1, le=3650),
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> dict[str, int]:
    if before is None:
        days = retention_days or get_int_config(db, "log_retention_days", 30)
        before = datetime.utcnow() - timedelta(days=days)
    deleted = cleanup_chat_logs(db=db, before=before)
    return {"deleted": deleted}
