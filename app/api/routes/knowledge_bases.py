from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.db.session import get_db
from app.schemas.knowledge_base import (
    KnowledgeBaseCloneRequest,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseDeleteRequest,
    KnowledgeBaseItem,
    KnowledgeBaseListResponse,
    KnowledgeBaseUpdateRequest,
)
from app.services.kb_service import (
    clone_knowledge_base,
    create_knowledge_base,
    delete_knowledge_base,
    list_knowledge_bases,
    update_knowledge_base,
)


router = APIRouter(prefix="/knowledge-bases", tags=["knowledge-bases"])


@router.post("", response_model=KnowledgeBaseItem)
def create_kb(
    payload: KnowledgeBaseCreateRequest,
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> KnowledgeBaseItem:
    kb = create_knowledge_base(db, payload)
    kb.document_count = 0
    return KnowledgeBaseItem.model_validate(kb)


@router.get("", response_model=KnowledgeBaseListResponse)
def list_kb(
    name: str | None = Query(default=None),
    department: str | None = Query(default=None),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _: object = Depends(require_roles("super_admin", "kb_admin", "user")),
    db: Session = Depends(get_db),
) -> KnowledgeBaseListResponse:
    total, items = list_knowledge_bases(
        db=db,
        name=name,
        department=department,
        created_from=created_from,
        created_to=created_to,
        offset=offset,
        limit=limit,
    )
    return KnowledgeBaseListResponse(
        total=total, items=[KnowledgeBaseItem.model_validate(item) for item in items]
    )


@router.post("/{kb_id}/clone", response_model=KnowledgeBaseItem)
def clone_kb(
    kb_id: str,
    payload: KnowledgeBaseCloneRequest,
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> KnowledgeBaseItem:
    kb = clone_knowledge_base(db, source_kb_id=kb_id, payload=payload)
    kb.document_count = 0
    return KnowledgeBaseItem.model_validate(kb)


@router.put("/{kb_id}", response_model=KnowledgeBaseItem)
def update_kb(
    kb_id: str,
    payload: KnowledgeBaseUpdateRequest,
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> KnowledgeBaseItem:
    kb = update_knowledge_base(db=db, kb_id=kb_id, payload=payload)
    kb.document_count = 0
    return KnowledgeBaseItem.model_validate(kb)


@router.delete("/{kb_id}")
def delete_kb(
    kb_id: str,
    physical: bool = Query(default=False),
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> dict[str, str | bool]:
    cleanup_queued = delete_knowledge_base(
        db=db, kb_id=kb_id, payload=KnowledgeBaseDeleteRequest(physical=physical)
    )
    return {"status": "ok", "cleanup_queued": cleanup_queued}
