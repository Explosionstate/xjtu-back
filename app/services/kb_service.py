from __future__ import annotations

from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import BusinessError
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.schemas.knowledge_base import (
    KnowledgeBaseCloneRequest,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseDeleteRequest,
    KnowledgeBaseUpdateRequest,
)
from app.vectorstore.chroma_manager import (
    clone_kb_vectorstore,
    delete_kb_vectorstore,
    ensure_kb_vectorstore,
)
from app.services.vectorstore_cleanup_service import enqueue_vectorstore_cleanup
from app.services.embedding_service import normalize_embedding_model_name


def create_knowledge_base(
    db: Session, payload: KnowledgeBaseCreateRequest
) -> KnowledgeBase:
    existing = db.scalar(
        select(KnowledgeBase).where(KnowledgeBase.name == payload.name)
    )
    if existing:
        raise BusinessError("知识库名称已存在", status_code=409)

    kb = KnowledgeBase(
        name=payload.name,
        description=payload.description,
        department=payload.department,
        owner=payload.owner,
        embedding_model=normalize_embedding_model_name(payload.embedding_model),
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)

    ensure_kb_vectorstore(kb.id)
    return kb


def list_knowledge_bases(
    db: Session,
    name: str | None,
    department: str | None,
    created_from: datetime | None,
    created_to: datetime | None,
    offset: int,
    limit: int,
) -> tuple[int, list[KnowledgeBase]]:
    stmt = select(KnowledgeBase)
    count_stmt = select(func.count()).select_from(KnowledgeBase)

    if name:
        stmt = stmt.where(KnowledgeBase.name.contains(name))
        count_stmt = count_stmt.where(KnowledgeBase.name.contains(name))
    if department:
        stmt = stmt.where(KnowledgeBase.department == department)
        count_stmt = count_stmt.where(KnowledgeBase.department == department)
    if created_from:
        stmt = stmt.where(KnowledgeBase.created_at >= created_from)
        count_stmt = count_stmt.where(KnowledgeBase.created_at >= created_from)
    if created_to:
        stmt = stmt.where(KnowledgeBase.created_at <= created_to)
        count_stmt = count_stmt.where(KnowledgeBase.created_at <= created_to)

    stmt = stmt.order_by(KnowledgeBase.created_at.desc()).offset(offset).limit(limit)
    total = db.scalar(count_stmt) or 0
    items = list(db.scalars(stmt).all())
    for kb in items:
        kb.document_count = (
            db.scalar(
                select(func.count())
                .select_from(Document)
                .where(Document.kb_id == kb.id)
            )
            or 0
        )
    return total, items


def update_knowledge_base(
    db: Session, kb_id: str, payload: KnowledgeBaseUpdateRequest
) -> KnowledgeBase:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None or kb.status == "deleted":
        raise BusinessError("知识库不存在", status_code=404)

    if payload.name and payload.name != kb.name:
        existing = db.scalar(
            select(KnowledgeBase).where(KnowledgeBase.name == payload.name)
        )
        if existing:
            raise BusinessError("知识库名称已存在", status_code=409)
        kb.name = payload.name
    if payload.description is not None:
        kb.description = payload.description
    if payload.department is not None:
        kb.department = payload.department
    if payload.owner is not None:
        kb.owner = payload.owner
    if payload.embedding_model is not None:
        kb.embedding_model = normalize_embedding_model_name(payload.embedding_model)

    db.commit()
    db.refresh(kb)
    return kb


def delete_knowledge_base(
    db: Session, kb_id: str, payload: KnowledgeBaseDeleteRequest
) -> bool:
    kb = db.get(KnowledgeBase, kb_id)
    if kb is None:
        raise BusinessError("知识库不存在", status_code=404)

    has_documents = db.scalar(
        select(func.count()).select_from(Document).where(Document.kb_id == kb_id)
    )
    if payload.physical and has_documents:
        raise BusinessError(
            "知识库下存在文档，请先删除文档后再物理删除", status_code=409
        )

    if payload.physical:
        cleanup_queued = not delete_kb_vectorstore(kb_id, raise_on_failure=False)
        if cleanup_queued:
            enqueue_vectorstore_cleanup(kb_id)
        db.delete(kb)
    else:
        kb.status = "deleted"
        cleanup_queued = False
    db.commit()
    return cleanup_queued


def clone_knowledge_base(
    db: Session, source_kb_id: str, payload: KnowledgeBaseCloneRequest
) -> KnowledgeBase:
    source = db.get(KnowledgeBase, source_kb_id)
    if not source:
        raise BusinessError("源知识库不存在", status_code=404)

    existing = db.scalar(
        select(KnowledgeBase).where(KnowledgeBase.name == payload.name)
    )
    if existing:
        raise BusinessError("目标知识库名称已存在", status_code=409)

    cloned = KnowledgeBase(
        name=payload.name,
        description=payload.description
        if payload.description is not None
        else source.description,
        department=payload.department
        if payload.department is not None
        else source.department,
        owner=payload.owner if payload.owner is not None else source.owner,
        embedding_model=normalize_embedding_model_name(
            payload.embedding_model
            if payload.embedding_model is not None
            else source.embedding_model
        ),
    )
    db.add(cloned)
    db.commit()
    db.refresh(cloned)

    clone_kb_vectorstore(source_kb_id=source.id, target_kb_id=cloned.id)
    cloned.document_count = 0
    return cloned
