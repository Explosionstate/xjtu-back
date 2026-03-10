from __future__ import annotations

from fastapi import APIRouter, Depends, File, Query, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.db.session import get_db
from app.schemas.document import (
    DocumentItem,
    DocumentListResponse,
    DocumentPreviewResponse,
    ReindexRequest,
    SplitPreviewRequest,
    SplitPreviewResponse,
)
from app.services.document_service import (
    delete_document,
    get_document_preview,
    list_documents,
    reindex_document,
    split_preview,
    upload_documents,
)

router = APIRouter(prefix="/knowledge-bases/{kb_id}/documents", tags=["documents"])


@router.post("/upload", response_model=list[DocumentItem])
def upload_docs(
    kb_id: str,
    files: list[UploadFile] = File(...),
    chunk_size: int | None = Query(default=None, ge=100, le=4000),
    chunk_overlap: int | None = Query(default=None, ge=0, le=1000),
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> list[DocumentItem]:
    items = upload_documents(
        db=db,
        kb_id=kb_id,
        files=files,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [DocumentItem.model_validate(item) for item in items]


@router.get("", response_model=DocumentListResponse)
def list_docs(
    kb_id: str,
    name: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=200),
    _: object = Depends(require_roles("super_admin", "kb_admin", "user")),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    total, items = list_documents(
        db=db, kb_id=kb_id, name=name, offset=offset, limit=limit
    )
    return DocumentListResponse(
        total=total, items=[DocumentItem.model_validate(item) for item in items]
    )


@router.get("/{document_id}/preview", response_model=DocumentPreviewResponse)
def preview_doc(
    kb_id: str,
    document_id: str,
    _: object = Depends(require_roles("super_admin", "kb_admin", "user")),
    db: Session = Depends(get_db),
) -> DocumentPreviewResponse:
    doc, raw_text, chunks = get_document_preview(
        db=db, kb_id=kb_id, document_id=document_id
    )
    return DocumentPreviewResponse(
        document_id=doc.id,
        file_name=doc.file_name,
        raw_text=raw_text,
        chunks=chunks,
    )


@router.post("/split-preview", response_model=SplitPreviewResponse)
def split_preview_api(
    kb_id: str,
    payload: SplitPreviewRequest,
    __: object = Depends(require_roles("super_admin", "kb_admin")),
) -> SplitPreviewResponse:
    _ = kb_id
    chunks = split_preview(
        text=payload.text,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    return SplitPreviewResponse(chunks=chunks)


@router.post("/{document_id}/reindex", response_model=DocumentItem)
def reindex_doc(
    kb_id: str,
    document_id: str,
    payload: ReindexRequest,
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> DocumentItem:
    doc = reindex_document(
        db=db,
        kb_id=kb_id,
        document_id=document_id,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
    )
    return DocumentItem.model_validate(doc)


@router.delete("/{document_id}")
def delete_doc(
    kb_id: str,
    document_id: str,
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    delete_document(db=db, kb_id=kb_id, document_id=document_id)
    return {"status": "ok"}
