from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DocumentItem(BaseModel):
    id: str
    kb_id: str
    file_name: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    uploaded_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentItem]


class DocumentPreviewResponse(BaseModel):
    document_id: str
    file_name: str
    raw_text: str
    chunks: list[str]


class SplitPreviewRequest(BaseModel):
    text: str = Field(min_length=1)
    chunk_size: int = Field(default=500, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)


class SplitPreviewResponse(BaseModel):
    chunks: list[str]


class ReindexRequest(BaseModel):
    chunk_size: int | None = Field(default=None, ge=100, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)


class DocumentBatchDeleteRequest(BaseModel):
    document_ids: list[str] = Field(default_factory=list, min_length=1)
