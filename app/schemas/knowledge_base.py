from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str = ""
    department: str = ""
    owner: str = ""
    embedding_model: str | None = None


class KnowledgeBaseCloneRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    description: str | None = None
    department: str | None = None
    owner: str | None = None
    embedding_model: str | None = None


class KnowledgeBaseUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=128)
    description: str | None = None
    department: str | None = None
    owner: str | None = None
    embedding_model: str | None = None


class KnowledgeBaseItem(BaseModel):
    id: str
    name: str
    description: str
    department: str
    owner: str
    status: str
    embedding_model: str
    created_at: datetime
    updated_at: datetime
    document_count: int = 0

    class Config:
        from_attributes = True


class KnowledgeBaseListResponse(BaseModel):
    total: int
    items: list[KnowledgeBaseItem]


class KnowledgeBaseDeleteRequest(BaseModel):
    physical: bool = False
