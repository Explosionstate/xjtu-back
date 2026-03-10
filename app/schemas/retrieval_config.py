from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalConfigItem(BaseModel):
    retrieval_top_k: int
    score_threshold: float
    fusion_mode: str
    alpha: float


class RetrievalConfigUpdateRequest(BaseModel):
    retrieval_top_k: int | None = Field(default=None, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion_mode: str | None = None
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
