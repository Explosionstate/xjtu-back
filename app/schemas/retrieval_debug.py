from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalDebugRequest(BaseModel):
    query: str = Field(min_length=1)
    kb_ids: list[str] | None = None
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion_mode: str | None = None
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)


class RetrievalDebugScoreItem(BaseModel):
    chunk_id: str
    document_id: str
    source_location: str
    bm25_raw: float
    bm25_norm: float
    dense_raw: float
    dense_norm: float
    fused_score: float
    rerank_score: float
    final_score: float
    content: str


class RetrievalDebugResponse(BaseModel):
    top_k_results: list[RetrievalDebugScoreItem]
    all_candidates: list[RetrievalDebugScoreItem]
