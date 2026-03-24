from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RetrievalDebugRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    query: str = Field(min_length=1)
    agent_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("agent_key", "agentKey"),
    )
    kb_ids: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("kb_ids", "kbIds"),
    )
    document_ids: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("document_ids", "documentIds"),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        validation_alias=AliasChoices("top_k", "topK"),
    )
    score_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("score_threshold", "scoreThreshold"),
    )
    fusion_mode: str | None = Field(
        default=None,
        validation_alias=AliasChoices("fusion_mode", "fusionMode"),
    )
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
