from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class RetrievalConfigItem(BaseModel):
    retrieval_top_k: int
    score_threshold: float
    fusion_mode: str
    alpha: float


class RetrievalConfigUpdateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    retrieval_top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        validation_alias=AliasChoices("retrieval_top_k", "retrievalTopK", "top_k", "topK"),
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
