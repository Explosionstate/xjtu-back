from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.chat import ChatMessage, SourceItem


class TransformerChatRequest(BaseModel):
    provider: str = Field(default="local_transformer")
    model: str | None = None
    messages: list[ChatMessage]
    kb_ids: list[str] | None = None
    document_ids: list[str] | None = None
    top_k: int = Field(default=6, ge=1, le=30)
    score_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    fusion_mode: str = Field(default="weighted")
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    max_new_tokens: int | None = Field(default=None, ge=64, le=2048)
    temperature: float | None = Field(default=None, ge=0.0, le=1.5)
    rag_enabled: bool = True


class TransformerChatResponse(BaseModel):
    provider: str
    model: str
    answer: str
    sources: list[SourceItem]
    diagnostics: dict[str, int | float | str | bool]


class TransformerClassifyRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=200)
    labels: list[str] = Field(min_length=2, max_length=50)
    model: str | None = None


class TransformerClassifyItem(BaseModel):
    text: str
    label: str
    score: float
    ranking: list[dict[str, float]]


class TransformerClassifyResponse(BaseModel):
    model: str
    items: list[TransformerClassifyItem]


class TransformerClusterRequest(BaseModel):
    texts: list[str] = Field(min_length=2, max_length=500)
    k: int = Field(default=4, ge=2, le=20)
    model: str | None = None
    max_iter: int = Field(default=20, ge=5, le=200)


class TransformerClusterGroup(BaseModel):
    cluster_id: int
    size: int
    sample_texts: list[str]


class TransformerClusterResponse(BaseModel):
    model: str
    assignments: list[int]
    groups: list[TransformerClusterGroup]


class TransformerRagAnalyzeRequest(BaseModel):
    topic: str = Field(min_length=2, max_length=200)
    aspects: list[str] = Field(default_factory=list)
    provider: str = "local_transformer"
    model: str | None = None
    kb_ids: list[str] | None = None
    document_ids: list[str] | None = None
    top_k: int = Field(default=8, ge=1, le=30)
    score_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    fusion_mode: str = Field(default="weighted")
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)


class TransformerRagAnalyzeResponse(BaseModel):
    topic: str
    provider: str
    model: str
    analysis: str
    sources: list[SourceItem]
    diagnostics: dict[str, int | float | str | bool]


class TransformerEvalSample(BaseModel):
    text: str
    expected_label: str


class TransformerEvalRequest(BaseModel):
    samples: list[TransformerEvalSample] = Field(min_length=1, max_length=1000)
    labels: list[str] = Field(min_length=2, max_length=50)
    model: str | None = None


class TransformerEvalResponse(BaseModel):
    model: str
    total: int
    correct: int
    accuracy: float


class TransformerRuntimeResponse(BaseModel):
    local_transformer_enabled: bool
    local_model: str
    transformer_device: str
    active_device: str
    cuda_available: bool
    max_concurrency: int
    queue_timeout_seconds: int
    embedding_model: str


class TransformerTopicTemplate(BaseModel):
    code: str
    title: str
    prompt: str
    keywords: list[str]


class TransformerTopicTemplateResponse(BaseModel):
    items: list[TransformerTopicTemplate]


class TransformerQuickTestRequest(BaseModel):
    provider: str = "local_transformer"
    model: str | None = None
    topic_codes: list[str] | None = None
    kb_ids: list[str] | None = None
    document_ids: list[str] | None = None
    top_k: int = Field(default=6, ge=1, le=30)
    score_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    fusion_mode: str = Field(default="weighted")
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    pass_threshold: float = Field(default=55.0, ge=0.0, le=100.0)
    max_topics: int = Field(default=11, ge=1, le=11)
    run_generation: bool = False


class TransformerQuickTestItem(BaseModel):
    code: str
    title: str
    prompt: str
    score: float
    passed: bool
    keyword_hits: int
    total_keywords: int
    retrieved_chunks: int
    mode: str
    latency_ms: int
    answer_preview: str


class TransformerQuickTestResponse(BaseModel):
    provider: str
    model: str
    total_topics: int
    pass_count: int
    average_score: float
    items: list[TransformerQuickTestItem]
