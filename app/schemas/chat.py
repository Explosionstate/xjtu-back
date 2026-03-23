from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    model: str | None = None
    agent_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("agent_key", "agentKey"),
    )
    messages: list[ChatMessage]
    stream: bool = False
    conversation_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("conversation_id", "conversationId"),
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
    context_max_rounds: int | None = Field(
        default=None,
        ge=1,
        le=100,
        validation_alias=AliasChoices("context_max_rounds", "contextMaxRounds"),
    )
    context_max_tokens: int | None = Field(
        default=None,
        ge=100,
        le=16000,
        validation_alias=AliasChoices("context_max_tokens", "contextMaxTokens"),
    )
    llm_enabled: bool | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "llm_enabled",
            "llmEnabled",
            "enable_qwen35_plus",
            "enableQwen35Plus",
            "use_cloud",
            "useCloud",
        ),
    )
    local_transformer_enabled: bool | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "local_transformer_enabled",
            "localTransformerEnabled",
            "enable_local_qwen",
            "enableLocalQwen",
            "use_local_qwen",
            "useLocalQwen",
        ),
    )


class SourceItem(BaseModel):
    source_location: str
    content: str
    score: float


class ChatThinking(BaseModel):
    title: str
    content: str
    kind: str = "summary"
    is_real: bool = False
    collapsed: bool = True


class ChatCompletionResponseMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    conversation_id: str
    choices: list[ChatCompletionChoice]
    sources: list[SourceItem]
    thinking: ChatThinking | None = None
