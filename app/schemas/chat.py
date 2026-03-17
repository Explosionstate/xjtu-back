from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    agent_key: str | None = None
    messages: list[ChatMessage]
    stream: bool = False
    conversation_id: str | None = None
    kb_ids: list[str] | None = None
    document_ids: list[str] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fusion_mode: str | None = None
    alpha: float | None = Field(default=None, ge=0.0, le=1.0)
    context_max_rounds: int | None = Field(default=None, ge=1, le=100)
    context_max_tokens: int | None = Field(default=None, ge=100, le=16000)
    llm_enabled: bool | None = None


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
