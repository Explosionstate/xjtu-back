from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    database_url: str
    chroma_root: Path
    default_embedding_model: str
    api_key: str | None
    docs_root: Path
    jwt_secret: str
    jwt_algorithm: str
    access_token_expire_minutes: int
    default_chunk_size: int
    default_chunk_overlap: int
    max_answer_chars: int
    retrieval_top_k: int
    retrieval_score_threshold: float
    retrieval_fusion_mode: str
    retrieval_alpha: float
    llm_base_url: str
    llm_model: str
    llm_temperature: float
    llm_enabled: bool
    embedding_model_root: Path | None
    local_modules_root: Path | None
    chat_stream_delay_ms: int
    default_log_retention_days: int


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    workspace_root = project_root.parent
    data_root = project_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    embedding_root_raw = os.getenv("EMBEDDING_MODEL_ROOT")
    local_modules_raw = os.getenv("LOCAL_MODULES_ROOT")

    return Settings(
        app_name=os.getenv("APP_NAME", "xjtu-back"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        database_url=os.getenv(
            "DB_URL", f"sqlite:///{(data_root / 'app.db').as_posix()}"
        ),
        chroma_root=Path(os.getenv("CHROMA_ROOT", (data_root / "chroma").as_posix())),
        default_embedding_model=os.getenv(
            "DEFAULT_EMBEDDING_MODEL", "bge-small-zh-v1.5"
        ),
        api_key=os.getenv("API_KEY"),
        docs_root=Path(os.getenv("DOCS_ROOT", (data_root / "docs").as_posix())),
        jwt_secret=os.getenv("JWT_SECRET", "xjtu-back-dev-secret"),
        jwt_algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        access_token_expire_minutes=int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120")
        ),
        default_chunk_size=int(os.getenv("DEFAULT_CHUNK_SIZE", "500")),
        default_chunk_overlap=int(os.getenv("DEFAULT_CHUNK_OVERLAP", "80")),
        max_answer_chars=int(os.getenv("MAX_ANSWER_CHARS", "800")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "8")),
        retrieval_score_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.15")),
        retrieval_fusion_mode=os.getenv("RETRIEVAL_FUSION_MODE", "weighted"),
        retrieval_alpha=float(os.getenv("RETRIEVAL_ALPHA", "0.6")),
        llm_base_url=os.getenv(
            "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        llm_model=os.getenv("LLM_MODEL", "qwen3.5-plus"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
        llm_enabled=os.getenv("LLM_ENABLED", "true").lower() in {"1", "true", "yes"},
        embedding_model_root=Path(embedding_root_raw)
        if embedding_root_raw
        else (workspace_root / "local_modules" / "models"),
        local_modules_root=Path(local_modules_raw)
        if local_modules_raw
        else (workspace_root / "local_modules"),
        chat_stream_delay_ms=int(os.getenv("CHAT_STREAM_DELAY_MS", "10")),
        default_log_retention_days=int(os.getenv("DEFAULT_LOG_RETENTION_DAYS", "30")),
    )


settings = get_settings()
