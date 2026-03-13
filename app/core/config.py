from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Disable Chroma telemetry in local deployments before chromadb import.
os.environ["ANONYMIZED_TELEMETRY"] = "False"


# Project-level fixed defaults for local deployment.
FIXED_LOCAL_MODELS_ROOT = Path("D:/xjtu/local_models")
FIXED_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"
FIXED_RERANKER_ENABLED = False
FIXED_RERANKER_MODEL = "BAAI/bge-reranker-base"
FIXED_RERANKER_TOP_N = 20
FIXED_RERANKER_WEIGHT = 0.7
FIXED_LLM_ENABLED = False
FIXED_LLM_TIMEOUT_SECONDS = 8
FIXED_XJTUEXER_SSO_CONSUME_URL = "http://127.0.0.1:8080/api/sso/consume"
FIXED_XJTUEXER_SSO_TIMEOUT_SECONDS = 5


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
    llm_timeout_seconds: int
    embedding_model_root: Path | None
    local_modules_root: Path | None
    local_models_root: Path | None
    chat_stream_delay_ms: int
    default_log_retention_days: int
    reranker_enabled: bool
    reranker_model: str
    reranker_top_n: int
    reranker_weight: float
    xjtuexer_sso_consume_url: str
    xjtuexer_sso_timeout_seconds: int


def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    workspace_root = project_root.parent
    data_root = project_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    embedding_root_raw = os.getenv("EMBEDDING_MODEL_ROOT")
    local_modules_raw = os.getenv("LOCAL_MODULES_ROOT")
    default_local_models = workspace_root / "local_models"
    default_local_modules = workspace_root / "local_modules"

    if FIXED_LOCAL_MODELS_ROOT.exists():
        local_models_root: Path | None = FIXED_LOCAL_MODELS_ROOT
    else:
        local_models_root = (
            default_local_models if default_local_models.exists() else None
        )
    local_modules_root = (
        Path(local_modules_raw)
        if local_modules_raw
        else (default_local_modules if default_local_modules.exists() else None)
    )

    return Settings(
        app_name=os.getenv("APP_NAME", "xjtu-back"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        database_url=os.getenv(
            "DB_URL", f"sqlite:///{(data_root / 'app.db').as_posix()}"
        ),
        chroma_root=Path(os.getenv("CHROMA_ROOT", (data_root / "chroma").as_posix())),
        default_embedding_model=FIXED_DEFAULT_EMBEDDING_MODEL,
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
        llm_enabled=FIXED_LLM_ENABLED,
        llm_timeout_seconds=FIXED_LLM_TIMEOUT_SECONDS,
        embedding_model_root=Path(embedding_root_raw)
        if embedding_root_raw
        else local_models_root,
        local_modules_root=local_modules_root,
        local_models_root=local_models_root,
        chat_stream_delay_ms=int(os.getenv("CHAT_STREAM_DELAY_MS", "10")),
        default_log_retention_days=int(os.getenv("DEFAULT_LOG_RETENTION_DAYS", "30")),
        reranker_enabled=FIXED_RERANKER_ENABLED,
        reranker_model=FIXED_RERANKER_MODEL,
        reranker_top_n=FIXED_RERANKER_TOP_N,
        reranker_weight=FIXED_RERANKER_WEIGHT,
        xjtuexer_sso_consume_url=FIXED_XJTUEXER_SSO_CONSUME_URL,
        xjtuexer_sso_timeout_seconds=FIXED_XJTUEXER_SSO_TIMEOUT_SECONDS,
    )


settings = get_settings()
