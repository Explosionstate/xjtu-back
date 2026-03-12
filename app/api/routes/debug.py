from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.core.config import settings
from app.db.session import get_db
from app.services.embedding_service import resolve_local_model_reference
from app.services.retrieval_config_service import get_global_retrieval_config

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/runtime")
def runtime_debug(
    _: object = Depends(require_roles("super_admin", "kb_admin", "user")),
    db: Session = Depends(get_db),
) -> dict:
    retrieval = get_global_retrieval_config(db)
    try:
        embedding_ref = resolve_local_model_reference(settings.default_embedding_model)
    except Exception as exc:  # pragma: no cover
        embedding_ref = f"unresolved: {exc}"

    return {
        "llm": {
            "enabled": settings.llm_enabled,
            "model": settings.llm_model,
            "base_url": settings.llm_base_url,
            "timeout_seconds": settings.llm_timeout_seconds,
        },
        "retrieval": retrieval,
        "embedding": {
            "default_model": settings.default_embedding_model,
            "resolved_reference": embedding_ref,
            "local_models_root": str(settings.local_models_root)
            if settings.local_models_root
            else "",
            "reranker_model": settings.reranker_model,
            "reranker_enabled": settings.reranker_enabled,
            "reranker_top_n": settings.reranker_top_n,
            "reranker_weight": settings.reranker_weight,
        },
        "storage": {
            "database_url": settings.database_url,
            "docs_root": str(settings.docs_root),
            "chroma_root": str(settings.chroma_root),
        },
    }
