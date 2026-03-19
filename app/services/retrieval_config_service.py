from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chat import ConversationSetting
from app.services.system_config_service import (
    get_float_config,
    get_int_config,
    get_str_config,
)


def _normalize_retrieval_config(
    config: Mapping[str, float | int | str | None],
) -> dict[str, float | int | str]:
    top_k = int(config.get("retrieval_top_k") or settings.retrieval_top_k)
    threshold = float(
        config.get("score_threshold") or settings.retrieval_score_threshold
    )
    alpha = float(config.get("alpha") or settings.retrieval_alpha)
    fusion_mode = str(config.get("fusion_mode") or settings.retrieval_fusion_mode)

    # Real-time oriented defaults to keep end-to-end response under 30s.
    top_k = max(2, min(8, top_k))
    threshold = max(0.1, min(0.5, threshold))
    alpha = max(0.0, min(1.0, alpha))
    if fusion_mode not in {"weighted", "rrf"}:
        fusion_mode = "weighted"

    normalized: dict[str, float | int | str] = {
        "retrieval_top_k": top_k,
        "score_threshold": threshold,
        "alpha": alpha,
        "fusion_mode": fusion_mode,
    }
    return normalized


def get_global_retrieval_config(db: Session) -> dict[str, float | int | str]:
    config = {
        "retrieval_top_k": get_int_config(
            db, "retrieval_top_k", settings.retrieval_top_k
        ),
        "score_threshold": get_float_config(
            db, "retrieval_score_threshold", settings.retrieval_score_threshold
        ),
        "fusion_mode": get_str_config(
            db, "retrieval_fusion_mode", settings.retrieval_fusion_mode
        ),
        "alpha": get_float_config(db, "retrieval_alpha", settings.retrieval_alpha),
    }
    return _normalize_retrieval_config(config)


def get_effective_retrieval_config(
    db: Session,
    conversation_id: str | None,
    payload_top_k: int | None,
    payload_score_threshold: float | None,
    payload_fusion_mode: str | None,
    payload_alpha: float | None,
) -> dict[str, float | int | str]:
    config = get_global_retrieval_config(db)
    if conversation_id:
        session_item = db.get(ConversationSetting, conversation_id)
        if session_item:
            if session_item.retrieval_top_k is not None:
                config["retrieval_top_k"] = session_item.retrieval_top_k
            if session_item.score_threshold is not None:
                config["score_threshold"] = session_item.score_threshold
            if session_item.fusion_mode is not None:
                config["fusion_mode"] = session_item.fusion_mode
            if session_item.alpha is not None:
                config["alpha"] = session_item.alpha

    if payload_top_k is not None:
        config["retrieval_top_k"] = payload_top_k
    if payload_score_threshold is not None:
        config["score_threshold"] = payload_score_threshold
    if payload_fusion_mode is not None:
        config["fusion_mode"] = payload_fusion_mode
    if payload_alpha is not None:
        config["alpha"] = payload_alpha
    return _normalize_retrieval_config(config)


def get_session_retrieval_config(
    db: Session, conversation_id: str
) -> ConversationSetting | None:
    return db.get(ConversationSetting, conversation_id)


def upsert_session_retrieval_config(
    db: Session,
    conversation_id: str,
    retrieval_top_k: int | None,
    score_threshold: float | None,
    fusion_mode: str | None,
    alpha: float | None,
) -> ConversationSetting:
    def _apply(target: ConversationSetting) -> None:
        target.retrieval_top_k = retrieval_top_k
        target.score_threshold = score_threshold
        target.fusion_mode = fusion_mode
        target.alpha = alpha
        normalized = _normalize_retrieval_config(
            {
                "retrieval_top_k": target.retrieval_top_k,
                "score_threshold": target.score_threshold,
                "fusion_mode": target.fusion_mode,
                "alpha": target.alpha,
            }
        )
        target.retrieval_top_k = int(normalized["retrieval_top_k"])
        target.score_threshold = float(normalized["score_threshold"])
        target.fusion_mode = str(normalized["fusion_mode"])
        target.alpha = float(normalized["alpha"])

    item = db.get(ConversationSetting, conversation_id)
    if item is None:
        item = ConversationSetting(conversation_id=conversation_id)
        db.add(item)
    _apply(item)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        existing = db.get(ConversationSetting, conversation_id)
        if existing is None:
            raise
        _apply(existing)
        db.commit()
        item = existing

    db.refresh(item)
    return item
