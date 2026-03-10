from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.chat import ConversationSetting
from app.services.system_config_service import (
    get_float_config,
    get_int_config,
    get_str_config,
)


def get_global_retrieval_config(db: Session) -> dict[str, float | int | str]:
    return {
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
    return config


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
    item = db.get(ConversationSetting, conversation_id)
    if item is None:
        item = ConversationSetting(conversation_id=conversation_id)
        db.add(item)

    item.retrieval_top_k = retrieval_top_k
    item.score_threshold = score_threshold
    item.fusion_mode = fusion_mode
    item.alpha = alpha
    db.commit()
    db.refresh(item)
    return item
