from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.db.session import get_db
from app.schemas.retrieval_config import (
    RetrievalConfigItem,
    RetrievalConfigUpdateRequest,
)
from app.services.retrieval_config_service import (
    get_global_retrieval_config,
    get_session_retrieval_config,
    upsert_session_retrieval_config,
)
from app.services.system_config_service import upsert_system_config

router = APIRouter(prefix="/retrieval-config", tags=["retrieval-config"])


@router.get("/global", response_model=RetrievalConfigItem)
def get_global_config_api(
    _: object = Depends(require_roles("super_admin", "kb_admin", "user")),
    db: Session = Depends(get_db),
) -> RetrievalConfigItem:
    config = get_global_retrieval_config(db)
    return RetrievalConfigItem(
        retrieval_top_k=int(config["retrieval_top_k"]),
        score_threshold=float(config["score_threshold"]),
        fusion_mode=str(config["fusion_mode"]),
        alpha=float(config["alpha"]),
    )


@router.put("/global", response_model=RetrievalConfigItem)
def update_global_config_api(
    payload: RetrievalConfigUpdateRequest,
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> RetrievalConfigItem:
    current = get_global_retrieval_config(db)
    top_k = payload.retrieval_top_k or int(current["retrieval_top_k"])
    score = (
        payload.score_threshold
        if payload.score_threshold is not None
        else float(current["score_threshold"])
    )
    fusion_mode = payload.fusion_mode or str(current["fusion_mode"])
    alpha = payload.alpha if payload.alpha is not None else float(current["alpha"])

    upsert_system_config(db, "retrieval_top_k", str(top_k), "int", "全局检索top_k")
    upsert_system_config(
        db,
        "retrieval_score_threshold",
        str(score),
        "float",
        "全局检索分数阈值",
    )
    upsert_system_config(
        db,
        "retrieval_fusion_mode",
        fusion_mode,
        "string",
        "全局混合检索融合模式",
    )
    upsert_system_config(db, "retrieval_alpha", str(alpha), "float", "全局融合系数")
    return RetrievalConfigItem(
        retrieval_top_k=top_k,
        score_threshold=score,
        fusion_mode=fusion_mode,
        alpha=alpha,
    )


@router.get("/sessions/{conversation_id}", response_model=RetrievalConfigItem)
def get_session_config_api(
    conversation_id: str,
    _: object = Depends(require_roles("super_admin", "kb_admin", "user")),
    db: Session = Depends(get_db),
) -> RetrievalConfigItem:
    global_cfg = get_global_retrieval_config(db)
    item = get_session_retrieval_config(db, conversation_id)
    if item is None:
        return RetrievalConfigItem(
            retrieval_top_k=int(global_cfg["retrieval_top_k"]),
            score_threshold=float(global_cfg["score_threshold"]),
            fusion_mode=str(global_cfg["fusion_mode"]),
            alpha=float(global_cfg["alpha"]),
        )
    return RetrievalConfigItem(
        retrieval_top_k=item.retrieval_top_k or int(global_cfg["retrieval_top_k"]),
        score_threshold=item.score_threshold
        if item.score_threshold is not None
        else float(global_cfg["score_threshold"]),
        fusion_mode=item.fusion_mode or str(global_cfg["fusion_mode"]),
        alpha=item.alpha if item.alpha is not None else float(global_cfg["alpha"]),
    )


@router.put("/sessions/{conversation_id}", response_model=RetrievalConfigItem)
def update_session_config_api(
    conversation_id: str,
    payload: RetrievalConfigUpdateRequest,
    _: object = Depends(require_roles("super_admin", "kb_admin", "user")),
    db: Session = Depends(get_db),
) -> RetrievalConfigItem:
    item = upsert_session_retrieval_config(
        db=db,
        conversation_id=conversation_id,
        retrieval_top_k=payload.retrieval_top_k,
        score_threshold=payload.score_threshold,
        fusion_mode=payload.fusion_mode,
        alpha=payload.alpha,
    )
    global_cfg = get_global_retrieval_config(db)
    return RetrievalConfigItem(
        retrieval_top_k=item.retrieval_top_k or int(global_cfg["retrieval_top_k"]),
        score_threshold=item.score_threshold
        if item.score_threshold is not None
        else float(global_cfg["score_threshold"]),
        fusion_mode=item.fusion_mode or str(global_cfg["fusion_mode"]),
        alpha=item.alpha if item.alpha is not None else float(global_cfg["alpha"]),
    )
