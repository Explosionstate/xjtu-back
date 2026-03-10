from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.db.session import get_db
from app.schemas.system_config import (
    ContextPolicyResponse,
    SystemConfigItem,
    SystemConfigUpsertRequest,
)
from app.services.system_config_service import (
    DEFAULT_CONTEXT_MAX_ROUNDS,
    DEFAULT_CONTEXT_MAX_TOKENS,
    get_int_config,
    list_system_configs,
    upsert_system_config,
)

router = APIRouter(prefix="/system-config", tags=["system-config"])


@router.get("", response_model=list[SystemConfigItem])
def list_system_configs_api(
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> list[SystemConfigItem]:
    return [SystemConfigItem.model_validate(item) for item in list_system_configs(db)]


@router.put("/{config_key}", response_model=SystemConfigItem)
def upsert_system_config_api(
    config_key: str,
    payload: SystemConfigUpsertRequest,
    _: object = Depends(require_roles("super_admin")),
    db: Session = Depends(get_db),
) -> SystemConfigItem:
    item = upsert_system_config(
        db=db,
        config_key=config_key,
        config_value=payload.config_value,
        value_type=payload.value_type,
        description=payload.description,
    )
    return SystemConfigItem.model_validate(item)


@router.get("/context-policy", response_model=ContextPolicyResponse)
def get_context_policy_api(
    _: object = Depends(require_roles("super_admin", "kb_admin", "user")),
    db: Session = Depends(get_db),
) -> ContextPolicyResponse:
    return ContextPolicyResponse(
        max_rounds=get_int_config(db, "context_max_rounds", DEFAULT_CONTEXT_MAX_ROUNDS),
        max_tokens=get_int_config(db, "context_max_tokens", DEFAULT_CONTEXT_MAX_TOKENS),
    )
