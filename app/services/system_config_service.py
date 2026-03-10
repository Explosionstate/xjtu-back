from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.sys_config import SysConfig
from app.core.config import settings

DEFAULT_CONTEXT_MAX_ROUNDS = 10
DEFAULT_CONTEXT_MAX_TOKENS = 3000


def list_system_configs(db: Session) -> list[SysConfig]:
    return list(
        db.scalars(
            select(SysConfig)
            .where(SysConfig.is_deleted == 0)
            .order_by(SysConfig.config_key.asc())
        ).all()
    )


def upsert_system_config(
    db: Session, config_key: str, config_value: str, value_type: str, description: str
) -> SysConfig:
    item = db.scalar(
        select(SysConfig).where(
            SysConfig.config_key == config_key,
        )
    )
    if item is None:
        item = SysConfig(
            config_key=config_key,
            config_value=config_value,
            value_type=value_type,
            description=description,
        )
        db.add(item)
    else:
        item.config_value = config_value
        item.value_type = value_type
        item.description = description
        item.is_deleted = 0
    db.commit()
    db.refresh(item)
    return item


def get_int_config(db: Session, key: str, default_value: int) -> int:
    item = db.scalar(
        select(SysConfig).where(
            SysConfig.config_key == key,
            SysConfig.is_deleted == 0,
        )
    )
    if item is None:
        return default_value
    try:
        return int(item.config_value)
    except ValueError:
        return default_value


def get_float_config(db: Session, key: str, default_value: float) -> float:
    item = db.scalar(
        select(SysConfig).where(
            SysConfig.config_key == key,
            SysConfig.is_deleted == 0,
        )
    )
    if item is None:
        return default_value
    try:
        return float(item.config_value)
    except ValueError:
        return default_value


def get_str_config(db: Session, key: str, default_value: str) -> str:
    item = db.scalar(
        select(SysConfig).where(
            SysConfig.config_key == key,
            SysConfig.is_deleted == 0,
        )
    )
    if item is None or not item.config_value:
        return default_value
    return item.config_value


def bootstrap_system_config(db: Session) -> None:
    upsert_system_config(
        db=db,
        config_key="context_max_rounds",
        config_value=str(DEFAULT_CONTEXT_MAX_ROUNDS),
        value_type="int",
        description="会话上下文最大轮数",
    )
    upsert_system_config(
        db=db,
        config_key="context_max_tokens",
        config_value=str(DEFAULT_CONTEXT_MAX_TOKENS),
        value_type="int",
        description="会话上下文最大token估算值",
    )
    upsert_system_config(
        db=db,
        config_key="retrieval_top_k",
        config_value="8",
        value_type="int",
        description="全局检索top_k",
    )
    upsert_system_config(
        db=db,
        config_key="retrieval_score_threshold",
        config_value="0.15",
        value_type="float",
        description="全局检索分数阈值",
    )
    upsert_system_config(
        db=db,
        config_key="retrieval_fusion_mode",
        config_value="weighted",
        value_type="string",
        description="全局混合检索融合模式",
    )
    upsert_system_config(
        db=db,
        config_key="retrieval_alpha",
        config_value="0.6",
        value_type="float",
        description="全局混合检索加权系数",
    )
    upsert_system_config(
        db=db,
        config_key="log_retention_days",
        config_value=str(settings.default_log_retention_days),
        value_type="int",
        description="对话日志默认保留天数",
    )
