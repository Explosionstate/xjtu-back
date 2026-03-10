from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SysConfig(Base):
    __tablename__ = "sys_config"

    config_id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    config_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    config_value: Mapped[str] = mapped_column(Text, default="")
    value_type: Mapped[str] = mapped_column(String(32), default="string")
    description: Mapped[str] = mapped_column(String(255), default="")
    is_deleted: Mapped[int] = mapped_column(Integer, default=0, index=True)
    gmt_created: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    gmt_modified: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
