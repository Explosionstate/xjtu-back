from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    login_name: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="user", index=True)
    name: Mapped[str] = mapped_column(String(64), default="")
    email: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    department_name: Mapped[str] = mapped_column(String(128), default="")
    last_login_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    is_deleted: Mapped[int] = mapped_column(Integer, default=0, index=True)
    gmt_created: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    gmt_modified: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user_roles: Mapped[list[SysUserRole]] = relationship(back_populates="user")


class SysRole(Base):
    __tablename__ = "sys_role"

    role_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    role_code: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    role_name: Mapped[str] = mapped_column(String(128))
    remark: Mapped[str] = mapped_column(String(255), default="")
    is_deleted: Mapped[int] = mapped_column(Integer, default=0, index=True)
    gmt_created: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    gmt_modified: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    role_permissions: Mapped[list[SysRolePermission]] = relationship(
        back_populates="role"
    )
    user_roles: Mapped[list[SysUserRole]] = relationship(back_populates="role")


class SysPermission(Base):
    __tablename__ = "sys_permission"

    perm_id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    perm_code: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    perm_name: Mapped[str] = mapped_column(String(255))
    resource: Mapped[str] = mapped_column(String(255), default="")
    http_method: Mapped[str] = mapped_column(String(16), default="")
    perm_type: Mapped[str] = mapped_column(String(32), default="api")
    is_deleted: Mapped[int] = mapped_column(Integer, default=0, index=True)
    gmt_created: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    gmt_modified: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    role_permissions: Mapped[list[SysRolePermission]] = relationship(
        back_populates="permission"
    )


class SysRolePermission(Base):
    __tablename__ = "sys_role_permission"

    role_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("sys_role.role_id", ondelete="CASCADE"), primary_key=True
    )
    perm_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("sys_permission.perm_id", ondelete="CASCADE"),
        primary_key=True,
    )

    role: Mapped[SysRole] = relationship(back_populates="role_permissions")
    permission: Mapped[SysPermission] = relationship(back_populates="role_permissions")


class SysUserRole(Base):
    __tablename__ = "sys_user_role"

    users_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("sys_role.role_id", ondelete="CASCADE"), primary_key=True
    )

    user: Mapped[User] = relationship(back_populates="user_roles")
    role: Mapped[SysRole] = relationship(back_populates="user_roles")


class AuthLoginAttempt(Base):
    __tablename__ = "auth_login_attempt"

    login_name: Mapped[str] = mapped_column(String(64), primary_key=True)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    last_failed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
