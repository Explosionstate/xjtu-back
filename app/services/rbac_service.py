from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.core.errors import BusinessError
from app.core.security import hash_password
from app.models.rbac import SysRole, SysUserRole, User
from app.schemas.rbac import (
    RoleCreateRequest,
    UserCreateRequest,
    UserRoleAssignRequest,
    UserUpdateRequest,
)


def create_user(db: Session, payload: UserCreateRequest) -> User:
    exists = db.scalar(select(User).where(User.login_name == payload.login_name))
    if exists:
        raise BusinessError("用户名已存在", status_code=409)
    user = User(
        login_name=payload.login_name,
        password=hash_password(payload.password),
        role=payload.role,
        name=payload.name,
        email=payload.email,
        department_name=payload.department_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def list_users(db: Session) -> list[User]:
    return list(db.scalars(select(User).order_by(User.gmt_created.desc())).all())


def update_user(db: Session, user_id: int, payload: UserUpdateRequest) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise BusinessError("用户不存在", status_code=404)
    if payload.password is not None:
        user.password = hash_password(payload.password)
    if payload.role is not None:
        user.role = payload.role
    if payload.name is not None:
        user.name = payload.name
    if payload.email is not None:
        user.email = payload.email
    if payload.department_name is not None:
        user.department_name = payload.department_name
    if payload.is_deleted is not None:
        user.is_deleted = payload.is_deleted
    db.commit()
    db.refresh(user)
    return user


def create_role(db: Session, payload: RoleCreateRequest) -> SysRole:
    exists = db.scalar(select(SysRole).where(SysRole.role_code == payload.role_code))
    if exists:
        raise BusinessError("角色编码已存在", status_code=409)
    role = SysRole(
        role_code=payload.role_code,
        role_name=payload.role_name,
        remark=payload.remark,
    )
    db.add(role)
    db.commit()
    db.refresh(role)
    return role


def list_roles(db: Session) -> list[SysRole]:
    return list(db.scalars(select(SysRole).where(SysRole.is_deleted == 0)).all())


def assign_user_roles(
    db: Session, user_id: int, payload: UserRoleAssignRequest
) -> None:
    user = db.get(User, user_id)
    if user is None:
        raise BusinessError("用户不存在", status_code=404)
    db.execute(delete(SysUserRole).where(SysUserRole.users_id == user_id))
    if payload.role_ids:
        roles = list(
            db.scalars(
                select(SysRole).where(
                    SysRole.role_id.in_(payload.role_ids), SysRole.is_deleted == 0
                )
            ).all()
        )
        for role in roles:
            db.add(SysUserRole(users_id=user_id, role_id=role.role_id))
    db.commit()
