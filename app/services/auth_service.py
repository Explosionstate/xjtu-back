from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.errors import BusinessError
from app.core.security import create_access_token, hash_password, verify_password
from app.models.rbac import AuthLoginAttempt, SysRole, SysUserRole, User

MAX_LOGIN_RETRY = 5
LOCK_MINUTES = 10


def bootstrap_rbac(db: Session) -> None:
    role_codes = {
        "super_admin": "超级管理员",
        "kb_admin": "知识库管理员",
        "user": "普通用户",
    }
    existing = {item.role_code for item in db.scalars(select(SysRole)).all()}
    for code, name in role_codes.items():
        if code not in existing:
            db.add(SysRole(role_code=code, role_name=name, remark="bootstrap"))
    db.commit()

    admin = db.scalar(
        select(User).where(User.login_name == "admin", User.is_deleted == 0)
    )
    if admin is None:
        admin = User(
            login_name="admin",
            password=hash_password("admin123"),
            role="super_admin",
            name="管理员",
            email="admin@xjtu.local",
            department_name="system",
        )
        db.add(admin)
        db.commit()
        db.refresh(admin)

    super_admin_role = db.scalar(
        select(SysRole).where(
            SysRole.role_code == "super_admin", SysRole.is_deleted == 0
        )
    )
    if super_admin_role:
        linked = db.scalar(
            select(SysUserRole).where(
                SysUserRole.users_id == admin.id,
                SysUserRole.role_id == super_admin_role.role_id,
            )
        )
        if linked is None:
            db.add(SysUserRole(users_id=admin.id, role_id=super_admin_role.role_id))
            db.commit()


def login(db: Session, login_name: str, password: str) -> str:
    attempt = db.get(AuthLoginAttempt, login_name)
    if attempt and attempt.failed_count >= MAX_LOGIN_RETRY and attempt.last_failed_at:
        if datetime.utcnow() - attempt.last_failed_at < timedelta(minutes=LOCK_MINUTES):
            raise BusinessError("登录失败次数过多，请稍后重试", status_code=429)
        attempt.failed_count = 0

    user = db.scalar(
        select(User).where(User.login_name == login_name, User.is_deleted == 0)
    )
    if user is None or not verify_password(password, user.password):
        if attempt is None:
            attempt = AuthLoginAttempt(
                login_name=login_name, failed_count=1, last_failed_at=datetime.utcnow()
            )
            db.add(attempt)
        else:
            attempt.failed_count += 1
            attempt.last_failed_at = datetime.utcnow()
        db.commit()
        raise BusinessError("用户名或密码错误", status_code=401)

    if attempt is not None and attempt.failed_count > 0:
        attempt.failed_count = 0
        attempt.last_failed_at = None

    user.last_login_time = datetime.utcnow()
    db.commit()
    return create_access_token(str(user.id))


def get_active_user(db: Session, user_id: int) -> User:
    user = db.get(User, user_id)
    if user is None or user.is_deleted != 0:
        raise BusinessError("用户不存在或已禁用", status_code=401)
    return user


def get_user_role_codes(db: Session, user_id: int) -> set[str]:
    rows = db.execute(
        select(SysRole.role_code)
        .join(SysUserRole, SysUserRole.role_id == SysRole.role_id)
        .where(SysUserRole.users_id == user_id, SysRole.is_deleted == 0)
    ).all()
    role_codes = {r[0] for r in rows}
    if not role_codes:
        user = db.get(User, user_id)
        if user:
            role_codes.add(user.role)
    return role_codes
