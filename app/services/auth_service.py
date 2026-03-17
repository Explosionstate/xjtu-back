from __future__ import annotations

from datetime import datetime, timedelta
import json
from urllib import error as urllib_error
from urllib import request as urllib_request
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import BusinessError
from app.core.security import create_access_token, hash_password, verify_password
from app.db.academic_session import academic_session_scope
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

    user = db.scalar(select(User).where(User.login_name == login_name))
    local_verified = _verify_local_password(password, user.password) if user else False

    if user is None or not local_verified:
        external = _authenticate_external_user(login_name=login_name, password=password)
        if external is not None:
            user = _upsert_external_user(
                db=db,
                existing_user=user,
                external=external,
                plain_password=password,
            )
            local_verified = True

    if user is None or not local_verified:
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

    if user and user.password == password:
        # Auto-upgrade legacy plain text password records.
        user.password = hash_password(password)

    if attempt is not None and attempt.failed_count > 0:
        attempt.failed_count = 0
        attempt.last_failed_at = None

    user.last_login_time = datetime.utcnow()
    db.commit()
    return create_access_token(str(user.id))


def _verify_local_password(plain_password: str, stored_password: str) -> bool:
    if not stored_password:
        return False
    if verify_password(plain_password, stored_password):
        return True
    # Compatible with legacy plaintext records.
    return plain_password == stored_password


def _authenticate_external_user(login_name: str, password: str) -> dict | None:
    try:
        with academic_session_scope() as adb:
            row = (
                adb.execute(
                    text(
                        """
                    SELECT id, login_name, role, name, email, department_name
                    FROM `user`
                    WHERE login_name = :login_name
                      AND password = :password
                      AND is_deleted = 0
                    LIMIT 1
                    """
                    ),
                    {"login_name": login_name, "password": password},
                )
                .mappings()
                .first()
            )
            if not row:
                return None
            return dict(row)
    except Exception:
        return None


def _upsert_external_user(
    db: Session,
    existing_user: User | None,
    external: dict,
    plain_password: str,
) -> User:
    source_role = str(external.get("role") or "student").strip().lower()
    mapped_role = "super_admin" if source_role == "admin" else "user"
    login_name = str(external.get("login_name") or "").strip()

    user = existing_user
    if user is None:
        user = User(
            login_name=login_name,
            password=hash_password(plain_password),
            role=mapped_role,
            name=str(external.get("name") or login_name),
            email=str(external.get("email") or "") or None,
            department_name=str(external.get("department_name") or "") or None,
            is_deleted=0,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        updated = False
        if user.is_deleted != 0:
            user.is_deleted = 0
            updated = True
        if user.role != mapped_role:
            user.role = mapped_role
            updated = True
        if not verify_password(plain_password, user.password):
            user.password = hash_password(plain_password)
            updated = True
        ext_name = str(external.get("name") or "").strip()
        if ext_name and user.name != ext_name:
            user.name = ext_name
            updated = True
        ext_email = str(external.get("email") or "").strip()
        if ext_email and user.email != ext_email:
            user.email = ext_email
            updated = True
        ext_dept = str(external.get("department_name") or "").strip()
        if ext_dept and user.department_name != ext_dept:
            user.department_name = ext_dept
            updated = True
        if updated:
            db.commit()

    _ensure_role_link(db, user.id, mapped_role)
    return user


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


def sso_exchange(db: Session, ticket: str) -> tuple[str, User, str, str]:
    payload = _consume_xjtuexer_ticket(ticket)
    login_name = payload.get("loginName")
    source_role = payload.get("role")
    source_table = payload.get("sourceTable") or "user"
    display_name = payload.get("displayName")

    if not login_name or not source_role:
        raise BusinessError("SSO票据缺少用户信息", status_code=401)

    normalized_source_role = str(source_role).strip().lower()
    mapped_role = _map_sso_role(normalized_source_role)
    user = db.scalar(
        select(User).where(User.login_name == login_name, User.is_deleted == 0)
    )

    if user is None:
        user = User(
            login_name=login_name,
            password=hash_password(f"sso:{uuid4()}"),
            role=mapped_role,
            name=display_name or login_name,
            email=None,
            department_name=source_table,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        updated = False
        if user.is_deleted != 0:
            user.is_deleted = 0
            updated = True
        if user.role != mapped_role:
            user.role = mapped_role
            updated = True
        if display_name and user.name != display_name:
            user.name = display_name
            updated = True
        if source_table and user.department_name != source_table:
            user.department_name = source_table
            updated = True
        if updated:
            db.commit()

    _ensure_role_link(db, user.id, mapped_role)
    user.last_login_time = datetime.utcnow()
    db.commit()

    token = create_access_token(str(user.id))
    frontend_role = _map_frontend_role(normalized_source_role, mapped_role)
    return token, user, source_table, frontend_role


def _consume_xjtuexer_ticket(ticket: str) -> dict:
    url = settings.xjtuexer_sso_consume_url
    req = urllib_request.Request(
        url=url,
        data=json.dumps({"ticket": ticket}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(
            req, timeout=settings.xjtuexer_sso_timeout_seconds
        ) as resp:
            raw = resp.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise BusinessError(
            f"SSO校验失败: {detail or exc.reason}", status_code=401
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise BusinessError("无法连接 xjtuexer SSO 服务", status_code=503) from exc

    try:
        body = json.loads(raw)
    except Exception as exc:
        raise BusinessError("SSO响应解析失败", status_code=502) from exc

    if not body.get("status"):
        raise BusinessError(body.get("message") or "SSO票据无效", status_code=401)
    data = body.get("data") or {}
    if not isinstance(data, dict):
        raise BusinessError("SSO响应格式错误", status_code=502)
    return data


def _map_sso_role(source_role: str) -> str:
    role = source_role.strip().lower()
    if role == "admin":
        return "super_admin"
    if role in {"teacher", "student"}:
        return "user"
    return "user"


def _map_frontend_role(source_role: str, mapped_role: str) -> str:
    if source_role in {"admin", "teacher", "student"}:
        return source_role
    if mapped_role == "super_admin":
        return "admin"
    return "student"


def _ensure_role_link(db: Session, user_id: int, role_code: str) -> None:
    role = db.scalar(
        select(SysRole).where(SysRole.role_code == role_code, SysRole.is_deleted == 0)
    )
    if role is None:
        return

    linked = db.scalar(
        select(SysUserRole).where(
            SysUserRole.users_id == user_id,
            SysUserRole.role_id == role.role_id,
        )
    )
    if linked is None:
        db.add(SysUserRole(users_id=user_id, role_id=role.role_id))
        db.commit()
