from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.rbac import User
from app.services.auth_service import get_active_user, get_user_role_codes

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(status_code=401, detail="未登录")
    subject = decode_access_token(credentials.credentials)
    if subject is None or not subject.isdigit():
        raise HTTPException(status_code=401, detail="Token 无效")
    return get_active_user(db, int(subject))


def require_roles(*roles: str) -> Callable[[User, Session], User]:
    def dependency(
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db),
    ) -> User:
        user_roles = get_user_role_codes(db, current_user.id)
        if not user_roles.intersection(roles):
            raise HTTPException(status_code=403, detail="无权限")
        return current_user

    return dependency
