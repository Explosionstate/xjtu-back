from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.db.session import get_db
from app.schemas.auth import UserItem
from app.schemas.rbac import (
    RoleCreateRequest,
    RoleItem,
    UserCreateRequest,
    UserRoleAssignRequest,
    UserUpdateRequest,
)
from app.services.rbac_service import (
    assign_user_roles,
    create_role,
    create_user,
    list_roles,
    list_users,
    update_user,
)

router = APIRouter(prefix="/rbac", tags=["rbac"])


@router.post("/users", response_model=UserItem)
def create_user_api(
    payload: UserCreateRequest,
    _: object = Depends(require_roles("super_admin")),
    db: Session = Depends(get_db),
) -> UserItem:
    user = create_user(db=db, payload=payload)
    return UserItem.model_validate(user)


@router.get("/users", response_model=list[UserItem])
def list_users_api(
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> list[UserItem]:
    return [UserItem.model_validate(item) for item in list_users(db)]


@router.put("/users/{user_id}", response_model=UserItem)
def update_user_api(
    user_id: int,
    payload: UserUpdateRequest,
    _: object = Depends(require_roles("super_admin")),
    db: Session = Depends(get_db),
) -> UserItem:
    user = update_user(db=db, user_id=user_id, payload=payload)
    return UserItem.model_validate(user)


@router.post("/users/{user_id}/roles")
def assign_user_roles_api(
    user_id: int,
    payload: UserRoleAssignRequest,
    _: object = Depends(require_roles("super_admin")),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    assign_user_roles(db=db, user_id=user_id, payload=payload)
    return {"status": "ok"}


@router.post("/roles", response_model=RoleItem)
def create_role_api(
    payload: RoleCreateRequest,
    _: object = Depends(require_roles("super_admin")),
    db: Session = Depends(get_db),
) -> RoleItem:
    role = create_role(db=db, payload=payload)
    return RoleItem.model_validate(role)


@router.get("/roles", response_model=list[RoleItem])
def list_roles_api(
    _: object = Depends(require_roles("super_admin", "kb_admin")),
    db: Session = Depends(get_db),
) -> list[RoleItem]:
    return [RoleItem.model_validate(item) for item in list_roles(db)]
