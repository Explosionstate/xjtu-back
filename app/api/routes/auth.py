from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.auth import (
    LoginRequest,
    SSOExchangeRequest,
    SSOExchangeResponse,
    TokenResponse,
    UserItem,
)
from app.services.auth_service import login, sso_exchange

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login_api(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    token = login(db=db, login_name=payload.login_name, password=payload.password)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserItem)
def me(current_user=Depends(get_current_user)) -> UserItem:
    return UserItem.model_validate(current_user)


@router.post("/sso/exchange", response_model=SSOExchangeResponse)
def sso_exchange_api(
    payload: SSOExchangeRequest, db: Session = Depends(get_db)
) -> SSOExchangeResponse:
    token, user, source_table, frontend_role = sso_exchange(
        db=db, ticket=payload.ticket
    )
    return SSOExchangeResponse(
        access_token=token,
        login_name=user.login_name,
        role=frontend_role,
        source_table=source_table,
    )
