from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.schemas.auth import LoginRequest, TokenResponse, UserItem
from app.services.auth_service import login

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login_api(payload: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    token = login(db=db, login_name=payload.login_name, password=payload.password)
    return TokenResponse(access_token=token)


@router.get("/me", response_model=UserItem)
def me(current_user=Depends(get_current_user)) -> UserItem:
    return UserItem.model_validate(current_user)
