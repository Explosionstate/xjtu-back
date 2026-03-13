from __future__ import annotations

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    login_name: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SSOExchangeRequest(BaseModel):
    ticket: str = Field(min_length=1, max_length=256)


class SSOExchangeResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    login_name: str
    role: str
    source_table: str


class UserItem(BaseModel):
    id: int
    login_name: str
    role: str
    name: str
    email: str | None = None
    department_name: str

    class Config:
        from_attributes = True
