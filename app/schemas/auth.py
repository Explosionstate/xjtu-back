from __future__ import annotations

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    login_name: str = Field(
        min_length=1,
        max_length=64,
        validation_alias=AliasChoices("login_name", "loginName"),
    )
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class SSOExchangeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

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
