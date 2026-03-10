from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field


class UserCreateRequest(BaseModel):
    login_name: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    role: str = Field(default="user", max_length=32)
    name: str = ""
    email: EmailStr | None = None
    department_name: str = ""


class UserUpdateRequest(BaseModel):
    password: str | None = Field(default=None, min_length=6, max_length=128)
    role: str | None = Field(default=None, max_length=32)
    name: str | None = None
    email: EmailStr | None = None
    department_name: str | None = None
    is_deleted: int | None = Field(default=None, ge=0, le=1)


class UserRoleAssignRequest(BaseModel):
    role_ids: list[int] = Field(default_factory=list)


class RoleCreateRequest(BaseModel):
    role_code: str = Field(min_length=1, max_length=64)
    role_name: str = Field(min_length=1, max_length=128)
    remark: str = ""


class RoleItem(BaseModel):
    role_id: int
    role_code: str
    role_name: str
    remark: str

    class Config:
        from_attributes = True
