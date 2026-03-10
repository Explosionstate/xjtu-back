from __future__ import annotations

from pydantic import BaseModel, Field


class SystemConfigItem(BaseModel):
    config_key: str
    config_value: str
    value_type: str
    description: str

    class Config:
        from_attributes = True


class SystemConfigUpsertRequest(BaseModel):
    config_value: str
    value_type: str = Field(default="string", max_length=32)
    description: str = ""


class ContextPolicyResponse(BaseModel):
    max_rounds: int
    max_tokens: int
