from __future__ import annotations

from collections.abc import Mapping
from typing import Any


API_PREFIX = "/api"
SUCCESS_CODE = 0
AUTH_REQUIRED_CODE = 70005


def is_api_compat_path(path: str | None) -> bool:
    value = (path or "").strip()
    return value == API_PREFIX or value.startswith(f"{API_PREFIX}/")


def map_error_code(status_code: int) -> int:
    if status_code in {401, 403}:
        return AUTH_REQUIRED_CODE
    if status_code >= 400:
        return status_code
    return 500


def build_api_success(
    data: Any,
    message: str = "",
    *,
    code: int = SUCCESS_CODE,
) -> dict[str, Any]:
    return {
        "status": True,
        "code": int(code),
        "message": message,
        "data": data,
    }


def build_api_error(
    message: str,
    *,
    status_code: int = 400,
    data: Any = None,
    code: int | None = None,
) -> dict[str, Any]:
    error_code = int(code) if code is not None else map_error_code(status_code)
    return {
        "status": False,
        "code": error_code,
        "message": (message or "请求失败").strip(),
        "data": data,
    }


def is_api_envelope(payload: Any) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return {"status", "code", "message", "data"}.issubset(payload.keys())

