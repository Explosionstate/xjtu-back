from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.core.errors import BusinessError
from app.models.rbac import User
from app.services.academic_service import get_my_academic_analysis


TOOL_GET_ACADEMIC_ANALYSIS = "get_academic_analysis"


@dataclass(frozen=True)
class ToolCallResult:
    name: str
    ok: bool
    data: dict[str, Any]
    error: str | None
    source: str
    generated_at: str


def execute_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None,
    *,
    current_user: User | None,
) -> ToolCallResult:
    args = dict(arguments or {})
    now = datetime.now(timezone.utc).isoformat()
    if tool_name != TOOL_GET_ACADEMIC_ANALYSIS:
        return ToolCallResult(
            name=tool_name,
            ok=False,
            data={},
            error="unsupported_tool",
            source="tooling_service",
            generated_at=now,
        )

    if current_user is None or not (current_user.login_name or "").strip():
        return ToolCallResult(
            name=tool_name,
            ok=False,
            data={},
            error="missing_user_context",
            source="academic_service",
            generated_at=now,
        )

    term_code = str(args.get("term_code") or "").strip() or None
    try:
        analysis = get_my_academic_analysis(
            login_name=current_user.login_name,
            term_code=term_code,
        )
    except BusinessError as exc:
        return ToolCallResult(
            name=tool_name,
            ok=False,
            data={},
            error=f"business_{exc.status_code}",
            source="academic_service",
            generated_at=now,
        )
    except Exception:
        return ToolCallResult(
            name=tool_name,
            ok=False,
            data={},
            error="internal_error",
            source="academic_service",
            generated_at=now,
        )

    payload = analysis.model_dump(mode="json")
    compact = {
        "risk_level": payload.get("risk_level"),
        "term": payload.get("term"),
        "metrics": payload.get("metrics"),
        "warnings": payload.get("warnings", [])[:5],
        "key_findings": payload.get("key_findings", [])[:6],
        "recommendations": payload.get("recommendations", [])[:6],
        "generated_at": payload.get("generated_at"),
    }
    return ToolCallResult(
        name=tool_name,
        ok=True,
        data=compact,
        error=None,
        source="academic_service",
        generated_at=now,
    )
