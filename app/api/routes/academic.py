from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_current_user
from app.core.errors import BusinessError
from app.models.rbac import User
from app.schemas.academic import AcademicAnalysisResponse
from app.services.academic_service import get_my_academic_analysis

router = APIRouter(prefix="/academic", tags=["academic"])


@router.get("/analysis/me", response_model=AcademicAnalysisResponse)
def get_academic_analysis_me(
    term_code: str | None = Query(default=None, max_length=32),
    current_user: User = Depends(get_current_user),
) -> AcademicAnalysisResponse:
    source_role = (current_user.department_name or "").strip().lower()
    if source_role == "teacher":
        raise BusinessError("Teachers are not allowed to access academic analysis", status_code=403)
    return get_my_academic_analysis(
        login_name=current_user.login_name,
        term_code=term_code,
    )
