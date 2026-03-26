from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class AcademicStudentProfile(BaseModel):
    student_id: int
    login_name: str
    student_no: str | None = None
    student_name: str
    college_id: int | None = None
    college_name: str | None = None
    major_id: int | None = None
    major_name: str | None = None
    class_id: int | None = None
    class_name: str | None = None
    grade_year: int | None = None


class AcademicTermInfo(BaseModel):
    term_id: int
    term_code: str
    term_name: str
    academic_year: int | None = None
    term_no: int | None = None


class AcademicMetricSnapshot(BaseModel):
    avg_score: float | None = None
    gpa: float | None = None
    total_credits: float | None = None
    passed_credits: float | None = None
    class_rank: int | None = None
    major_rank: int | None = None
    college_rank: int | None = None
    cohort_size: int | None = None
    cumulative_avg_score: float | None = None
    cumulative_gpa: float | None = None
    cumulative_total_credits: float | None = None
    cumulative_passed_credits: float | None = None
    failed_course_count: int | None = None
    portrait_risk_level: str | None = None


class AcademicTrendPoint(BaseModel):
    term_id: int
    term_code: str
    term_name: str
    avg_score: float | None = None
    gpa: float | None = None
    class_rank: int | None = None
    major_rank: int | None = None


class AcademicCourseScoreItem(BaseModel):
    course_id: int
    course_name: str
    final_score: float
    gpa_point: float | None = None
    rank_in_class: int | None = None
    rank_in_major: int | None = None
    is_passed: bool = True


class AcademicCohortComparisonItem(BaseModel):
    scope_type: str
    scope_id: int
    scope_name: str
    sample_size: int
    avg_score: float | None = None
    avg_gpa: float | None = None
    pass_rate: float | None = None
    excellent_rate: float | None = None
    failure_rate: float | None = None


class AcademicWarningItem(BaseModel):
    warning_id: int
    warning_type: str
    warning_level: str
    risk_score: float
    status: str
    opened_at: datetime
    resolved_at: datetime | None = None


class AcademicAnalysisResponse(BaseModel):
    student: AcademicStudentProfile
    term: AcademicTermInfo
    metrics: AcademicMetricSnapshot
    trend: list[AcademicTrendPoint] = Field(default_factory=list)
    course_scores: list[AcademicCourseScoreItem] = Field(default_factory=list)
    cohort_comparison: list[AcademicCohortComparisonItem] = Field(default_factory=list)
    warnings: list[AcademicWarningItem] = Field(default_factory=list)
    risk_level: str
    key_findings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    generated_at: datetime


class AcademicInterpretRequest(BaseModel):
    term_code: str | None = Field(default=None, max_length=32)
    detail_level: Literal["brief", "detailed"] = "brief"


class AcademicInterpretResponse(BaseModel):
    analysis: AcademicAnalysisResponse
    interpretation: str
    detail_level: Literal["brief", "detailed"]
    llm_mode: str
    tool_used: bool = True
    generated_at: datetime
