from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.errors import BusinessError
from app.db.academic_session import academic_session_scope
from app.schemas.academic import (
    AcademicAnalysisResponse,
    AcademicCohortComparisonItem,
    AcademicCourseScoreItem,
    AcademicMetricSnapshot,
    AcademicStudentProfile,
    AcademicTermInfo,
    AcademicTrendPoint,
    AcademicWarningItem,
)


def get_my_academic_analysis(
    login_name: str, term_code: str | None = None
) -> AcademicAnalysisResponse:
    if not login_name or not login_name.strip():
        raise BusinessError("login_name is required", status_code=400)

    normalized_login_name = login_name.strip()
    normalized_term_code = term_code.strip() if term_code else None

    try:
        with academic_session_scope() as db:
            _apply_query_timeout(db)
            student_row = _query_student_profile(db, normalized_login_name)
            if student_row is None:
                raise BusinessError(
                    "Current user is not a student in academic database",
                    status_code=404,
                )

            term_row = _resolve_term(db, normalized_term_code)
            if term_row is None:
                raise BusinessError("No term data available", status_code=404)

            student = AcademicStudentProfile(
                student_id=int(student_row["student_id"]),
                login_name=str(student_row["login_name"]),
                student_no=_as_optional_str(student_row.get("student_no")),
                student_name=str(student_row["student_name"]),
                college_id=_as_optional_int(student_row.get("college_id")),
                college_name=_as_optional_str(student_row.get("college_name")),
                major_id=_as_optional_int(student_row.get("major_id")),
                major_name=_as_optional_str(student_row.get("major_name")),
                class_id=_as_optional_int(student_row.get("class_id")),
                class_name=_as_optional_str(student_row.get("class_name")),
                grade_year=_as_optional_int(student_row.get("grade_year")),
            )

            term = AcademicTermInfo(
                term_id=int(term_row["term_id"]),
                term_code=str(term_row["term_code"]),
                term_name=str(term_row["term_name"]),
                academic_year=_as_optional_int(term_row.get("academic_year")),
                term_no=_as_optional_int(term_row.get("term_no")),
            )

            metrics = _load_metrics(db, student.student_id, term.term_id)
            trend = _load_trend(db, student.student_id, limit=6)
            course_scores = _load_course_scores(db, student.student_id, term.term_id)
            cohort_comparison = _load_cohort_comparison(
                db=db,
                term_id=term.term_id,
                college_id=student.college_id,
                college_name=student.college_name,
                major_id=student.major_id,
                major_name=student.major_name,
                class_id=student.class_id,
                class_name=student.class_name,
            )
            warnings = _load_warnings(db, student.student_id, term.term_id)

            risk_level = _evaluate_risk_level(metrics, warnings)
            key_findings = _build_key_findings(
                metrics=metrics,
                course_scores=course_scores,
                cohort_comparison=cohort_comparison,
                warnings=warnings,
            )
            recommendations = _build_recommendations(
                metrics=metrics,
                course_scores=course_scores,
                cohort_comparison=cohort_comparison,
                warnings=warnings,
                risk_level=risk_level,
            )

            return AcademicAnalysisResponse(
                student=student,
                term=term,
                metrics=metrics,
                trend=trend,
                course_scores=course_scores,
                cohort_comparison=cohort_comparison,
                warnings=warnings,
                risk_level=risk_level,
                key_findings=key_findings,
                recommendations=recommendations,
                generated_at=datetime.now(timezone.utc),
            )
    except BusinessError:
        raise
    except SQLAlchemyError as exc:
        raise BusinessError("Academic database query failed", status_code=502) from exc
    except Exception as exc:  # pragma: no cover
        raise BusinessError("Failed to build academic analysis", status_code=500) from exc


def _apply_query_timeout(db: Session) -> None:
    if not settings.academic_db_url.startswith("mysql"):
        return
    timeout_ms = max(1, int(settings.academic_query_timeout_seconds * 1000))
    db.execute(
        text("SET SESSION MAX_EXECUTION_TIME = :timeout_ms"),
        {"timeout_ms": timeout_ms},
    )


def _query_student_profile(db: Session, login_name: str) -> dict | None:
    sql = text(
        """
        SELECT
          s.student_id AS student_id,
          u.login_name AS login_name,
          s.student_no AS student_no,
          s.name AS student_name,
          COALESCE(se.college_id, s.college_id) AS college_id,
          COALESCE(dc.college_name, s.college, u.department_name) AS college_name,
          COALESCE(se.major_id, s.major_id) AS major_id,
          dm.major_name AS major_name,
          COALESCE(se.class_id, s.class_id) AS class_id,
          dcl.class_name AS class_name,
          COALESCE(se.grade_year, s.grade_year, dcl.grade_year) AS grade_year
        FROM `user` u
        JOIN student s ON s.student_id = u.id
        LEFT JOIN (
          SELECT
            se1.student_id,
            se1.college_id,
            se1.major_id,
            se1.class_id,
            se1.grade_year
          FROM student_enrollment se1
          JOIN (
            SELECT student_id, MAX(enrollment_id) AS max_enrollment_id
            FROM student_enrollment
            WHERE status = 'active'
            GROUP BY student_id
          ) latest
            ON latest.student_id = se1.student_id
           AND latest.max_enrollment_id = se1.enrollment_id
        ) se
          ON se.student_id = s.student_id
        LEFT JOIN dim_college dc
          ON dc.college_id = COALESCE(se.college_id, s.college_id)
        LEFT JOIN dim_major dm
          ON dm.major_id = COALESCE(se.major_id, s.major_id)
        LEFT JOIN dim_class dcl
          ON dcl.class_id = COALESCE(se.class_id, s.class_id)
        WHERE u.login_name = :login_name
          AND u.role = 'student'
          AND u.is_deleted = 0
        LIMIT 1
        """
    )
    row = db.execute(sql, {"login_name": login_name}).mappings().first()
    return dict(row) if row else None


def _resolve_term(db: Session, term_code: str | None) -> dict | None:
    if term_code:
        by_code = db.execute(
            text(
                """
                SELECT
                  term_id, term_code, term_name, academic_year, term_no
                FROM dim_term
                WHERE term_code = :term_code
                LIMIT 1
                """
            ),
            {"term_code": term_code},
        ).mappings().first()
        if by_code:
            return dict(by_code)

    ongoing = db.execute(
        text(
            """
            SELECT
              term_id, term_code, term_name, academic_year, term_no
            FROM dim_term
            WHERE status IN ('ongoing', 'current')
            ORDER BY academic_year DESC, term_no DESC, term_id DESC
            LIMIT 1
            """
        )
    ).mappings().first()
    if ongoing:
        return dict(ongoing)

    latest = db.execute(
        text(
            """
            SELECT
              term_id, term_code, term_name, academic_year, term_no
            FROM dim_term
            ORDER BY academic_year DESC, term_no DESC, term_id DESC
            LIMIT 1
            """
        )
    ).mappings().first()
    return dict(latest) if latest else None


def _load_metrics(db: Session, student_id: int, term_id: int) -> AcademicMetricSnapshot:
    ftg_row = db.execute(
        text(
            """
            SELECT
              avg_score,
              gpa,
              total_credits,
              passed_credits,
              class_rank,
              major_rank,
              college_rank,
              cohort_size
            FROM fact_term_gpa
            WHERE student_id = :student_id
              AND term_id = :term_id
            LIMIT 1
            """
        ),
        {"student_id": student_id, "term_id": term_id},
    ).mappings().first()

    portrait_row = db.execute(
        text(
            """
            SELECT
              cumulative_avg_score,
              cumulative_gpa,
              total_credits AS cumulative_total_credits,
              passed_credits AS cumulative_passed_credits,
              failed_course_count,
              risk_level AS portrait_risk_level
            FROM student_portrait
            WHERE student_id = :student_id
            LIMIT 1
            """
        ),
        {"student_id": student_id},
    ).mappings().first()

    legacy_row = None
    if ftg_row is None:
        legacy_row = db.execute(
            text(
                """
                SELECT
                  average_course_scores AS avg_score,
                  CASE
                    WHEN learning_index IS NULL THEN NULL
                    ELSE ROUND(learning_index / 25, 2)
                  END AS gpa
                FROM student
                WHERE student_id = :student_id
                LIMIT 1
                """
            ),
            {"student_id": student_id},
        ).mappings().first()

    merged = {}
    if ftg_row:
        merged.update(dict(ftg_row))
    elif legacy_row:
        merged.update(dict(legacy_row))
    if portrait_row:
        merged.update(dict(portrait_row))

    return AcademicMetricSnapshot(
        avg_score=_as_optional_float(merged.get("avg_score")),
        gpa=_as_optional_float(merged.get("gpa")),
        total_credits=_as_optional_float(merged.get("total_credits")),
        passed_credits=_as_optional_float(merged.get("passed_credits")),
        class_rank=_as_optional_int(merged.get("class_rank")),
        major_rank=_as_optional_int(merged.get("major_rank")),
        college_rank=_as_optional_int(merged.get("college_rank")),
        cohort_size=_as_optional_int(merged.get("cohort_size")),
        cumulative_avg_score=_as_optional_float(merged.get("cumulative_avg_score")),
        cumulative_gpa=_as_optional_float(merged.get("cumulative_gpa")),
        cumulative_total_credits=_as_optional_float(
            merged.get("cumulative_total_credits")
        ),
        cumulative_passed_credits=_as_optional_float(
            merged.get("cumulative_passed_credits")
        ),
        failed_course_count=_as_optional_int(merged.get("failed_course_count")),
        portrait_risk_level=_as_optional_str(merged.get("portrait_risk_level")),
    )


def _load_trend(db: Session, student_id: int, limit: int = 6) -> list[AcademicTrendPoint]:
    rows = db.execute(
        text(
            """
            SELECT
              dt.term_id,
              dt.term_code,
              dt.term_name,
              ftg.avg_score,
              ftg.gpa,
              ftg.class_rank,
              ftg.major_rank
            FROM fact_term_gpa ftg
            JOIN dim_term dt ON dt.term_id = ftg.term_id
            WHERE ftg.student_id = :student_id
            ORDER BY dt.academic_year DESC, dt.term_no DESC, dt.term_id DESC
            LIMIT :limit_count
            """
        ),
        {"student_id": student_id, "limit_count": limit},
    ).mappings().all()

    data = [
        AcademicTrendPoint(
            term_id=int(row["term_id"]),
            term_code=str(row["term_code"]),
            term_name=str(row["term_name"]),
            avg_score=_as_optional_float(row.get("avg_score")),
            gpa=_as_optional_float(row.get("gpa")),
            class_rank=_as_optional_int(row.get("class_rank")),
            major_rank=_as_optional_int(row.get("major_rank")),
        )
        for row in rows
    ]
    data.reverse()
    return data


def _load_course_scores(
    db: Session, student_id: int, term_id: int
) -> list[AcademicCourseScoreItem]:
    rows = db.execute(
        text(
            """
            SELECT
              fcs.course_id,
              c.title AS course_name,
              fcs.final_score,
              fcs.gpa_point,
              fcs.rank_in_class,
              fcs.rank_in_major,
              fcs.is_passed
            FROM fact_course_score fcs
            JOIN course c ON c.course_id = fcs.course_id
            WHERE fcs.student_id = :student_id
              AND fcs.term_id = :term_id
            ORDER BY fcs.final_score DESC, c.course_id ASC
            """
        ),
        {"student_id": student_id, "term_id": term_id},
    ).mappings().all()

    return [
        AcademicCourseScoreItem(
            course_id=int(row["course_id"]),
            course_name=str(row["course_name"]),
            final_score=float(row["final_score"]),
            gpa_point=_as_optional_float(row.get("gpa_point")),
            rank_in_class=_as_optional_int(row.get("rank_in_class")),
            rank_in_major=_as_optional_int(row.get("rank_in_major")),
            is_passed=bool(int(row.get("is_passed") or 0)),
        )
        for row in rows
    ]


def _load_warnings(
    db: Session, student_id: int, term_id: int
) -> list[AcademicWarningItem]:
    rows = db.execute(
        text(
            """
            SELECT
              warning_id,
              warning_type,
              warning_level,
              risk_score,
              status,
              opened_at,
              resolved_at
            FROM fact_warning_event
            WHERE student_id = :student_id
              AND (term_id = :term_id OR term_id IS NULL)
            ORDER BY
              CASE status WHEN 'open' THEN 0 WHEN 'resolved' THEN 1 ELSE 2 END,
              risk_score DESC,
              opened_at DESC
            LIMIT 20
            """
        ),
        {"student_id": student_id, "term_id": term_id},
    ).mappings().all()

    return [
        AcademicWarningItem(
            warning_id=int(row["warning_id"]),
            warning_type=str(row["warning_type"]),
            warning_level=str(row["warning_level"]),
            risk_score=float(row["risk_score"]),
            status=str(row["status"]),
            opened_at=row["opened_at"],
            resolved_at=row["resolved_at"],
        )
        for row in rows
    ]


def _load_cohort_comparison(
    db: Session,
    term_id: int,
    college_id: int | None,
    college_name: str | None,
    major_id: int | None,
    major_name: str | None,
    class_id: int | None,
    class_name: str | None,
) -> list[AcademicCohortComparisonItem]:
    scopes = [
        ("class", class_id, class_name, "class_id"),
        ("major", major_id, major_name, "major_id"),
        ("college", college_id, college_name, "college_id"),
    ]
    result: list[AcademicCohortComparisonItem] = []
    for scope_type, scope_id, scope_name, fallback_column in scopes:
        if scope_id is None:
            continue

        item = _query_agg_cohort_stat(db, term_id, scope_type, scope_id)
        if item is None:
            item = _fallback_cohort_stat(
                db=db,
                term_id=term_id,
                scope_id=scope_id,
                fallback_column=fallback_column,
            )
        if item is None:
            continue

        result.append(
            AcademicCohortComparisonItem(
                scope_type=scope_type,
                scope_id=scope_id,
                scope_name=scope_name or scope_type,
                sample_size=int(item["sample_size"]),
                avg_score=_as_optional_float(item.get("avg_score")),
                avg_gpa=_as_optional_float(item.get("avg_gpa")),
                pass_rate=_as_optional_float(item.get("pass_rate")),
                excellent_rate=_as_optional_float(item.get("excellent_rate")),
                failure_rate=_as_optional_float(item.get("failure_rate")),
            )
        )
    return result


def _query_agg_cohort_stat(
    db: Session, term_id: int, scope_type: str, scope_id: int
) -> dict | None:
    row = db.execute(
        text(
            """
            SELECT
              sample_size,
              avg_score,
              avg_gpa,
              pass_rate,
              excellent_rate,
              failure_rate
            FROM agg_cohort_stat
            WHERE term_id = :term_id
              AND scope_type = :scope_type
              AND scope_id = :scope_id
              AND course_id IS NULL
              AND metric_type = 'term_gpa'
            ORDER BY gmt_modified DESC
            LIMIT 1
            """
        ),
        {"term_id": term_id, "scope_type": scope_type, "scope_id": scope_id},
    ).mappings().first()
    return dict(row) if row else None


def _fallback_cohort_stat(
    db: Session, term_id: int, scope_id: int, fallback_column: str
) -> dict | None:
    if fallback_column not in {"class_id", "major_id", "college_id"}:
        return None

    sql = text(
        f"""
        SELECT
          COUNT(*) AS sample_size,
          ROUND(AVG(ftg.avg_score), 2) AS avg_score,
          ROUND(AVG(ftg.gpa), 2) AS avg_gpa,
          ROUND(100 * AVG(CASE WHEN ftg.avg_score >= 60 THEN 1 ELSE 0 END), 2) AS pass_rate,
          ROUND(100 * AVG(CASE WHEN ftg.avg_score >= 85 THEN 1 ELSE 0 END), 2) AS excellent_rate,
          ROUND(100 * AVG(CASE WHEN ftg.avg_score < 60 THEN 1 ELSE 0 END), 2) AS failure_rate
        FROM fact_term_gpa ftg
        JOIN student s ON s.student_id = ftg.student_id
        LEFT JOIN (
          SELECT
            se1.student_id,
            se1.college_id,
            se1.major_id,
            se1.class_id
          FROM student_enrollment se1
          JOIN (
            SELECT student_id, MAX(enrollment_id) AS max_enrollment_id
            FROM student_enrollment
            WHERE status = 'active'
            GROUP BY student_id
          ) latest
            ON latest.student_id = se1.student_id
           AND latest.max_enrollment_id = se1.enrollment_id
        ) se ON se.student_id = ftg.student_id
        WHERE ftg.term_id = :term_id
          AND COALESCE(se.{fallback_column}, s.{fallback_column}) = :scope_id
        """
    )
    row = db.execute(sql, {"term_id": term_id, "scope_id": scope_id}).mappings().first()
    if not row:
        return None
    sample_size = _as_optional_int(row.get("sample_size")) or 0
    if sample_size <= 0:
        return None
    return dict(row)


def _evaluate_risk_level(
    metrics: AcademicMetricSnapshot, warnings: list[AcademicWarningItem]
) -> str:
    risk = _normalize_risk(metrics.portrait_risk_level) or "low"

    for warning in warnings:
        if warning.status.lower() in {"open", "new"}:
            risk = _max_risk(risk, warning.warning_level)

    if metrics.avg_score is not None:
        if metrics.avg_score < 65:
            risk = _max_risk(risk, "high")
        elif metrics.avg_score < 75:
            risk = _max_risk(risk, "medium")

    if metrics.gpa is not None:
        if metrics.gpa < 2.0:
            risk = _max_risk(risk, "high")
        elif metrics.gpa < 2.5:
            risk = _max_risk(risk, "medium")

    return risk


def _build_key_findings(
    metrics: AcademicMetricSnapshot,
    course_scores: list[AcademicCourseScoreItem],
    cohort_comparison: list[AcademicCohortComparisonItem],
    warnings: list[AcademicWarningItem],
) -> list[str]:
    findings: list[str] = []
    if metrics.avg_score is not None:
        findings.append(f"Term average score: {metrics.avg_score:.2f}.")
    if metrics.gpa is not None:
        findings.append(f"Term GPA: {metrics.gpa:.2f}.")
    if metrics.class_rank is not None and metrics.cohort_size:
        findings.append(
            f"Class rank: {metrics.class_rank}/{metrics.cohort_size}."
        )

    weak_courses = sorted(course_scores, key=lambda x: x.final_score)[:3]
    weak_courses = [item for item in weak_courses if item.final_score < 70]
    if weak_courses:
        weak_text = ", ".join(
            f"{item.course_name}({item.final_score:.1f})" for item in weak_courses
        )
        findings.append(f"Weak courses detected: {weak_text}.")

    class_cmp = next(
        (item for item in cohort_comparison if item.scope_type == "class"),
        None,
    )
    if class_cmp and class_cmp.avg_score is not None and metrics.avg_score is not None:
        delta = metrics.avg_score - class_cmp.avg_score
        findings.append(f"Score gap to class average: {delta:+.2f}.")

    open_warnings = [item for item in warnings if item.status.lower() == "open"]
    if open_warnings:
        findings.append(f"Open warning events: {len(open_warnings)}.")

    if not findings:
        findings.append("Insufficient structured data for deeper findings.")
    return findings


def _build_recommendations(
    metrics: AcademicMetricSnapshot,
    course_scores: list[AcademicCourseScoreItem],
    cohort_comparison: list[AcademicCohortComparisonItem],
    warnings: list[AcademicWarningItem],
    risk_level: str,
) -> list[str]:
    recommendations: list[str] = []

    if risk_level in {"high", "critical"}:
        recommendations.append(
            "Prioritize high-risk items this week and review with counselor or advisor."
        )

    weak_courses = sorted(course_scores, key=lambda x: x.final_score)[:3]
    weak_courses = [item for item in weak_courses if item.final_score < 75]
    if weak_courses:
        names = ", ".join(item.course_name for item in weak_courses)
        recommendations.append(
            f"Focus on weak courses first: {names}. Use weekly checkpoints and quiz review."
        )

    class_cmp = next(
        (item for item in cohort_comparison if item.scope_type == "class"),
        None,
    )
    if (
        class_cmp
        and class_cmp.avg_score is not None
        and metrics.avg_score is not None
        and metrics.avg_score < class_cmp.avg_score - 5
    ):
        recommendations.append(
            "Current score is significantly below class average. Add 8-10 focused study hours per week."
        )

    open_high = [
        item
        for item in warnings
        if item.status.lower() == "open"
        and _normalize_risk(item.warning_level) in {"high", "critical"}
    ]
    if open_high:
        recommendations.append(
            "Resolve open high-risk warnings before starting advanced improvement plans."
        )

    if not recommendations:
        recommendations.append(
            "Performance is stable. Keep the current rhythm and continue strengthening core courses."
        )
    return recommendations


def _normalize_risk(level: str | None) -> str | None:
    if not level:
        return None
    normalized = level.strip().lower()
    if normalized in {"low", "medium", "high", "critical"}:
        return normalized
    return None


def _max_risk(left: str, right: str | None) -> str:
    order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
    r = _normalize_risk(right) or "low"
    return left if order.get(left, 0) >= order.get(r, 0) else r


def _as_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _as_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
