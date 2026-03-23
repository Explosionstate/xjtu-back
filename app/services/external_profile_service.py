from __future__ import annotations

from time import monotonic

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.db.academic_session import academic_session_scope

_PROFILE_CACHE_TTL_SECONDS = 180.0
_PROFILE_CACHE: dict[str, tuple[float, str]] = {}


def load_user_profile_context(login_name: str) -> str:
    if not login_name or not login_name.strip():
        return ""
    cache_key = login_name.strip().lower()
    cached = _PROFILE_CACHE.get(cache_key)
    now = monotonic()
    if cached and cached[0] > now:
        return cached[1]

    try:
        with academic_session_scope() as db:
            user_row = (
                db.execute(
                    text(
                        """
                    SELECT id, login_name, role, name, email, department_name
                    FROM `user`
                    WHERE login_name = :login_name AND is_deleted = 0
                    LIMIT 1
                    """
                    ),
                    {"login_name": login_name.strip()},
                )
                .mappings()
                .first()
            )
            if not user_row:
                _PROFILE_CACHE[cache_key] = (now + 30.0, "")
                return ""

            role = str(user_row.get("role") or "").strip().lower()
            lines: list[str] = [
                "外部业务库用户画像：",
                f"- 登录名: {user_row.get('login_name') or ''}",
                f"- 角色: {role}",
                f"- 姓名: {user_row.get('name') or ''}",
                f"- 学院/部门: {user_row.get('department_name') or ''}",
            ]

            if role == "student":
                student_row = (
                    db.execute(
                        text(
                            """
                        SELECT student_no, college, learning_index, comparison_last_month,
                               total_warnings, resolved_warnings, learning_scores, average_course_scores
                        FROM student
                        WHERE student_id = :uid
                        LIMIT 1
                        """
                        ),
                        {"uid": int(user_row["id"])},
                    )
                    .mappings()
                    .first()
                )
                if student_row:
                    lines.extend(
                        [
                            "学生画像补充：",
                            f"- 学号: {student_row.get('student_no') or ''}",
                            f"- 学院: {student_row.get('college') or ''}",
                            f"- 学情指数: {student_row.get('learning_index') or ''}",
                            f"- 对比上月: {student_row.get('comparison_last_month') or ''}",
                            f"- 累计预警: {student_row.get('total_warnings') or 0}",
                            f"- 累计解除: {student_row.get('resolved_warnings') or 0}",
                            f"- 学习成绩: {student_row.get('learning_scores') or ''}",
                            f"- 课程平均分: {student_row.get('average_course_scores') or ''}",
                        ]
                    )

            if role == "teacher":
                teacher_row = (
                    db.execute(
                        text(
                            """
                        SELECT teacher_no, title
                        FROM teacher
                        WHERE teacher_id = :uid
                        LIMIT 1
                        """
                        ),
                        {"uid": int(user_row["id"])},
                    )
                    .mappings()
                    .first()
                )
                if teacher_row:
                    lines.extend(
                        [
                            "教师画像补充：",
                            f"- 工号: {teacher_row.get('teacher_no') or ''}",
                            f"- 职称: {teacher_row.get('title') or ''}",
                        ]
                    )

            output = "\n".join(lines)
            _PROFILE_CACHE[cache_key] = (now + _PROFILE_CACHE_TTL_SECONDS, output)
            return output
    except SQLAlchemyError:
        return ""
