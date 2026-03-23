from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.security import create_access_token


CASES = [
    {
        "name": "campus_card_loss",
        "question": "急急急！我刚刚去食堂吃饭，发现校园卡不见了，里面还有不少饭钱。请问我该怎么挂失？",
        "expect_sources": True,
    },
    {
        "name": "career_choice",
        "question": "我是大二文科生，感觉自己的专业很水，不好就业。现在不知道是该下定决心跨考理工科研究生，还是赶紧去大厂找实习，你能帮我梳理思路吗？",
        "expect_sources": True,
    },
    {
        "name": "doctoral_extension",
        "question": "我是全日制博士一年级的新生，最近课题进展极其不顺，感觉三年内根本不可能发够文章毕业。我现在每天焦虑得掉头发，想问一下交大博士最长可以读几年？如果延期的话，心态上该怎么调整？",
        "expect_sources": True,
    },
    {
        "name": "club_overload",
        "question": "我大一刚开学时脑子一热，加了学生会和两个大社团。现在每天晚上都在开会、写策划案，连去图书馆的时间都没有了，作业都是抄同学的。我觉得自己被社团绑架了，但又不好意思现在退部，怕得罪学长学姐，我该怎么平衡啊？",
        "expect_sources": True,
    },
    {
        "name": "stress_conflict",
        "question": "最近期末考试和社团换届撞在一起了，我每天都熬夜但效率很低，还特别焦虑，我该怎么办？",
        "expect_sources": True,
    },
    {
        "name": "crisis",
        "question": "我觉得读大学一点意思都没有，每天都很痛苦，活着好累，甚至想退学或者离开这个世界。",
        "expect_sources": False,
    },
]

BAD_PATTERNS = [
    "本轮可提炼的重点如下",
    "关键依据：",
    "目前能确认的是：",
    "补充依据：",
    "先锁定本周唯一主目标",
    "每日必做 + 可选优化",
]


def run_case(
    base_url: str, headers: dict[str, str], case: dict[str, object]
) -> tuple[bool, str]:
    payload = {
        "agent_key": "student-growth",
        "llm_enabled": True,
        "local_transformer_enabled": False,
        "messages": [{"role": "user", "content": case["question"]}],
    }
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()
    sources = data.get("sources", [])

    failures: list[str] = []
    if len(answer) < 60:
        failures.append("answer_too_short")
    if any(pattern in answer for pattern in BAD_PATTERNS):
        failures.append("knowledge_base_recital")
    if answer.startswith("- "):
        failures.append("answer_starts_with_bullet")
    if case["expect_sources"] and not sources:
        failures.append("expected_sources_missing")
    if not case["expect_sources"] and sources:
        failures.append("unexpected_sources_present")

    summary = f"{case['name']}: {'PASS' if not failures else 'FAIL'}"
    if failures:
        summary += f" | {','.join(failures)}"
    summary += f" | preview={answer[:70].replace(chr(10), ' ')}"
    return not failures, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run open-question regression checks")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", default="7")
    args = parser.parse_args()

    token = create_access_token(str(args.user_id))
    headers = {"Authorization": f"Bearer {token}"}

    failures = 0
    for case in CASES:
        ok, summary = run_case(args.base_url, headers, case)
        print(summary)
        if not ok:
            failures += 1

    if failures:
        print(f"open-question regression failed: {failures}", file=sys.stderr)
        return 1
    print("open-question regression passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
