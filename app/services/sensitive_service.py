from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.models.sensitive_audit import SensitiveBlockLog
from app.services.system_config_service import get_str_config


def get_sensitive_words(db: Session) -> list[str]:
    raw = get_str_config(db, "sensitive_words", "")
    words = [item.strip() for item in raw.split(",") if item.strip()]
    words.sort(key=len, reverse=True)
    return words


def mask_sensitive_text(text: str, words: list[str]) -> str:
    if not text or not words:
        return text
    output = text
    for word in words:
        pattern = re.escape(word)
        output = re.sub(pattern, "*" * len(word), output, flags=re.IGNORECASE)
    return output


def detect_sensitive_text(text: str, words: list[str]) -> str | None:
    if not text or not words:
        return None
    for word in words:
        pattern = re.escape(word)
        if re.search(pattern, text, flags=re.IGNORECASE):
            return word
    return None


def log_sensitive_block(
    db: Session,
    *,
    user_id: int | None,
    conversation_id: str,
    agent_key: str | None,
    direction: str,
    blocked_word: str,
    content: str,
) -> None:
    row = SensitiveBlockLog(
        user_id=user_id,
        conversation_id=conversation_id or "",
        agent_key=(agent_key or "").strip(),
        direction=direction,
        blocked_word=blocked_word,
        content_preview=(content or "")[:300],
    )
    db.add(row)
    db.commit()
