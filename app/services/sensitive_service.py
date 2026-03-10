from __future__ import annotations

import re

from sqlalchemy.orm import Session

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
