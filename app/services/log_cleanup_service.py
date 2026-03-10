from __future__ import annotations

from datetime import datetime, timedelta
from threading import Event

from app.db.session import SessionLocal
from app.services.chat_log_service import cleanup_chat_logs
from app.services.system_config_service import get_int_config


def run_periodic_log_cleanup(stop_event: Event, interval_minutes: int = 60) -> None:
    while not stop_event.is_set():
        db = SessionLocal()
        try:
            retention_days = get_int_config(db, "log_retention_days", 30)
            before = datetime.utcnow() - timedelta(days=retention_days)
            cleanup_chat_logs(db=db, before=before)
        finally:
            db.close()
        stop_event.wait(timeout=max(60, interval_minutes * 60))
