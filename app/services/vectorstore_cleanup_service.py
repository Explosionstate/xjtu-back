from __future__ import annotations

from datetime import datetime, timedelta
from threading import Event

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.cleanup_task import VectorstoreCleanupTask
from app.vectorstore.chroma_manager import delete_kb_vectorstore, kb_vectorstore_path


def enqueue_vectorstore_cleanup(kb_id: str) -> None:
    db = SessionLocal()
    try:
        task = db.scalar(
            select(VectorstoreCleanupTask).where(
                VectorstoreCleanupTask.kb_id == kb_id,
                VectorstoreCleanupTask.status.in_(["pending", "retrying"]),
            )
        )
        if task is None:
            db.add(
                VectorstoreCleanupTask(
                    kb_id=kb_id,
                    target_path=kb_vectorstore_path(kb_id).as_posix(),
                    status="pending",
                )
            )
            db.commit()
    finally:
        db.close()


def _process_one_task(task: VectorstoreCleanupTask) -> None:
    ok = delete_kb_vectorstore(task.kb_id, raise_on_failure=False)
    if ok:
        task.status = "done"
        task.last_error = ""
        return

    task.retry_count += 1
    task.status = "failed" if task.retry_count >= task.max_retries else "retrying"
    task.last_error = "vectorstore locked, retry later"
    backoff_seconds = min(300, 2 ** min(task.retry_count, 8))
    task.run_after = datetime.utcnow() + timedelta(seconds=backoff_seconds)


def run_periodic_vectorstore_cleanup(
    stop_event: Event, interval_seconds: int = 20
) -> None:
    while not stop_event.is_set():
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            tasks = list(
                db.scalars(
                    select(VectorstoreCleanupTask)
                    .where(
                        VectorstoreCleanupTask.status.in_(["pending", "retrying"]),
                        VectorstoreCleanupTask.run_after <= now,
                    )
                    .order_by(VectorstoreCleanupTask.created_at.asc())
                    .limit(20)
                ).all()
            )
            for task in tasks:
                _process_one_task(task)
            if tasks:
                db.commit()
        finally:
            db.close()

        stop_event.wait(timeout=max(5, interval_seconds))
