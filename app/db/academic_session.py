from __future__ import annotations

from contextlib import contextmanager
from threading import RLock

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


_academic_engine: Engine | None = None
_academic_session_factory: sessionmaker[Session] | None = None
_academic_lock = RLock()


def get_academic_engine() -> Engine:
    global _academic_engine
    if _academic_engine is not None:
        return _academic_engine

    with _academic_lock:
        if _academic_engine is None:
            connect_args = (
                {"check_same_thread": False}
                if settings.academic_db_url.startswith("sqlite")
                else {
                    "connect_timeout": 3,
                    "read_timeout": 6,
                    "write_timeout": 6,
                }
            )
            _academic_engine = create_engine(
                settings.academic_db_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=2,
                pool_timeout=6,
                pool_recycle=300,
                future=True,
                connect_args=connect_args,
            )
    return _academic_engine


def get_academic_session_factory() -> sessionmaker[Session]:
    global _academic_session_factory
    if _academic_session_factory is not None:
        return _academic_session_factory

    with _academic_lock:
        if _academic_session_factory is None:
            _academic_session_factory = sessionmaker(
                bind=get_academic_engine(),
                autocommit=False,
                autoflush=False,
                class_=Session,
            )
    return _academic_session_factory


@contextmanager
def academic_session_scope() -> Session:
    session = get_academic_session_factory()()
    try:
        yield session
    finally:
        session.close()
