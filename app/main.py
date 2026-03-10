from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import api_router
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.db.base import Base
from app.db.session import engine
from app.db.session import SessionLocal
from app.services.auth_service import bootstrap_rbac
from app import models  # noqa: F401


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version)

    register_exception_handlers(app)
    app.include_router(api_router)

    @app.on_event("startup")
    def startup() -> None:
        settings.chroma_root.mkdir(parents=True, exist_ok=True)
        settings.docs_root.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            bootstrap_rbac(db)
        finally:
            db.close()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
