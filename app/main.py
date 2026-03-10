from __future__ import annotations

from threading import Event, Thread

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

try:
    import swagger_ui_bundle
except ImportError:  # pragma: no cover
    swagger_ui_bundle = None

from app.api.routes import api_router
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.db.base import Base
from app.db.session import engine
from app.db.session import SessionLocal
from app.services.auth_service import bootstrap_rbac
from app.services.log_cleanup_service import run_periodic_log_cleanup
from app.services.system_config_service import bootstrap_system_config
from app import models  # noqa: F401


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version, docs_url=None)
    app.openapi_version = "3.0.3"
    cleanup_stop = Event()
    cleanup_thread: Thread | None = None

    register_exception_handlers(app)
    app.include_router(api_router)
    if swagger_ui_bundle is not None:
        app.mount(
            "/_static/swagger",
            StaticFiles(directory=swagger_ui_bundle.swagger_ui_path),
            name="swagger-static",
        )

    @app.on_event("startup")
    def startup() -> None:
        nonlocal cleanup_thread
        settings.chroma_root.mkdir(parents=True, exist_ok=True)
        settings.docs_root.mkdir(parents=True, exist_ok=True)
        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            bootstrap_rbac(db)
            bootstrap_system_config(db)
        finally:
            db.close()

        cleanup_thread = Thread(
            target=run_periodic_log_cleanup,
            kwargs={"stop_event": cleanup_stop, "interval_minutes": 60},
            daemon=True,
        )
        cleanup_thread.start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        cleanup_stop.set()
        if cleanup_thread and cleanup_thread.is_alive():
            cleanup_thread.join(timeout=2)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/docs", include_in_schema=False)
    def custom_swagger_ui_html():
        if swagger_ui_bundle is None:
            return get_swagger_ui_html(
                openapi_url=app.openapi_url,
                title=f"{app.title} - Swagger UI",
            )
        return get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=f"{app.title} - Swagger UI",
            swagger_js_url="/_static/swagger/swagger-ui-bundle.js",
            swagger_css_url="/_static/swagger/swagger-ui.css",
        )

    return app


app = create_app()
