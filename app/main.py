from __future__ import annotations

import json
from threading import Event, Thread
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, Response

try:
    import swagger_ui_bundle
except ImportError:  # pragma: no cover
    swagger_ui_bundle = None

from app.api.routes import api_router
from app.core.api_compat import (
    build_api_success,
    is_api_compat_path,
    is_api_envelope,
)
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.db.base import Base
from app.db.session import engine
from app.db.session import SessionLocal
from app.services.auth_service import bootstrap_rbac
from app.services.log_cleanup_service import run_periodic_log_cleanup
from app.services.system_config_service import bootstrap_system_config
from app.services.vectorstore_cleanup_service import run_periodic_vectorstore_cleanup
from app import models  # noqa: F401


def create_app() -> FastAPI:
    app = FastAPI(title=settings.app_name, version=settings.app_version, docs_url=None)
    app.openapi_version = "3.0.3"
    cleanup_stop = Event()
    cleanup_thread: Thread | None = None
    vector_cleanup_thread: Thread | None = None

    register_exception_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5174",
            "http://localhost:5174",
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:5175",
            "http://localhost:5175",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(api_router)
    app.include_router(api_router, prefix="/api")
    if swagger_ui_bundle is not None:
        app.mount(
            "/_static/swagger",
            StaticFiles(directory=swagger_ui_bundle.swagger_ui_path),
            name="swagger-static",
        )

    @app.middleware("http")
    async def api_compat_envelope_middleware(request, call_next):
        response = await call_next(request)
        if not is_api_compat_path(request.url.path):
            return response

        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type:
            return response

        body_bytes = b""
        async for chunk in response.body_iterator:
            body_bytes += chunk

        if not body_bytes:
            payload: Any = None
        else:
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                passthrough_headers = {
                    key: value
                    for key, value in response.headers.items()
                    if key.lower() != "content-length"
                }
                return Response(
                    content=body_bytes,
                    status_code=response.status_code,
                    headers=passthrough_headers,
                    media_type=response.media_type,
                )

        if is_api_envelope(payload):
            wrapped = payload
        else:
            wrapped = build_api_success(payload)

        passthrough_headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }
        return JSONResponse(status_code=200, content=wrapped, headers=passthrough_headers)

    @app.on_event("startup")
    def startup() -> None:
        nonlocal cleanup_thread, vector_cleanup_thread
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

        vector_cleanup_thread = Thread(
            target=run_periodic_vectorstore_cleanup,
            kwargs={"stop_event": cleanup_stop, "interval_seconds": 20},
            daemon=True,
        )
        vector_cleanup_thread.start()

    @app.on_event("shutdown")
    def shutdown() -> None:
        cleanup_stop.set()
        if cleanup_thread and cleanup_thread.is_alive():
            cleanup_thread.join(timeout=2)
        if vector_cleanup_thread and vector_cleanup_thread.is_alive():
            vector_cleanup_thread.join(timeout=2)

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
