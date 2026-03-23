from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.api_compat import build_api_error, is_api_compat_path


logger = logging.getLogger(__name__)


class BusinessError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def register_exception_handlers(app: FastAPI) -> None:
    def _api_compat_error(
        request: Request, *, message: str, status_code: int, data: object = None
    ) -> JSONResponse:
        if is_api_compat_path(request.url.path):
            return JSONResponse(
                status_code=200,
                content=build_api_error(
                    message=message,
                    status_code=status_code,
                    data=data,
                ),
            )
        return JSONResponse(status_code=status_code, content={"detail": message})

    @app.exception_handler(BusinessError)
    async def handle_business_error(
        request: Request, exc: BusinessError
    ) -> JSONResponse:
        return _api_compat_error(
            request,
            message=exc.message,
            status_code=exc.status_code,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return _api_compat_error(
            request,
            message=str(exc.detail or "请求失败"),
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = exc.errors()
        first_error = errors[0] if errors else {}
        message = str(first_error.get("msg") or "参数校验失败")
        return _api_compat_error(
            request,
            message=message,
            status_code=422,
            data={"errors": errors},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("unexpected error: %s", exc)
        return _api_compat_error(
            request,
            message="服务器内部异常",
            status_code=500,
        )
