from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.sessions import SessionMiddleware

from app import ai_service
from app.api import router
from app.config import get_settings
from app.db import initialize_database
from app.errors import ApiError
from app.judge import cancel_all_submission_tasks

logger = logging.getLogger("oj")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await initialize_database()
    yield
    await cancel_all_submission_tasks()
    await ai_service.cancel_all_ai_tasks()


app = FastAPI(
    title="Async Online Judge",
    version="1.0.0",
    description="Course Online Judge implemented with fully asynchronous FastAPI APIs.",
    lifespan=lifespan,
)
app.add_middleware(
    SessionMiddleware,
    secret_key=get_settings().session_secret,
    session_cookie="oj_session",
    same_site="lax",
    https_only=get_settings().cookie_secure,
    max_age=8 * 60 * 60,
)
app.include_router(router)


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"code": status_code, "msg": message, "data": None},
    )


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return _error(exc.status_code, exc.message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, __: RequestValidationError) -> JSONResponse:
    return _error(400, "invalid parameters")


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    message = "resource not found" if exc.status_code == 404 else str(exc.detail)
    return _error(exc.status_code, message)


@app.exception_handler(Exception)
async def unexpected_error_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled server error", exc_info=exc)
    return _error(500, "internal server error")


def api_schema() -> dict[str, Any]:
    """Small importable hook used by smoke tests and documentation tooling."""
    return app.openapi()
