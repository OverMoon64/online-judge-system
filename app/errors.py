from __future__ import annotations

import json
from typing import Any, TypeVar

from fastapi import Request
from pydantic import BaseModel, ValidationError


class ApiError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def success(data: Any = None, message: str = "success") -> dict[str, Any]:
    return {"code": 200, "msg": message, "data": data}


SchemaT = TypeVar("SchemaT", bound=BaseModel)


async def parse_body(request: Request, schema: type[SchemaT]) -> SchemaT:
    try:
        raw = await request.body()
        if not raw:
            raise ValueError("request body is required")
        value = json.loads(raw)
        return schema.model_validate(value)
    except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
        raise ApiError(400, "invalid request body") from exc


def parse_positive_int(value: str | None, name: str) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(400, f"invalid {name}") from exc
    if number <= 0:
        raise ApiError(400, f"invalid {name}")
    return number


def parse_resource_id(value: str, name: str) -> int:
    result = parse_positive_int(value, name)
    assert result is not None
    return result


def parse_pagination(page_value: str | None, page_size_value: str | None) -> tuple[int, int | None]:
    if page_value is not None and page_size_value is None:
        raise ApiError(400, "page_size is required when page is provided")
    page_size = parse_positive_int(page_size_value, "page_size")
    page = parse_positive_int(page_value, "page") or 1
    if page_size is not None and page_size > 200:
        raise ApiError(400, "page_size must not exceed 200")
    return page, page_size
