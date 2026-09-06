from __future__ import annotations

import json
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_COOKIE_NAME = "oj_session"
BROWSER_COOKIE_KEY = "backend_session"


class BrowserSessionSettings(BaseSettings):
    session_secret: str = "development-only-change-this-secret"

    model_config = SettingsConfigDict(
        env_prefix="OJ_",
        env_file=Path(__file__).resolve().parents[1] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def browser_cookie_password() -> str:
    return BrowserSessionSettings().session_secret


@lru_cache
def browser_runtime_id() -> str:
    """Identify one Streamlit server process for refresh-only session restore."""
    return secrets.token_urlsafe(24)


class BrowserCookieStore(Protocol):
    def ready(self) -> bool: ...

    def save(self) -> Any: ...

    def __contains__(self, key: object) -> bool: ...

    def __getitem__(self, key: str) -> str: ...

    def __setitem__(self, key: str, value: str) -> None: ...

    def __delitem__(self, key: str) -> None: ...


def backend_session_cookie(client: httpx.Client) -> str | None:
    for cookie in client.cookies.jar:
        if cookie.name == BACKEND_COOKIE_NAME and cookie.value:
            return cookie.value
    return None


def serialize_backend_session(client: httpx.Client, *, runtime_id: str | None = None) -> str | None:
    value = backend_session_cookie(client)
    if value is None:
        return None
    return json.dumps(
        {
            "version": 2,
            "runtime_id": runtime_id or browser_runtime_id(),
            "cookie": value,
        },
        separators=(",", ":"),
    )


def restore_backend_session(
    client: httpx.Client,
    payload: str | None,
    *,
    runtime_id: str | None = None,
) -> bool:
    client.cookies.clear()
    if not payload:
        return False
    try:
        parsed = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return False
    expected_runtime_id = runtime_id or browser_runtime_id()
    value = None
    if (
        isinstance(parsed, dict)
        and parsed.get("version") == 2
        and parsed.get("runtime_id") == expected_runtime_id
    ):
        value = parsed.get("cookie")
    if not isinstance(value, str) or not value or len(value) > 8_192:
        return False
    client.cookies.set(BACKEND_COOKIE_NAME, value)
    return True


def load_browser_session(store: BrowserCookieStore) -> str | None:
    if not store.ready():
        return None
    try:
        return store[BROWSER_COOKIE_KEY] if BROWSER_COOKIE_KEY in store else None
    except (KeyError, TypeError, ValueError):
        return None


def save_browser_session(store: BrowserCookieStore, payload: str) -> bool:
    if not store.ready():
        return False
    store[BROWSER_COOKIE_KEY] = payload
    store.save()
    return True


def clear_browser_session(store: BrowserCookieStore | None) -> bool:
    if store is None or not store.ready():
        return False
    if BROWSER_COOKIE_KEY in store:
        del store[BROWSER_COOKIE_KEY]
        store.save()
    return True
