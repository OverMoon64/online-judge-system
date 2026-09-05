from __future__ import annotations

import json
from typing import Any, Protocol

import httpx

BACKEND_COOKIE_NAME = "oj_session"
BROWSER_COOKIE_KEY = "backend_session"


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


def serialize_backend_session(client: httpx.Client) -> str | None:
    value = backend_session_cookie(client)
    if value is None:
        return None
    return json.dumps({"version": 1, "cookie": value}, separators=(",", ":"))


def restore_backend_session(client: httpx.Client, payload: str | None) -> bool:
    if not payload:
        return False
    try:
        parsed = json.loads(payload)
    except (TypeError, json.JSONDecodeError):
        return False
    value = (
        parsed.get("cookie") if isinstance(parsed, dict) and parsed.get("version") == 1 else None
    )
    if not isinstance(value, str) or not value or len(value) > 8_192:
        return False
    client.cookies.clear()
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
