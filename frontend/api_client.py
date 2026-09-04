from __future__ import annotations

import json
from collections.abc import Callable, MutableMapping
from typing import Any

import httpx

CLIENT_STATE_KEY = "_api_http_client"
BASE_URL_STATE_KEY = "_api_http_base_url"


def normalize_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    if not normalized:
        raise ValueError("API Base URL cannot be empty")
    return normalized


def _create_client(base_url: str) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(20.0),
        follow_redirects=True,
    )


def get_persistent_client(
    state: MutableMapping[str, Any],
    base_url: str,
    *,
    client_factory: Callable[[str], httpx.Client] | None = None,
) -> httpx.Client:
    """Return one stable httpx client for the lifetime of a Streamlit session."""
    normalized = normalize_base_url(base_url)
    existing = state.get(CLIENT_STATE_KEY)
    stored_base_url = state.get(BASE_URL_STATE_KEY)
    if (
        isinstance(existing, httpx.Client)
        and not existing.is_closed
        and stored_base_url == normalized
    ):
        return existing

    if isinstance(existing, httpx.Client):
        existing.cookies.clear()
        existing.close()

    factory = client_factory or _create_client
    client = factory(normalized)
    state[CLIENT_STATE_KEY] = client
    state[BASE_URL_STATE_KEY] = normalized
    return client


def clear_client_cookies(state: MutableMapping[str, Any]) -> None:
    client = state.get(CLIENT_STATE_KEY)
    if isinstance(client, httpx.Client) and not client.is_closed:
        client.cookies.clear()


def close_persistent_client(state: MutableMapping[str, Any]) -> None:
    client = state.pop(CLIENT_STATE_KEY, None)
    state.pop(BASE_URL_STATE_KEY, None)
    if isinstance(client, httpx.Client):
        client.cookies.clear()
        if not client.is_closed:
            client.close()


def request_json(client: httpx.Client, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = client.request(method, path, **kwargs)
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError
        return body
    except httpx.HTTPError as exc:
        return {"code": 503, "msg": f"无法连接后端：{exc}", "data": None}
    except (ValueError, json.JSONDecodeError):
        return {"code": 502, "msg": "后端返回了无法解析的响应", "data": None}
