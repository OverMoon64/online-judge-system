from __future__ import annotations

from typing import Any

import httpx

from frontend.api_client import (
    BASE_URL_STATE_KEY,
    CLIENT_STATE_KEY,
    close_persistent_client,
    get_persistent_client,
    request_json,
)
from frontend.browser_session import (
    clear_browser_session,
    load_browser_session,
    restore_backend_session,
    save_browser_session,
    serialize_backend_session,
)


class FakeBrowserCookies(dict[str, str]):
    def __init__(self) -> None:
        super().__init__()
        self.save_count = 0

    def ready(self) -> bool:
        return True

    def save(self) -> None:
        self.save_count += 1


def test_cookie_survives_rerun_and_logout_revokes_session() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        cookie = request.headers.get("cookie", "")
        if request.url.path == "/api/auth/login":
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "login success",
                    "data": {"user_id": "1", "username": "admin", "role": "admin"},
                },
                headers={"set-cookie": "oj_session=signed-session; Path=/; HttpOnly"},
            )
        if request.url.path == "/api/auth/logout":
            assert "oj_session=signed-session" in cookie
            return httpx.Response(
                200,
                json={"code": 200, "msg": "logout success", "data": None},
                headers={"set-cookie": "oj_session=; Path=/; Max-Age=0"},
            )
        if request.url.path == "/api/users/1":
            if "oj_session=signed-session" not in cookie:
                return httpx.Response(401, json={"code": 401, "msg": "not logged in", "data": None})
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "success",
                    "data": {"user_id": "1", "username": "admin", "role": "admin"},
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    transport = httpx.MockTransport(handler)
    state: dict[str, Any] = {}

    def factory(base_url: str) -> httpx.Client:
        return httpx.Client(base_url=base_url, transport=transport)

    first = get_persistent_client(state, "http://127.0.0.1:8000/", client_factory=factory)
    assert (
        request_json(
            first,
            "POST",
            "/api/auth/login",
            json={"username": "admin", "password": "secret"},
        )["code"]
        == 200
    )

    # A Streamlit rerun calls the helper again. The raw httpx.Client is reused,
    # so the signed cookie remains attached to the following profile request.
    after_rerun = get_persistent_client(state, "http://127.0.0.1:8000", client_factory=factory)
    assert after_rerun is first
    assert request_json(after_rerun, "GET", "/api/users/1")["code"] == 200

    assert request_json(after_rerun, "POST", "/api/auth/logout")["code"] == 200
    assert request_json(after_rerun, "GET", "/api/users/1")["code"] == 401

    close_persistent_client(state)
    assert CLIENT_STATE_KEY not in state
    assert BASE_URL_STATE_KEY not in state
    assert first.is_closed


def test_encrypted_browser_payload_restores_after_full_refresh_and_logout_clears_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        cookie = request.headers.get("cookie", "")
        if request.url.path == "/api/auth/login":
            return httpx.Response(
                200,
                json={
                    "code": 200,
                    "msg": "login success",
                    "data": {"user_id": "1", "username": "admin", "role": "admin"},
                },
                headers={"set-cookie": "oj_session=signed-session; Path=/; HttpOnly"},
            )
        if request.url.path == "/api/auth/session":
            status = 200 if "oj_session=signed-session" in cookie else 401
            return httpx.Response(
                status,
                json={
                    "code": status,
                    "msg": "success" if status == 200 else "not logged in",
                    "data": (
                        {"user_id": "1", "username": "admin", "role": "admin"}
                        if status == 200
                        else None
                    ),
                },
            )
        raise AssertionError(f"unexpected request: {request.url}")

    transport = httpx.MockTransport(handler)
    first = httpx.Client(base_url="http://127.0.0.1:8000", transport=transport)
    assert request_json(first, "POST", "/api/auth/login", json={})["code"] == 200
    payload = serialize_backend_session(first, runtime_id="frontend-runtime-a")
    assert payload is not None
    browser = FakeBrowserCookies()
    assert save_browser_session(browser, payload)

    refreshed = httpx.Client(base_url="http://127.0.0.1:8000", transport=transport)
    assert restore_backend_session(
        refreshed,
        load_browser_session(browser),
        runtime_id="frontend-runtime-a",
    )
    assert request_json(refreshed, "GET", "/api/auth/session")["code"] == 200

    restarted = httpx.Client(base_url="http://127.0.0.1:8000", transport=transport)
    assert not restore_backend_session(
        restarted,
        load_browser_session(browser),
        runtime_id="frontend-runtime-b",
    )
    assert request_json(restarted, "GET", "/api/auth/session")["code"] == 401

    assert clear_browser_session(browser)
    logged_out = httpx.Client(base_url="http://127.0.0.1:8000", transport=transport)
    assert not restore_backend_session(logged_out, load_browser_session(browser))
    assert request_json(logged_out, "GET", "/api/auth/session")["code"] == 401
    assert browser.save_count == 2
