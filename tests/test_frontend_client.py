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
