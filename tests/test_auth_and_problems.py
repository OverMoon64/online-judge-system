from __future__ import annotations

import httpx

from tests.conftest import add_sum_problem, login, register


async def test_error_envelope_and_auth_precedence(client: httpx.AsyncClient) -> None:
    response = await client.post("/api/problems/", content=b"not-json")
    assert response.status_code == 401
    assert response.json() == {"code": 401, "msg": "not logged in", "data": None}

    assert (await login(client)).status_code == 200
    response = await client.post("/api/problems/", content=b"not-json")
    assert response.status_code == 400
    assert response.json()["code"] == 400

    response = await client.get("/route-that-does-not-exist")
    assert response.status_code == 404
    assert response.json()["data"] is None


async def test_registration_login_roles_and_statistics(client: httpx.AsyncClient) -> None:
    alice = await register(client, "alice")
    duplicate = await client.post(
        "/api/users/", json={"username": "alice", "password": "password123"}
    )
    assert duplicate.status_code == 400
    assert (await login(client, "alice", "wrong-password")).status_code == 401
    assert (await login(client, "alice")).status_code == 200

    own_profile = await client.get(f"/api/users/{alice['user_id']}")
    assert own_profile.status_code == 200
    assert "password" not in own_profile.text
    assert (await client.get("/api/users/1")).status_code == 403

    await client.post("/api/auth/logout")
    await login(client)
    users = await client.get("/api/users/?page_size=10")
    assert users.status_code == 200
    assert users.json()["data"]["total"] == 2
    changed = await client.put(f"/api/users/{alice['user_id']}/role", json={"role": "banned"})
    assert changed.status_code == 200
    await client.post("/api/auth/logout")
    assert (await login(client, "alice")).status_code == 403


async def test_last_admin_cannot_remove_own_admin_access(client: httpx.AsyncClient) -> None:
    await login(client)
    response = await client.put("/api/users/1/role", json={"role": "banned"})
    assert response.status_code == 409
    assert response.json() == {
        "code": 409,
        "msg": "at least one admin is required",
        "data": None,
    }
    assert (await client.get("/api/users/")).status_code == 200


async def test_change_password_requires_owner_and_current_password(
    client: httpx.AsyncClient,
) -> None:
    alice = await register(client, "alice")
    bob = await register(client, "bob")
    await login(client, "alice")

    forbidden = await client.put(
        f"/api/users/{bob['user_id']}/password",
        json={"current_password": "password123", "new_password": "new-password123"},
    )
    assert forbidden.status_code == 403
    wrong_current = await client.put(
        f"/api/users/{alice['user_id']}/password",
        json={"current_password": "wrong-password", "new_password": "new-password123"},
    )
    assert wrong_current.status_code == 400
    unchanged = await client.put(
        f"/api/users/{alice['user_id']}/password",
        json={"current_password": "password123", "new_password": "password123"},
    )
    assert unchanged.status_code == 400

    changed = await client.put(
        f"/api/users/{alice['user_id']}/password",
        json={"current_password": "password123", "new_password": "new-password123"},
    )
    assert changed.status_code == 200
    await client.post("/api/auth/logout")
    assert (await login(client, "alice")).status_code == 401
    assert (await login(client, "alice", "new-password123")).status_code == 200


async def test_validation_pagination_and_permission_priority(
    client: httpx.AsyncClient,
) -> None:
    malformed = await client.post("/api/users/", json={"username": "x", "password": "123"})
    assert malformed.status_code == 400
    assert malformed.json() == {"code": 400, "msg": "invalid request body", "data": None}

    alice = await register(client, "alice")
    bob = await register(client, "bob")
    await login(client, "alice")
    missing_page_size = await client.get(f"/api/submissions/?user_id={alice['user_id']}&page=2")
    assert missing_page_size.status_code == 400

    forbidden_wins = await client.get(
        f"/api/submissions/?user_id={bob['user_id']}&status=invalid&page=2"
    )
    assert forbidden_wins.status_code == 403


async def test_problem_crud_defaults_and_language_safety(client: httpx.AsyncClient) -> None:
    await login(client)
    payload = await add_sum_problem(client)

    listing = await client.get("/api/problems/")
    assert listing.json()["data"] == [{"id": "sum_2", "title": "两数之和"}]
    detail = (await client.get("/api/problems/sum_2")).json()["data"]
    assert detail["hint"] == ""
    assert detail["tags"] == []
    assert detail["public_cases"] is False

    mismatch = dict(payload, id="other")
    assert (await client.put("/api/problems/sum_2", json=mismatch)).status_code == 400
    payload["title"] = "A+B Problem"
    updated = await client.put("/api/problems/sum_2", json=payload)
    assert updated.status_code == 200
    assert (await client.get("/api/problems/sum_2")).json()["data"]["title"] == "A+B Problem"
    assert (await client.post("/api/problems/", json=payload)).status_code == 409

    unsafe = await client.post(
        "/api/languages/",
        json={
            "name": "evil",
            "file_ext": ".x",
            "run_cmd": "python3 {src}; rm -rf /",
        },
    )
    assert unsafe.status_code == 400
    languages = await client.get("/api/languages/")
    assert languages.json()["data"]["name"] == ["cpp", "python"]

    deleted = await client.delete("/api/problems/sum_2")
    assert deleted.status_code == 200
    assert (await client.get("/api/problems/sum_2")).status_code == 404


async def test_regular_user_can_create_and_edit_problem(client: httpx.AsyncClient) -> None:
    await register(client, "author")
    await login(client, "author")
    payload = await add_sum_problem(client)
    assert (await client.get("/api/problems/sum_2")).status_code == 200

    payload["title"] = "普通用户维护的 A+B"
    updated = await client.put("/api/problems/sum_2", json=payload)
    assert updated.status_code == 200
    detail = (await client.get("/api/problems/sum_2")).json()["data"]
    assert detail["title"] == "普通用户维护的 A+B"
    assert (await client.delete("/api/problems/sum_2")).status_code == 403
