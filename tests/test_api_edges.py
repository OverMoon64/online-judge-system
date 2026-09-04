from __future__ import annotations

import httpx

from app.judge import wait_for_submission
from tests.conftest import add_sum_problem, login, register


async def test_misc_routes_admin_creation_and_permission_edges(
    client: httpx.AsyncClient,
) -> None:
    assert (await client.get("/")).json()["data"]["service"] == "Async Online Judge"
    assert (await client.get("/health")).json()["data"] == {"status": "ok"}
    assert (await client.post("/api/auth/logout")).status_code == 401
    assert (await client.post("/api/users/admin", content=b"bad")).status_code == 401

    await login(client)
    created = await client.post(
        "/api/users/admin", json={"username": "operator", "password": "operator-pass"}
    )
    assert created.status_code == 200
    assert (
        await client.post(
            "/api/users/admin", json={"username": "operator", "password": "operator-pass"}
        )
    ).status_code == 400
    assert (await client.get("/api/users/?page_size=1")).json()["data"]["total"] == 2
    assert (await client.get("/api/users/99999")).status_code == 404
    assert (await client.put("/api/users/99999/role", json={"role": "user"})).status_code == 404
    assert (await client.put("/api/users/1/role", json={"role": "invalid"})).status_code == 400

    await add_sum_problem(client)
    alice = await register(client, "alice")
    await client.post("/api/auth/logout")
    await login(client, "alice")
    assert (await client.get("/api/users/not-a-number")).status_code == 400
    assert (await client.get("/api/users/1")).status_code == 403
    assert (await client.delete("/api/problems/sum_2")).status_code == 403
    assert (
        await client.put("/api/problems/sum_2/log_visibility", json={"public_cases": True})
    ).status_code == 403
    assert (await client.post("/api/reset/")).status_code == 403
    assert alice["role"] == "user"


async def test_problem_language_and_submission_error_paths(client: httpx.AsyncClient) -> None:
    await login(client)
    payload = await add_sum_problem(client)

    assert (await client.delete("/api/problems/missing")).status_code == 404
    assert (
        await client.put("/api/problems/missing/log_visibility", json={"public_cases": True})
    ).status_code == 404
    assert (await client.put("/api/problems/sum_2/log_visibility", json={})).json()["data"][
        "public_cases"
    ] is False
    invalid_payload = dict(payload)
    invalid_payload["time_limit"] = 0
    assert (await client.post("/api/problems/", json=invalid_payload)).status_code == 400

    language = {
        "name": "python-copy",
        "file_ext": ".pyx",
        "run_cmd": "python3 {src}",
        "time_limit": 2,
        "memory_limit": 64,
    }
    assert (await client.post("/api/languages/", json=language)).status_code == 200
    assert (await client.post("/api/languages/", json=language)).status_code == 409
    assert (
        await client.post(
            "/api/languages/",
            json={"name": "bad", "file_ext": "bad", "run_cmd": "python3 {src}"},
        )
    ).status_code == 400

    assert (
        await client.post(
            "/api/submissions/",
            json={"problem_id": "missing", "language": "python", "code": "print(1)"},
        )
    ).status_code == 404
    assert (
        await client.post(
            "/api/submissions/",
            json={"problem_id": "sum_2", "language": "missing", "code": "print(1)"},
        )
    ).status_code == 404
    assert (await client.get("/api/submissions/")).status_code == 400
    assert (await client.get("/api/submissions/?user_id=bad")).status_code == 400
    assert (await client.get("/api/submissions/?user_id=99999")).status_code == 404
    assert (
        await client.get("/api/submissions/?problem_id=sum_2&status=invalid")
    ).status_code == 400
    assert (await client.get("/api/submissions/?problem_id=missing")).status_code == 404
    assert (await client.get("/api/submissions/not-a-number")).status_code == 400
    assert (await client.get("/api/submissions/99999")).status_code == 404
    assert (await client.put("/api/submissions/99999/rejudge")).status_code == 404
    assert (await client.get("/api/submissions/99999/log")).status_code == 404


async def test_problem_only_submission_filter_and_audit_pagination(
    client: httpx.AsyncClient,
) -> None:
    await login(client)
    await add_sum_problem(client)
    response = await client.post(
        "/api/submissions/",
        json={
            "problem_id": "sum_2",
            "language": "python",
            "code": "a,b=map(int,input().split());print(a+b)",
        },
    )
    submission_id = response.json()["data"]["submission_id"]
    pending = await client.get(f"/api/submissions/{submission_id}")
    assert set(pending.json()["data"]) >= {"submission_id", "status"}
    assert pending.json()["data"]["result"] == "pending"
    await wait_for_submission(int(submission_id))
    listing = await client.get("/api/submissions/?problem_id=sum_2&page=1&page_size=10")
    assert listing.json()["data"]["total"] == 1
    filtered = await client.get("/api/submissions/?problem_id=sum_2&status=success&page_size=5")
    assert filtered.json()["data"]["submissions"][0]["score"] == 30

    await client.get(f"/api/submissions/{submission_id}/log")
    logs = await client.get("/api/logs/access/?problem_id=sum_2&user_id=1&page=1&page_size=1")
    assert len(logs.json()["data"]) == 1


async def test_ai_configuration_and_task_route_errors(client: httpx.AsyncClient) -> None:
    await login(client)
    assert (await client.get("/api/ai/model-config")).status_code == 404
    local = await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "http://127.0.0.1:9000/v1",
            "model": "local",
            "api_key": "secret",
        },
    )
    assert local.status_code == 400
    assert (
        await client.post(
            "/api/ai/problem-tasks/",
            json={"requirement": "设计一道足够明确的课程算法题目"},
        )
    ).status_code == 400
    assert (
        await client.post(
            "/api/ai/problem-tasks/",
            json={
                "requirement": "修改一份当前并不存在的课程算法题目",
                "problem_id": "missing",
            },
        )
    ).status_code == 404
    assert (await client.get("/api/ai/problem-tasks/missing")).status_code == 404
    assert (await client.put("/api/ai/problem-tasks/missing/cancel")).status_code == 404
