from __future__ import annotations

import shutil

import httpx
import pytest

from app.judge import wait_for_submission
from tests.conftest import add_sum_problem, login, register


async def _submit(client: httpx.AsyncClient, code: str, language: str = "python") -> str:
    response = await client.post(
        "/api/submissions/",
        json={"problem_id": "sum_2", "language": language, "code": code},
    )
    assert response.status_code == 200, response.text
    assert response.json()["data"]["status"] == "pending"
    return response.json()["data"]["submission_id"]


async def test_python_judging_logs_rejudge_and_stats(client: httpx.AsyncClient) -> None:
    await login(client)
    await add_sum_problem(client)
    await client.post("/api/auth/logout")
    alice = await register(client, "alice")
    await login(client, "alice")

    submission_id = await _submit(client, "a, b = map(int, input().split())\nprint(a + b, '   ')\n")
    await wait_for_submission(int(submission_id))
    result = (await client.get(f"/api/submissions/{submission_id}")).json()["data"]
    assert result["status"] == "success"
    assert result["score"] == result["counts"] == 30
    assert result["compile_info"] is None

    log = (await client.get(f"/api/submissions/{submission_id}/log")).json()["data"]
    assert [case["result"] for case in log["details"]] == ["AC", "AC", "AC"]

    profile = (await client.get(f"/api/users/{alice['user_id']}")).json()["data"]
    assert profile["submit_count"] == 1
    assert profile["resolve_count"] == 1
    listing = await client.get(f"/api/submissions/?user_id={alice['user_id']}")
    assert listing.json()["data"]["submissions"][0]["score"] == 30

    await client.post("/api/auth/logout")
    await login(client)
    rejudge = await client.put(f"/api/submissions/{submission_id}/rejudge")
    assert rejudge.status_code == 200
    assert rejudge.json()["data"] == {"submission_id": submission_id, "status": "pending"}
    await wait_for_submission(int(submission_id))
    assert (await client.get(f"/api/submissions/{submission_id}")).json()["data"]["score"] == 30


async def test_wrong_answer_runtime_error_timeout_and_rate_limit(
    client: httpx.AsyncClient,
) -> None:
    await login(client)
    await add_sum_problem(client, time_limit=0.5)
    await client.post("/api/auth/logout")
    await register(client, "alice")
    await login(client, "alice")

    ids = [
        await _submit(client, "print(0)"),
        await _submit(client, "raise RuntimeError('boom')"),
        await _submit(client, "while True:\n    pass"),
    ]
    limited = await client.post(
        "/api/submissions/",
        json={"problem_id": "missing", "language": "missing", "code": "print(1)"},
    )
    assert limited.status_code == 429
    for submission_id in ids:
        await wait_for_submission(int(submission_id))
    outcomes = []
    for submission_id in ids:
        log = (await client.get(f"/api/submissions/{submission_id}/log")).json()["data"]
        outcomes.append(log["details"][0]["result"])
    assert outcomes == ["WA", "RE", "TLE"]


@pytest.mark.integration
@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is not installed")
async def test_cpp_compilation_success_and_failure(client: httpx.AsyncClient) -> None:
    await login(client)
    await add_sum_problem(client)
    success_id = await _submit(
        client,
        "#include <iostream>\nint main(){long long a,b;std::cin>>a>>b;std::cout<<a+b;}\n",
        "cpp",
    )
    failure_id = await _submit(client, "int main( {", "cpp")
    await wait_for_submission(int(success_id), timeout=30)
    await wait_for_submission(int(failure_id), timeout=30)
    success_result = (await client.get(f"/api/submissions/{success_id}")).json()["data"]
    assert success_result["score"] == 30
    assert success_result["compile_info"]["result"] == "success"
    failure_log = (await client.get(f"/api/submissions/{failure_id}/log")).json()["data"]
    assert {case["result"] for case in failure_log["details"]} == {"CE"}


async def test_private_and_public_log_access_is_audited(client: httpx.AsyncClient) -> None:
    await login(client)
    await add_sum_problem(client)
    owner = await register(client, "owner")
    viewer = await register(client, "viewer")
    await client.post("/api/auth/logout")
    await login(client, "owner")
    submission_id = await _submit(client, "a,b=map(int,input().split());print(a+b)")
    await wait_for_submission(int(submission_id))
    own_private_log = await client.get(f"/api/submissions/{submission_id}/log")
    assert own_private_log.status_code == 200
    await client.post("/api/auth/logout")
    await login(client, "viewer")
    assert (await client.get(f"/api/submissions/{submission_id}")).status_code == 403
    assert (await client.get(f"/api/submissions/{submission_id}/log")).status_code == 403

    await client.post("/api/auth/logout")
    await login(client)
    visibility = await client.put("/api/problems/sum_2/log_visibility", json={"public_cases": True})
    assert visibility.status_code == 200
    await client.post("/api/auth/logout")
    await login(client, "viewer")
    assert (await client.get(f"/api/submissions/{submission_id}/log")).status_code == 200

    await client.post("/api/auth/logout")
    await login(client)
    audit = (await client.get(f"/api/logs/access/?user_id={viewer['user_id']}")).json()["data"]
    assert [entry["status"] for entry in audit] == ["200", "403"]
    assert all(entry["action"] == "view_logs" for entry in audit)
    assert owner["user_id"] != viewer["user_id"]


async def test_memory_limit_and_reset(client: httpx.AsyncClient) -> None:
    await login(client)
    await add_sum_problem(client, memory_limit=20, time_limit=2.0)
    submission_id = await _submit(
        client, "payload = bytearray(200 * 1024 * 1024)\nprint(len(payload))"
    )
    await wait_for_submission(int(submission_id), timeout=10)
    log = (await client.get(f"/api/submissions/{submission_id}/log")).json()["data"]
    assert log["details"][0]["result"] in {"MLE", "RE"}

    reset = await client.post("/api/reset/")
    assert reset.status_code == 200
    assert (await client.get("/api/problems/")).status_code == 401
    assert (await login(client)).status_code == 200
    assert (await client.get("/api/problems/")).json()["data"] == []
