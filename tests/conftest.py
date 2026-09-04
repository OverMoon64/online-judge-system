from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest_asyncio

from app import ai_service
from app.db import configure_database, initialize_database
from app.judge import cancel_all_submission_tasks
from app.main import app


@pytest_asyncio.fixture
async def client(tmp_path) -> AsyncIterator[httpx.AsyncClient]:
    await cancel_all_submission_tasks()
    await ai_service.cancel_all_ai_tasks(clear_configs=True)
    database_file = tmp_path / "test.db"
    await configure_database(f"sqlite+aiosqlite:///{database_file}")
    await initialize_database()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver", follow_redirects=True
    ) as test_client:
        yield test_client
    await cancel_all_submission_tasks()
    await ai_service.cancel_all_ai_tasks(clear_configs=True)


async def login(
    client: httpx.AsyncClient, username: str = "admin", password: str | None = None
) -> httpx.Response:
    if password is None:
        password = "admintestpassword" if username == "admin" else "password123"
    return await client.post("/api/auth/login", json={"username": username, "password": password})


async def register(client: httpx.AsyncClient, username: str, password: str = "password123") -> dict:
    response = await client.post("/api/users/", json={"username": username, "password": password})
    assert response.status_code == 200
    return response.json()["data"]


async def add_sum_problem(
    client: httpx.AsyncClient,
    *,
    problem_id: str = "sum_2",
    time_limit: float = 1.0,
    memory_limit: int = 128,
) -> dict:
    payload = {
        "id": problem_id,
        "title": "两数之和",
        "description": "输入两个整数，输出它们的和。",
        "input_description": "一行两个整数。",
        "output_description": "输出两数之和。",
        "samples": [{"input": "1 2\n", "output": "3\n"}],
        "constraints": "|a|, |b| <= 10^9",
        "testcases": [
            {"input": "1 2\n", "output": "3\n"},
            {"input": "-5 8\n", "output": "3\n"},
            {"input": "1000000000 -1\n", "output": "999999999\n"},
        ],
        "time_limit": time_limit,
        "memory_limit": memory_limit,
    }
    response = await client.post("/api/problems/", json=payload)
    assert response.status_code == 200, response.text
    return payload
