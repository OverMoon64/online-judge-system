from __future__ import annotations

import asyncio
import json

import httpx

from app import ai_service
from tests.conftest import login


def generated_payload() -> dict:
    return {
        "problem": {
            "id": "AI001",
            "title": "三数之和",
            "description": "输入三个整数并输出它们的和。",
            "input_description": "一行三个整数。",
            "output_description": "输出整数之和。",
            "samples": [{"input": "1 2 3\n", "output": "6\n"}],
            "constraints": "每个整数绝对值不超过 10^9",
            "testcases": [
                {"input": "1 2 3\n", "output": "6\n"},
                {"input": "-1 0 1\n", "output": "0\n"},
                {"input": "1000000000 1 -2\n", "output": "999999999\n"},
            ],
            "tags": ["基础", "输入输出"],
            "time_limit": 1.0,
            "memory_limit": 128,
            "difficulty": "入门",
        },
        "reference_solution": "a,b,c=map(int,input().split())\nprint(a+b+c)\n",
        "solution_explanation": "直接求和，时间复杂度 O(1)。",
    }


async def _wait_task(client: httpx.AsyncClient, task_id: str) -> dict:
    for _ in range(200):
        data = (await client.get(f"/api/ai/problem-tasks/{task_id}")).json()["data"]
        if data["status"] not in {"pending", "running"}:
            return data
        await asyncio.sleep(0.02)
    raise AssertionError("AI task did not finish")


async def test_ai_config_generation_usage_and_secret_masking(
    client: httpx.AsyncClient, monkeypatch
) -> None:
    await login(client)
    config = {
        "provider_url": "https://example-model-provider.test/v1",
        "model": "course-demo-model",
        "api_key": "top-secret-key",
        "input_price": 2.0,
        "output_price": 8.0,
        "price_unit": 1_000_000,
        "currency": "CNY",
    }
    configured = await client.put("/api/ai/model-config", json=config)
    assert configured.status_code == 200
    assert "top-secret-key" not in configured.text
    assert "top-secret-key" not in (await client.get("/api/ai/model-config")).text

    calls = 0

    async def fake_provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        content = "命题蓝图" if calls == 1 else json.dumps(generated_payload(), ensure_ascii=False)
        return content, {"input_tokens": 100, "output_tokens": 50}

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": "设计一道用于入门课程的三个整数求和题目",
            "knowledge_points": ["标准输入输出"],
            "difficulty": "入门",
            "testcase_count": 3,
        },
    )
    assert response.status_code == 200
    assert response.json()["data"]["status"] == "pending"
    task = await _wait_task(client, response.json()["data"]["task_id"])
    assert task["status"] == "completed", task
    assert task["result"]["validation"]["passed"] is True
    assert task["usage"]["input_tokens"] == 200
    assert task["usage"]["output_tokens"] == 100
    assert task["usage"]["cost"] == 0.0012


async def test_ai_task_can_be_actually_cancelled(client: httpx.AsyncClient, monkeypatch) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "slow-model",
            "api_key": "secret",
        },
    )
    started = asyncio.Event()

    async def slow_provider(*_args, **_kwargs):
        started.set()
        await asyncio.sleep(60)
        return "never returned", {"input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(ai_service, "_provider_request", slow_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={"requirement": "设计一道可以验证取消功能的简单算法题"},
    )
    task_id = response.json()["data"]["task_id"]
    await asyncio.wait_for(started.wait(), timeout=2)
    cancelled = await client.put(f"/api/ai/problem-tasks/{task_id}/cancel")
    assert cancelled.status_code == 200
    task = (await client.get(f"/api/ai/problem-tasks/{task_id}")).json()["data"]
    assert task["status"] == "cancelled"
    assert (await client.put(f"/api/ai/problem-tasks/{task_id}/cancel")).status_code == 409


async def test_ai_invalid_draft_is_repaired_and_provider_failure_is_reported(
    client: httpx.AsyncClient, monkeypatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "repair-model",
            "api_key": "secret",
        },
    )
    calls = 0

    async def repair_provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "边界条件分析", {"input_tokens": 1, "output_tokens": 1}
        if calls == 2:
            return "this is not JSON", {"input_tokens": 2, "output_tokens": 2}
        return json.dumps(generated_payload(), ensure_ascii=False), {
            "input_tokens": 3,
            "output_tokens": 3,
        }

    monkeypatch.setattr(ai_service, "_provider_request", repair_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": "设计一道经过自动修复的三个整数求和课程题目",
            "testcase_count": 3,
        },
    )
    repaired = await _wait_task(client, response.json()["data"]["task_id"])
    assert repaired["status"] == "completed"
    assert repaired["result"]["validation"]["automatic_repair_used"] is True
    assert repaired["usage"]["total_tokens"] == 12

    async def failed_provider(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(ai_service, "_provider_request", failed_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={"requirement": "设计一道能够验证服务失败状态的课程题目"},
    )
    failed = await _wait_task(client, response.json()["data"]["task_id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "provider unavailable"


async def test_ai_task_is_private_to_creator(client: httpx.AsyncClient, monkeypatch) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "private-task-model",
            "api_key": "secret",
        },
    )
    started = asyncio.Event()

    async def slow_provider(*_args, **_kwargs):
        started.set()
        await asyncio.sleep(60)
        return "never", {"input_tokens": 0, "output_tokens": 0}

    monkeypatch.setattr(ai_service, "_provider_request", slow_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={"requirement": "设计一道用于验证任务私有性的课程算法题"},
    )
    task_id = response.json()["data"]["task_id"]
    await asyncio.wait_for(started.wait(), timeout=2)
    await client.post("/api/users/", json={"username": "viewer", "password": "password123"})
    await client.post("/api/auth/logout")
    await login(client, "viewer")
    assert (await client.get(f"/api/ai/problem-tasks/{task_id}")).status_code == 403
    assert (await client.put(f"/api/ai/problem-tasks/{task_id}/cancel")).status_code == 403

    await client.post("/api/auth/logout")
    await login(client)
    assert (await client.put(f"/api/ai/problem-tasks/{task_id}/cancel")).status_code == 200
