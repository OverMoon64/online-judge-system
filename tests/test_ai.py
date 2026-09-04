from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app import ai_service
from app.schemas import AIModelConfigPayload, AIProblemTaskPayload
from tests.conftest import login


def test_model_config_name_rejects_path_separators():
    with pytest.raises(ValidationError):
        AIModelConfigPayload(
            name="primary/model",
            provider_url="https://example.test/v1",
            model="fake-model",
            api_key="secret",
        )


def test_problem_task_uses_automatic_testcase_strategy_by_default():
    payload = AIProblemTaskPayload(requirement="设计一道用于课程验收的算法题目")
    assert payload.testcase_count is None


async def test_provider_failure_retries_twice(monkeypatch):
    attempts = 0

    async def flaky_provider(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary failure")
        return "ok", {"input_tokens": 1, "output_tokens": 1}

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(ai_service, "_provider_request", flaky_provider)
    monkeypatch.setattr(ai_service.asyncio, "sleep", no_delay)
    config = AIModelConfigPayload(
        name="retry-model",
        provider_url="https://example.test/v1",
        model="fake-model",
        api_key="secret",
    )

    result = await ai_service._provider_request_with_retry(config, [])

    assert result[0] == "ok"
    assert result[1]["request_count"] == 3
    assert result[1]["estimated"] is True
    assert attempts == 3


async def test_known_model_pricing_and_manual_fallback(client: httpx.AsyncClient) -> None:
    assert (
        await client.get(
            "/api/ai/model-pricing",
            params={
                "provider_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen3.7-plus",
            },
        )
    ).status_code == 401
    await login(client)
    known = await client.get(
        "/api/ai/model-pricing",
        params={
            "provider_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen3.7-plus",
        },
    )
    assert known.status_code == 200
    assert known.json()["data"]["input_price"] == 1.6
    assert known.json()["data"]["output_price"] == 6.4
    assert known.json()["data"]["currency"] == "CNY"
    unknown = await client.get(
        "/api/ai/model-pricing",
        params={"provider_url": "https://example.test/v1", "model": "custom-model"},
    )
    assert unknown.status_code == 404


async def test_stream_preview_callback_updates_running_task(monkeypatch):
    updates: list[dict] = []

    async def capture_update(task_id, **kwargs):
        updates.append({"task_id": task_id, **kwargs})

    monkeypatch.setattr(ai_service, "_update_task", capture_update)
    callback = ai_service._stream_preview_callback("task-stream", "题面")

    await callback("正在流式生成题面")

    assert updates == [
        {
            "task_id": "task-stream",
            "result": {
                "stream_stage": "题面",
                "stream_preview": "正在流式生成题面",
            },
        }
    ]


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
        "reference_solution_cpp": (
            "#include <iostream>\nusing namespace std;\n"
            "int main(){long long a,b,c;cin>>a>>b>>c;cout<<a+b+c;}\n"
        ),
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
        "name": "primary",
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
    backup = await client.put(
        "/api/ai/model-config",
        json={
            **config,
            "name": "backup",
            "model": "backup-model",
            "api_key": "backup-secret-key",
        },
    )
    assert backup.status_code == 200
    listed = await client.get("/api/ai/model-configs/")
    assert listed.status_code == 200
    assert listed.json()["data"]["active"] == "backup"
    assert {item["name"] for item in listed.json()["data"]["models"]} == {
        "primary",
        "backup",
    }
    assert "top-secret-key" not in listed.text
    assert "backup-secret-key" not in listed.text

    calls = 0
    used_models: set[str] = set()

    async def fake_provider(provider_config, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        used_models.add(provider_config.model)
        content = "命题蓝图" if calls == 1 else json.dumps(generated_payload(), ensure_ascii=False)
        return content, {"input_tokens": 100, "output_tokens": 50}

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": "设计一道用于入门课程的三个整数求和题目",
            "model_config_name": "primary",
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
    assert task["result"]["validation"]["reference_languages"] == ["python", "cpp"]
    assert task["usage"]["input_tokens"] == 200
    assert task["usage"]["output_tokens"] == 100
    assert task["usage"]["cost"] == 0.0012
    assert task["usage"]["model_config_name"] == "primary"
    assert task["usage"]["request_count"] == 2
    assert [item["stage"] for item in task["usage"]["calls"]] == [
        "需求分析",
        "题面、解法与测试点",
    ]
    assert task["result"]["problem"]["author"] == "course-demo-model"
    assert used_models == {"course-demo-model"}
    listed = await client.get("/api/ai/model-configs/")
    primary = next(item for item in listed.json()["data"]["models"] if item["name"] == "primary")
    assert primary["usage"]["input_tokens"] == 200
    assert primary["usage"]["output_tokens"] == 100
    assert primary["usage"]["cost"] == 0.0012


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
        if calls == 3:
            return json.dumps({"problem": generated_payload()["problem"]}, ensure_ascii=False), {
                "input_tokens": 3,
                "output_tokens": 3,
            }
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
    assert repaired["result"]["validation"]["automatic_repair_count"] == 2
    assert repaired["usage"]["total_tokens"] == 18

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
