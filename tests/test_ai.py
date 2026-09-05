from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app import ai_service, judge
from app.schemas import AIModelConfigPayload, AIProblemTaskPayload, GeneratedProblem, ProblemPayload
from app.stress_cases import infer_stress_candidate
from app.testcase_files import load_file_testcases
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
    assert payload.difficulty == "自动"
    assert (
        AIProblemTaskPayload(
            requirement="根据题意自动选择一道合适难度的算法题",
            difficulty="自动",
        ).difficulty
        == "自动"
    )
    with pytest.raises(ValidationError):
        AIProblemTaskPayload(
            requirement="无效难度应该在请求阶段直接被拒绝",
            difficulty="未知",
        )


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


def fibonacci_payload() -> dict:
    return {
        "problem": {
            "id": "FIB_MOD",
            "title": "斐波那契数列取模",
            "description": (
                "定义 F(0)=0、F(1)=1，且 F(n)=F(n-1)+F(n-2)。求 F(N) 对 1000000007 取模的结果。"
            ),
            "input_description": "输入一个非负整数 N。",
            "output_description": "输出 F(N) mod 1000000007。",
            "samples": [
                {"input": "10\n", "output": "55   \n"},
                {"input": "100\n", "output": "123456789\n"},
            ],
            "constraints": "0 <= N <= 100000",
            "testcases": [
                {"input": "100\n", "output": "987654321"},
                {"input": "1000\n", "output": "111111111"},
                {"input": "100000\n", "output": ""},
            ],
            "tags": ["递推", "取模"],
            "time_limit": 1.0,
            "memory_limit": 128,
            "difficulty": "入门",
        },
        "reference_solution": (
            "MOD=1000000007\n"
            "n=int(input())\n"
            "a,b=0,1\n"
            "for _ in range(n):\n"
            "    a,b=b,(a+b)%MOD\n"
            "print(a)\n"
        ),
        "reference_solution_cpp": (
            "#include <iostream>\nusing namespace std;\n"
            "int main(){const long long MOD=1000000007; int n; cin>>n; "
            "long long a=0,b=1; while(n--){long long c=(a+b)%MOD;a=b;b=c;} "
            "cout<<a;}\n"
        ),
        "solution_explanation": "从 F(0) 开始递推并在每一步取模。",
    }


async def _wait_task(client: httpx.AsyncClient, task_id: str) -> dict:
    last_response = ""
    for _ in range(200):
        response = await client.get(f"/api/ai/problem-tasks/{task_id}")
        last_response = f"{response.status_code}: {response.text[:500]}"
        if response.status_code != 200:
            await asyncio.sleep(0.02)
            continue
        data = response.json()["data"]
        if data["status"] not in {"pending", "running"}:
            return data
        await asyncio.sleep(0.02)
    raise AssertionError(f"AI task did not finish; last response: {last_response}")


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
    assert task["result"]["validation"]["file_stress"] == {
        "requested": False,
        "applied": False,
        "count": 0,
    }
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


async def test_ai_calibrates_fibonacci_outputs_with_two_model_calls(
    client: httpx.AsyncClient, monkeypatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "consensus-model",
            "api_key": "secret",
        },
    )
    calls = 0

    async def fake_provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        content = (
            "递推与取模边界分析"
            if calls == 1
            else json.dumps(fibonacci_payload(), ensure_ascii=False)
        )
        return content, {"input_tokens": 10, "output_tokens": 20}

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": (
                "求斐波那契数列的第 N 项，结果对 1000000007 取模，适合练习递推与取模。"
            ),
            "knowledge_points": ["递推", "取模"],
            "testcase_count": 3,
        },
    )
    task = await _wait_task(client, response.json()["data"]["task_id"])

    assert task["status"] == "completed", task
    assert calls == 2
    assert task["usage"]["request_count"] == 2
    assert task["result"]["validation"]["automatic_repair_count"] == 0
    calibration = task["result"]["validation"]["output_calibration"]
    assert calibration["applied"] is True
    assert calibration["count"] == 4
    assert {item["kind"] for item in calibration["items"]} == {"sample", "testcase"}
    assert task["result"]["problem"]["samples"][0]["output"] == "55   \n"
    assert task["result"]["problem"]["samples"][1]["output"] == "687995182"
    assert [case["output"] for case in task["result"]["problem"]["testcases"]] == [
        "687995182",
        "517691607",
        "911435502",
    ]


async def test_ai_refuses_consensus_without_anchor_and_reports_disagreement() -> None:
    request = AIProblemTaskPayload(requirement="设计一道用于测试双语言安全校准的题目")
    no_anchor_data = fibonacci_payload()
    for case in [
        *no_anchor_data["problem"]["samples"],
        *no_anchor_data["problem"]["testcases"],
    ]:
        case["output"] = "definitely-wrong"
    no_anchor = GeneratedProblem.model_validate(no_anchor_data)
    errors, calibration = await ai_service._validate_generated(no_anchor, request, None)
    assert any("不存在与双解结果一致的非空安全锚点" in error for error in errors)
    assert calibration == {"applied": False, "count": 0, "items": []}

    disagreement_data = fibonacci_payload()
    disagreement_data["reference_solution_cpp"] = (
        "#include <iostream>\nusing namespace std;\nint main(){long long n;cin>>n;cout<<n+42;}\n"
    )
    for case in [
        *disagreement_data["problem"]["samples"],
        *disagreement_data["problem"]["testcases"],
    ]:
        case["output"] = "not-an-answer"
    disagreement = GeneratedProblem.model_validate(disagreement_data)
    errors, calibration = await ai_service._validate_generated(disagreement, request, None)
    assert any("双解输出不一致" in error for error in errors)
    assert any("Python stdout=" in error and "C++ stdout=" in error for error in errors)
    assert calibration["applied"] is False


def test_ai_resource_limits_follow_explicit_request_and_policy() -> None:
    explicit_request = AIProblemTaskPayload(
        requirement="设计一道图搜索题，限制时间为 2.5 秒，空间限制 512 MB",
        difficulty="困难",
        knowledge_points=["BFS"],
    )
    explicit_generated = GeneratedProblem.model_validate(generated_payload())
    explicit = ai_service._apply_resource_limit_policy(explicit_generated, explicit_request)
    assert explicit["source"] == "explicit"
    assert explicit["final"] == {"time_limit": 2.5, "memory_limit": 512}
    assert explicit_generated.problem.time_limit == 2.5
    assert explicit_generated.problem.memory_limit == 512
    explicit_prompt = ai_service._generation_prompt(explicit_request, "分析", None)
    assert '"time_limit": 2.5, "memory_limit": 512' in explicit_prompt
    assert "必须严格为 2.5 秒" in explicit_prompt

    automatic_request = AIProblemTaskPayload(
        requirement="生成一道大规模网格最短路题，使用 BFS 搜索",
        difficulty="困难",
        knowledge_points=["图", "BFS"],
    )
    automatic_generated = GeneratedProblem.model_validate(generated_payload())
    automatic = ai_service._apply_resource_limit_policy(automatic_generated, automatic_request)
    assert automatic["source"] == "automatic"
    assert automatic["automatic"] == {"time_limit": 4.0, "memory_limit": 512}
    assert automatic["final"] == {"time_limit": 4.0, "memory_limit": 512}
    assert automatic_generated.problem.time_limit == 4.0
    assert automatic_generated.problem.memory_limit == 512


def test_auto_difficulty_controls_resource_limits_and_testcase_count() -> None:
    request = AIProblemTaskPayload(
        requirement="根据题目内容自动决定难度并生成完整测试",
        difficulty="自动",
    )
    generated = GeneratedProblem.model_validate(generated_payload())
    generated.problem.difficulty = "困难"
    generated.problem.time_limit = 1.0
    generated.problem.memory_limit = 128
    generated.problem.testcases.extend(
        [
            generated.problem.testcases[0].model_copy(update={"input": "0 0 0\n", "output": "0\n"}),
            generated.problem.testcases[0].model_copy(
                update={"input": "9 9 9\n", "output": "27\n"}
            ),
        ]
    )

    difficulty = ai_service._apply_difficulty_policy(generated, request)
    resources = ai_service._apply_resource_limit_policy(generated, request)
    errors = ai_service._static_validation_errors(generated, request, None)

    assert difficulty == {"source": "model", "model": "困难", "final": "困难"}
    assert resources["final"] == {"time_limit": 3.0, "memory_limit": 512}
    assert ai_service._target_testcase_count(request, generated) == 5
    assert errors == []

    generated.problem.difficulty = "不规范"
    fallback = ai_service._apply_difficulty_policy(generated, request)
    assert fallback["source"] == "fallback"
    assert fallback["final"] == "中等"


async def test_ai_auto_difficulty_completes_without_stress_generator(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "stable-model",
            "api_key": "secret",
        },
    )
    payload = generated_payload()
    payload["problem"]["difficulty"] = "困难"
    payload["problem"]["testcases"].extend(
        [
            {"input": "0 0 0\n", "output": "0\n"},
            {"input": "9 9 9\n", "output": "27\n"},
        ]
    )
    calls = 0

    async def fake_provider(*_args, **kwargs):
        nonlocal calls
        calls += 1
        if not kwargs.get("json_mode"):
            return "自动判定为困难题", {"input_tokens": 5, "output_tokens": 5}
        return json.dumps(payload), {"input_tokens": 10, "output_tokens": 20}

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": "根据题意自动决定难度并生成稳定可用的完整测试点",
            "difficulty": "自动",
        },
    )
    task = await _wait_task(client, response.json()["data"]["task_id"])

    assert task["status"] == "completed", task
    assert calls == 2
    problem = task["result"]["problem"]
    validation = task["result"]["validation"]
    assert problem["difficulty"] == "困难"
    assert len(problem["testcases"]) == 5
    assert problem["time_limit"] == 3.0
    assert problem["memory_limit"] == 512
    assert validation["difficulty"]["source"] == "model"
    assert validation["resource_limits"]["final"] == {
        "time_limit": 3.0,
        "memory_limit": 512,
    }
    assert "stress_test_generator" not in task["result"]


async def test_ai_assigns_sequential_ids(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "id-model",
            "api_key": "secret",
        },
    )

    async def fake_provider(*_args, **kwargs):
        if not kwargs.get("json_mode"):
            return "基础题分析", {"input_tokens": 1, "output_tokens": 1}
        return json.dumps(generated_payload()), {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    ids = []
    for suffix in ("第一题", "第二题"):
        response = await client.post(
            "/api/ai/problem-tasks/",
            json={"requirement": f"设计一道用于验证顺序编号的三数求和{suffix}"},
        )
        task = await _wait_task(client, response.json()["data"]["task_id"])
        assert task["status"] == "completed", task
        ids.append(task["result"]["problem"]["id"])
    assert ids == ["AI0001", "AI0002"]


async def test_ai_repairs_placeholder_test_data_instead_of_reference_solution(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "test-data-repair-model",
            "api_key": "secret",
        },
    )
    valid = generated_payload()
    invalid = generated_payload()
    invalid["problem"]["testcases"][1]["input"] = "1000 1000\nGENERATE_FULL_ZERO_GRID"
    calls = 0

    async def fake_provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            content = "边界与资源分析"
        elif calls == 2:
            content = json.dumps(invalid, ensure_ascii=False)
        else:
            content = json.dumps(
                {
                    "samples": valid["problem"]["samples"],
                    "testcases": valid["problem"]["testcases"],
                },
                ensure_ascii=False,
            )
        return content, {"input_tokens": 10, "output_tokens": 20}

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": "生成一道包含完整可执行测试输入的三个整数求和题目",
            "testcase_count": 3,
        },
    )
    task = await _wait_task(client, response.json()["data"]["task_id"])

    assert task["status"] == "completed", task
    assert calls == 3
    assert task["usage"]["calls"][-1]["stage"] == "定向修复测试数据 1/2"
    assert task["result"]["reference_solution"] == valid["reference_solution"]
    assert task["result"]["reference_solution_cpp"] == valid["reference_solution_cpp"]
    assert all("GENERATE_" not in case["input"] for case in task["result"]["problem"]["testcases"])
    assert task["result"]["validation"]["resource_limits"]["source"] == "automatic"


@pytest.mark.parametrize("failure_status", ["RE", "TLE", "MLE"])
async def test_ai_does_not_calibrate_when_one_reference_cannot_run(
    failure_status: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    generated = GeneratedProblem.model_validate(generated_payload())
    request = AIProblemTaskPayload(requirement="设计一道用于测试参考程序失败状态的题目")

    async def fake_execution(
        _problem, _solution, language="python", *, inputs=None
    ) -> judge.ReferenceExecution:
        assert inputs is not None
        if language == "python":
            results = [
                judge.ProcessResult(failure_status, 1, "partial", "failed", 0.01, 1.0)
                for _ in inputs
            ]
        else:
            expected = [
                case.output for case in [*generated.problem.samples, *generated.problem.testcases]
            ]
            results = [judge.ProcessResult("OK", 0, output, "", 0.01, 1.0) for output in expected]
        return judge.ReferenceExecution(language, None, results)

    monkeypatch.setattr(ai_service, "execute_reference_solution", fake_execution)
    errors, calibration = await ai_service._validate_generated(generated, request, None)
    assert errors and all(error.startswith("Python ") for error in errors)
    assert all(f": {failure_status};" in error for error in errors)
    assert calibration["applied"] is False


async def test_ai_does_not_calibrate_compile_failure_or_oversized_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = GeneratedProblem.model_validate(generated_payload())
    request = AIProblemTaskPayload(requirement="设计一道用于测试编译和输出上限的题目")

    async def compile_failure(
        _problem, _solution, language="python", *, inputs=None
    ) -> judge.ReferenceExecution:
        assert inputs is not None
        if language == "cpp":
            return judge.ReferenceExecution(language, "compile: invalid program", [])
        expected = [
            case.output for case in [*generated.problem.samples, *generated.problem.testcases]
        ]
        return judge.ReferenceExecution(
            language,
            None,
            [judge.ProcessResult("OK", 0, output, "", 0.01, 1.0) for output in expected],
        )

    monkeypatch.setattr(ai_service, "execute_reference_solution", compile_failure)
    errors, calibration = await ai_service._validate_generated(generated, request, None)
    assert errors == ["C++ compile: invalid program"]
    assert calibration["applied"] is False

    async def oversized_consensus(
        _problem, _solution, language="python", *, inputs=None
    ) -> judge.ReferenceExecution:
        assert inputs is not None
        expected = [
            case.output for case in [*generated.problem.samples, *generated.problem.testcases]
        ]
        expected[-1] = "x" * 4_001
        return judge.ReferenceExecution(
            language,
            None,
            [judge.ProcessResult("OK", 0, output, "", 0.01, 1.0) for output in expected],
        )

    monkeypatch.setattr(ai_service, "execute_reference_solution", oversized_consensus)
    errors, calibration = await ai_service._validate_generated(generated, request, None)
    assert any("双解输出超过 4000 字符" in error for error in errors)
    assert calibration["applied"] is False


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


async def test_ai_repairs_only_the_failing_reference_language(
    client: httpx.AsyncClient, monkeypatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "targeted-repair-model",
            "api_key": "secret",
        },
    )
    correct_cpp = generated_payload()["reference_solution_cpp"]
    calls = 0

    async def targeted_provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "边界条件分析", {"input_tokens": 1, "output_tokens": 1}
        if calls == 2:
            draft = generated_payload()
            draft["reference_solution_cpp"] = "int main(){return 0;}"
            return json.dumps(draft, ensure_ascii=False), {
                "input_tokens": 2,
                "output_tokens": 2,
            }
        return json.dumps({"reference_solution_cpp": correct_cpp}, ensure_ascii=False), {
            "input_tokens": 3,
            "output_tokens": 3,
        }

    monkeypatch.setattr(ai_service, "_provider_request", targeted_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": "设计一道能验证 C++ 定向修复的三数求和题",
            "testcase_count": 3,
        },
    )
    repaired = await _wait_task(client, response.json()["data"]["task_id"])

    assert repaired["status"] == "completed", repaired
    assert repaired["result"]["reference_solution_cpp"] == correct_cpp
    assert repaired["result"]["validation"]["automatic_repair_count"] == 1
    assert repaired["usage"]["calls"][-1]["stage"] == "定向修复 C++ 1/2"
    assert calls == 3


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


def stress_array_payload() -> dict:
    payload = generated_payload()
    payload["problem"].update(
        {
            "title": "大规模数组求和",
            "description": "输入 n 个整数，输出所有整数之和。",
            "input_description": "第一行 n，第二行 n 个整数。",
            "constraints": "1 <= n <= 50000，每个整数绝对值不超过 10^9。",
            "samples": [{"input": "3\n1 2 3\n", "output": "6\n"}],
            "testcases": [
                {"input": "1\n5\n", "output": "5\n"},
                {"input": "4\n1 2 3 4\n", "output": "10\n"},
                {"input": "5\n-2 -1 0 1 2\n", "output": "0\n"},
            ],
            "tags": ["数组", "复杂度"],
            "difficulty": "困难",
        }
    )
    payload["reference_solution"] = (
        "import sys\n"
        "data=list(map(int,sys.stdin.buffer.read().split()))\n"
        "print(sum(data[1:1+data[0]]))\n"
    )
    payload["reference_solution_cpp"] = (
        "#include <iostream>\nusing namespace std;\n"
        "int main(){ios::sync_with_stdio(false);cin.tie(nullptr);"
        "int n;if(!(cin>>n))return 0;long long s=0,x;"
        "while(n--&&cin>>x)s+=x;cout<<s;}\n"
    )
    payload["solution_explanation"] = "线性扫描数组，时间复杂度 O(n)，空间复杂度 O(1)。"
    return payload


@pytest.mark.parametrize(
    ("title", "sample_input", "constraints", "tags", "expected_template"),
    [
        (
            "字符串回文统计",
            "abba\n",
            "1 <= |S| <= 10000",
            ["字符串", "回文"],
            "single_string",
        ),
        (
            "网格路径",
            "2 2\n0 1\n1 0\n",
            "1 <= n <= 100，1 <= m <= 100",
            ["网格", "动态规划"],
            "integer_grid",
        ),
        (
            "图的连通性",
            "3 2\n1 2\n2 3\n",
            "1 <= n <= 5000，0 <= m <= 10000",
            ["图", "并查集"],
            "graph_chain",
        ),
    ],
)
def test_file_stress_templates_are_inferred_without_model_code(
    title: str,
    sample_input: str,
    constraints: str,
    tags: list[str],
    expected_template: str,
) -> None:
    payload = generated_payload()
    payload["problem"].update(
        {
            "title": title,
            "description": title,
            "input_description": title,
            "constraints": constraints,
            "samples": [{"input": sample_input, "output": "0\n"}],
            "tags": tags,
        }
    )
    generated = GeneratedProblem.model_validate(payload)
    request = AIProblemTaskPayload(requirement="使用大数据压力测试严格验证时间复杂度")

    candidate = infer_stress_candidate(generated, request)

    assert candidate is not None
    assert candidate.template == expected_template
    assert len(candidate.input) > len(sample_input)


def test_fibonacci_file_stress_uses_declared_integer_boundary() -> None:
    generated = GeneratedProblem.model_validate(fibonacci_payload())
    request = AIProblemTaskPayload(requirement="使用最大规模数据严格验证递推算法的时间复杂度")

    candidate = infer_stress_candidate(generated, request)

    assert candidate is not None
    assert candidate.template == "bounded_integer"
    assert candidate.input == "100000\n"
    assert candidate.scale == {"n": 100000}


async def test_ai_file_stress_is_optional_and_used_by_judge(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "file-stress-model",
            "api_key": "secret",
        },
    )
    payload = stress_array_payload()
    calls = 0

    async def fake_provider(*_args, **kwargs):
        nonlocal calls
        calls += 1
        if not kwargs.get("json_mode"):
            return "采用线性扫描并用最大规模数组淘汰平方复杂度算法", {
                "input_tokens": 5,
                "output_tokens": 5,
            }
        return json.dumps(payload, ensure_ascii=False), {
            "input_tokens": 10,
            "output_tokens": 20,
        }

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": (
                "生成数组求和题，用最大规模大数据严格测试复杂度，"
                "时间限制 0.5 秒，确保额外校验失败不影响普通题生成"
            ),
            "difficulty": "困难",
            "testcase_count": 3,
        },
    )
    task = await _wait_task(client, response.json()["data"]["task_id"])

    assert task["status"] == "completed", task
    assert calls == 2
    file_stress = task["result"]["validation"]["file_stress"]
    assert file_stress["applied"] is True
    assert file_stress["template"] == "counted_integer_array"
    assert file_stress["count"] == 1
    assert file_stress["input_bytes"] > 50_000

    problem_data = task["result"]["problem"]
    problem = ProblemPayload.model_validate(problem_data)
    file_cases = await load_file_testcases(problem)
    assert len(file_cases) == 1
    assert file_cases[0]["input_file"].endswith(".in")
    assert file_cases[0]["output_file"].endswith(".out")

    assert (await client.post("/api/problems/", json=problem_data)).status_code == 200
    detail = (await client.get(f"/api/problems/{problem.id}")).json()["data"]
    assert detail["file_testcase_count"] == 1

    submitted = await client.post(
        "/api/submissions/",
        json={
            "problem_id": problem.id,
            "language": "python",
            "code": (
                "import sys\n"
                "d=list(map(int,sys.stdin.buffer.read().split()));n=d[0];a=d[1:]\n"
                "for _ in range(n):\n"
                "    for __ in range(n):\n"
                "        pass\n"
                "print(sum(a))\n"
            ),
        },
    )
    submission_id = int(submitted.json()["data"]["submission_id"])
    await judge.wait_for_submission(submission_id, timeout=10)
    judged = (await client.get(f"/api/submissions/{submission_id}")).json()["data"]
    assert judged["result"] == "TLE"
    assert judged["counts"] == 40
    log = (await client.get(f"/api/submissions/{submission_id}/log")).json()["data"]
    assert log["details"][-1]["source"] == "file"

    assert (await client.delete(f"/api/problems/{problem.id}")).status_code == 200
    assert await load_file_testcases(problem) == []


async def test_unsupported_file_stress_never_fails_ai_task(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "fallback-model",
            "api_key": "secret",
        },
    )
    calls = 0

    async def fake_provider(*_args, **kwargs):
        nonlocal calls
        calls += 1
        content = "复杂度分析" if not kwargs.get("json_mode") else json.dumps(generated_payload())
        return content, {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": "生成三整数输入题并尝试大数据复杂度验证，无法识别时必须正常完成",
            "testcase_count": 3,
        },
    )
    task = await _wait_task(client, response.json()["data"]["task_id"])

    assert task["status"] == "completed", task
    assert calls == 2
    assert task["result"]["validation"]["file_stress"] == {
        "requested": True,
        "applied": False,
        "count": 0,
        "reason": "未能安全识别受支持的输入结构",
    }
