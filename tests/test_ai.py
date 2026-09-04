from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from pydantic import ValidationError

from app import ai_service, judge
from app.schemas import (
    AIModelConfigPayload,
    AIProblemTaskPayload,
    GeneratedProblem,
)
from app.schemas import (
    TestCase as GeneratedTestCase,
)
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


def complexity_payload(*, stress_size: int = 12_000) -> dict:
    generator = (
        "import json\n"
        f"n={stress_size}\n"
        "data=f'{n}\\n'+' '.join(map(str,range(n,0,-1)))+'\\n'\n"
        "print(json.dumps([{'label':'最大规模逆序数组','scale':n,'input':data}]))\n"
    )
    return {
        "problem": {
            "id": "AUTO",
            "title": "逆序对计数",
            "description": "给定一个整数数组，计算满足 i<j 且 a[i]>a[j] 的下标对数量。",
            "input_description": "第一行 n，第二行 n 个整数。",
            "output_description": "输出逆序对数量。",
            "samples": [{"input": "5\n2 3 1 5 4\n", "output": "3\n"}],
            "constraints": "1 <= n <= 12000，数组元素绝对值不超过 10^9。",
            "testcases": [
                {"input": "1\n7\n", "output": "0\n"},
                {"input": "5\n1 2 3 4 5\n", "output": "0\n"},
                {"input": "5\n5 4 3 2 1\n", "output": "10\n"},
            ],
            "tags": ["归并排序", "复杂度"],
            "time_limit": 2.0,
            "memory_limit": 128,
            "difficulty": "中等",
        },
        "reference_solution": (
            "import sys\n"
            "it=iter(map(int,sys.stdin.buffer.read().split()))\n"
            "n=next(it); a=[next(it) for _ in range(n)]\n"
            "vals={v:i+1 for i,v in enumerate(sorted(set(a)))}\n"
            "bit=[0]*(len(vals)+1); ans=0\n"
            "for value in reversed(a):\n"
            "    x=vals[value]; i=x-1\n"
            "    while i: ans+=bit[i]; i-=i&-i\n"
            "    i=x\n"
            "    while i<len(bit): bit[i]+=1; i+=i&-i\n"
            "print(ans)\n"
        ),
        "reference_solution_cpp": (
            "#include <bits/stdc++.h>\nusing namespace std;\n"
            "long long solve(vector<long long>&a,vector<long long>&t,int l,int r){"
            "if(r-l<2)return 0;int m=(l+r)/2;long long z=solve(a,t,l,m)+solve(a,t,m,r);"
            "int i=l,j=m,k=l;while(i<m||j<r){if(j==r||(i<m&&a[i]<=a[j]))t[k++]=a[i++];"
            "else{t[k++]=a[j++];z+=m-i;}}for(i=l;i<r;i++)a[i]=t[i];return z;}\n"
            "int main(){ios::sync_with_stdio(false);cin.tie(nullptr);int n;if(!(cin>>n))return 0;"
            "vector<long long>a(n),t(n);for(auto&x:a)cin>>x;cout<<solve(a,t,0,n);}\n"
        ),
        "solution_explanation": "使用树状数组统计，时间复杂度 O(n log n)，空间复杂度 O(n)。",
        "stress_test_generator": generator,
        "complexity_probe_solution": (
            "import sys\n"
            "data=list(map(int,sys.stdin.buffer.read().split())); n=data[0]; a=data[1:]\n"
            "ans=0\n"
            "for i in range(n):\n"
            "    for j in range(i+1,n): ans += a[i] > a[j]\n"
            "print(ans)\n"
        ),
        "complexity_contract": {
            "expected_time_complexity": "O(n log n)",
            "expected_space_complexity": "O(n)",
            "forbidden_time_complexities": ["O(n^2)"],
            "stress_rationale": "n=12000 的逆序数组会让二重循环执行约七千万次。",
        },
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


def test_ai_resource_limits_follow_explicit_request_and_complexity() -> None:
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
    assert automatic["recommended_minimum"] == {"time_limit": 3.0, "memory_limit": 256}
    assert automatic["final"] == {"time_limit": 3.0, "memory_limit": 256}
    assert automatic_generated.problem.time_limit == 3.0
    assert automatic_generated.problem.memory_limit == 256


def test_complexity_testcase_count_includes_materialized_stress_cases() -> None:
    request = AIProblemTaskPayload(
        requirement="设计一道需要大数据淘汰暴力算法的逆序对题",
        difficulty="中等",
        testcase_count=10,
    )
    generated = GeneratedProblem.model_validate(complexity_payload())
    generated.problem.testcases = [
        GeneratedTestCase(input=f"1\n{value}\n", output="0\n") for value in range(8)
    ]

    before_errors = ai_service._static_validation_errors(generated, request, None)
    assert not any("少于要求" in error for error in before_errors)

    generated.problem.testcases.extend(
        [
            GeneratedTestCase(input="2\n2 1\n", output="1\n"),
            GeneratedTestCase(input="3\n3 2 1\n", output="3\n"),
        ]
    )
    final_errors = ai_service._static_validation_errors(
        generated,
        request,
        None,
        stress_testcase_indexes={8, 9},
    )
    assert final_errors == []


async def test_ai_materializes_stress_data_rejects_bruteforce_and_assigns_id(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "complexity-model",
            "api_key": "secret",
        },
    )
    payload = complexity_payload()
    calls = 0

    async def fake_provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        content = "逆序对复杂度与最大规模分析" if calls == 1 else json.dumps(payload)
        return content, {"input_tokens": 10, "output_tokens": 20}

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": "生成逆序对计数题，必须用大数据验证复杂度并淘汰 O(n^2) 暴力解",
            "knowledge_points": ["归并排序", "复杂度"],
            "difficulty": "中等",
            "testcase_count": 3,
        },
    )
    task = await _wait_task(client, response.json()["data"]["task_id"])

    assert task["status"] == "completed", task
    assert calls == 2
    assert task["result"]["problem"]["id"] == "AI0001"
    assert task["result"]["validation"]["id_assignment"] == {
        "source": "automatic",
        "model_id": "AUTO",
        "final_id": "AI0001",
    }
    testcases = task["result"]["problem"]["testcases"]
    assert len(testcases) == 4
    assert len(testcases[-1]["input"]) > 20_000
    assert testcases[-1]["output"] == str(12_000 * 11_999 // 2)
    complexity = task["result"]["validation"]["complexity_validation"]
    assert complexity["passed"] is True
    assert complexity["rejected_stress_cases"] == 1
    assert complexity["probe_results"][-1]["status"] == "TLE"
    assert task["result"]["validation"]["resource_tuning"]["applied"] is True
    assert task["result"]["problem"]["time_limit"] <= 2.0

    created = await client.post("/api/problems/", json=task["result"]["problem"])
    assert created.status_code == 200
    submitted = await client.post(
        "/api/submissions/",
        json={
            "problem_id": "AI0001",
            "language": "python",
            "code": payload["complexity_probe_solution"],
        },
    )
    submission_id = int(submitted.json()["data"]["submission_id"])
    await judge.wait_for_submission(submission_id, timeout=10)
    detail = (await client.get(f"/api/submissions/{submission_id}")).json()["data"]
    assert detail["result"] == "TLE"


async def test_ai_repairs_stress_data_when_bruteforce_still_passes(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await login(client)
    await client.put(
        "/api/ai/model-config",
        json={
            "provider_url": "https://example-model-provider.test/v1",
            "model": "stress-repair-model",
            "api_key": "secret",
        },
    )
    weak = complexity_payload(stress_size=50)
    strong = complexity_payload(stress_size=12_000)
    calls = 0

    async def fake_provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            content = "先验证低效算法是否会在最大规模超时"
        elif calls == 2:
            content = json.dumps(weak)
        else:
            content = json.dumps(
                {
                    "stress_test_generator": strong["stress_test_generator"],
                    "complexity_probe_solution": strong["complexity_probe_solution"],
                    "complexity_contract": strong["complexity_contract"],
                }
            )
        return content, {"input_tokens": 10, "output_tokens": 20}

    monkeypatch.setattr(ai_service, "_provider_request", fake_provider)
    response = await client.post(
        "/api/ai/problem-tasks/",
        json={
            "requirement": "生成逆序对题并严格考验复杂度，确保 O(n^2) 暴力算法超时",
            "knowledge_points": ["归并排序", "复杂度"],
            "difficulty": "中等",
        },
    )
    task = await _wait_task(client, response.json()["data"]["task_id"])

    assert task["status"] == "completed", task
    assert calls == 3
    assert task["usage"]["calls"][-1]["stage"] == "强化复杂度压力 1/2"
    assert task["result"]["validation"]["automatic_repair_count"] == 1
    complexity = task["result"]["validation"]["complexity_validation"]
    assert complexity["passed"] is True
    assert complexity["items"][0]["scale"] == 12_000
    assert complexity["probe_results"][-1]["status"] == "TLE"


async def test_complexity_validation_rejects_fake_delays_and_accepts_real_mle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = AIProblemTaskPayload(
        requirement="设计一道严格验证时空复杂度并淘汰暴力解的题目",
        difficulty="中等",
    )
    unsafe_payload = complexity_payload()
    unsafe_payload["stress_test_generator"] = "import os\nprint('[]')\n"
    unsafe = GeneratedProblem.model_validate(unsafe_payload)
    _, errors, _ = await ai_service._materialize_stress_cases(unsafe, request)
    assert any("禁用操作" in error for error in errors)

    optional_request = AIProblemTaskPayload(
        requirement="设计一道基础数组求和题，帮助初学者练习循环",
        difficulty="简单",
    )
    unchanged, errors, metadata = await ai_service._materialize_stress_cases(
        unsafe, optional_request
    )
    assert unchanged is unsafe
    assert errors == []
    assert metadata["required"] is False

    delayed = GeneratedProblem.model_validate(complexity_payload())
    delayed.complexity_probe_solution = "while True:\n    pass\n"
    delayed.problem.testcases.append(GeneratedTestCase(input="12000\n", output="71994000\n"))
    errors, _ = await ai_service._validate_complexity_probe(
        delayed,
        request,
        {3},
        {"stress_testcase_indexes": [3]},
    )
    assert any("人为延迟" in error for error in errors)

    memory_probe = GeneratedProblem.model_validate(complexity_payload())
    memory_probe.problem.testcases.append(GeneratedTestCase(input="12000\n", output="71994000\n"))

    async def fake_probe_execution(
        _problem, _solution, language="python", *, inputs=None
    ) -> judge.ReferenceExecution:
        assert language == "python"
        assert inputs is not None
        expected = [
            case.output
            for case in [
                *memory_probe.problem.samples,
                *memory_probe.problem.testcases[:3],
            ]
        ]
        results = [judge.ProcessResult("OK", 0, output, "", 0.01, 8.0) for output in expected]
        results.append(judge.ProcessResult("MLE", -9, "", "MLE", 0.2, 65.0))
        return judge.ReferenceExecution(language, None, results)

    monkeypatch.setattr(ai_service, "execute_reference_solution", fake_probe_execution)
    errors, metadata = await ai_service._validate_complexity_probe(
        memory_probe,
        request,
        {3},
        {"stress_testcase_indexes": [3]},
    )
    assert errors == []
    assert metadata["passed"] is True
    assert metadata["probe_results"][-1]["status"] == "MLE"


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
