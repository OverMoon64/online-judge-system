from __future__ import annotations

import json
import os
from types import SimpleNamespace

import httpx
import pytest

from app import ai_service, judge
from app.schemas import AIModelConfigPayload, LanguagePayload, ProblemPayload


def _config() -> AIModelConfigPayload:
    return AIModelConfigPayload(
        provider_url="https://provider.test/v1",
        model="model",
        api_key="secret",
        input_price=2,
        output_price=4,
        price_unit=1000,
    )


def _problem() -> ProblemPayload:
    return ProblemPayload(
        id="unit",
        title="单位测试",
        description="读取一个整数并原样输出。",
        input_description="一个整数。",
        output_description="同一个整数。",
        samples=[{"input": "1\n", "output": "1\n"}],
        constraints="整数绝对值不超过 100",
        testcases=[{"input": "2\n", "output": "2\n"}],
        time_limit=0.5,
        memory_limit=128,
    )


def test_language_command_validation_and_output_normalization() -> None:
    safe = LanguagePayload(
        name="safe",
        file_ext=".py",
        compile_cmd=None,
        run_cmd="python3 {src}",
    )
    judge.validate_language_commands(safe)
    assert judge.render_command("python3 {src}", judge.Path("a.py"), judge.Path("a")) == [
        "python3",
        "a.py",
    ]
    assert judge.normalize_output("  keep  \r\nnext \n\n") == "  keep\nnext"
    assert judge._sanitize_message("/tmp/private\\main.py", "/tmp/private") == "<judge>/main.py"

    invalid_commands = [
        SimpleNamespace(compile_cmd=None, run_cmd=""),
        SimpleNamespace(compile_cmd=None, run_cmd="python3 {src}; echo bad"),
        SimpleNamespace(compile_cmd=None, run_cmd="python3 '"),
        SimpleNamespace(compile_cmd=None, run_cmd="python3 {unknown}"),
        SimpleNamespace(compile_cmd=None, run_cmd="not-allowed {src}"),
        SimpleNamespace(compile_cmd="g++ -o {exe}", run_cmd="{exe}"),
        SimpleNamespace(compile_cmd=None, run_cmd="python3 -V"),
    ]
    for language in invalid_commands:
        with pytest.raises(ValueError):
            judge.validate_language_commands(language)


async def test_process_runner_unknown_runtime_output_limit_and_reference_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = await judge.run_process(["definitely-not-an-executable"], "", 0.2, 64)
    assert missing.status == "UNK"

    failed = await judge.run_process(["python3", "-c", "raise RuntimeError('x')"], "", 1, 128)
    assert failed.status == "RE"
    assert failed.returncode != 0

    monkeypatch.setattr(
        judge,
        "get_settings",
        lambda: SimpleNamespace(max_output_bytes=32, compile_timeout_seconds=5),
    )
    flooded = await judge.run_process(["python3", "-c", "print('x'*10000)"], "", 1, 128)
    assert flooded.status == "RE"

    problem = _problem()
    python_execution = await judge.execute_reference_solution(
        problem,
        "print(input().strip())",
        inputs=[case.input for case in [*problem.samples, *problem.testcases]],
    )
    assert python_execution.setup_error is None
    assert [result.status for result in python_execution.results] == ["OK", "OK"]
    assert [judge.normalize_output(result.stdout) for result in python_execution.results] == [
        "1",
        "2",
    ]
    assert await judge.validate_reference_solution(problem, "print(input().strip())") == []
    mismatch = await judge.validate_reference_solution(problem, "print(999)")
    assert mismatch == ["testcase 1: reference solution output mismatch"]
    runtime = await judge.validate_reference_solution(problem, "raise RuntimeError('bad')")
    assert runtime == ["testcase 1: RE"]
    cpp_solution = (
        "#include <iostream>\nusing namespace std;\n"
        "int main(){string value;getline(cin,value);cout<<value;}\n"
    )
    assert await judge.validate_reference_solution(problem, cpp_solution, language="cpp") == []
    cpp_execution = await judge.execute_reference_solution(
        problem,
        cpp_solution,
        language="cpp",
        inputs=["7\n", "8\n"],
    )
    assert cpp_execution.setup_error is None
    assert [judge.normalize_output(result.stdout) for result in cpp_execution.results] == [
        "7",
        "8",
    ]
    cpp_compile_error = await judge.validate_reference_solution(
        problem, "int main( {", language="cpp"
    )
    assert cpp_compile_error[0].startswith("compile:")
    await judge.cancel_submission_task(99999)
    await judge.cancel_all_submission_tasks()


class _FakeResponse:
    def __init__(self, payload=None, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://provider.test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("failed", request=request, response=response)

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload

    async def aiter_lines(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        if isinstance(self.payload, list):
            for line in self.payload:
                yield line
            return
        yield json.dumps(self.payload)


class _FakeStreamContext:
    def __init__(self, response) -> None:
        self.response = response

    async def __aenter__(self):
        if isinstance(self.response, Exception):
            raise self.response
        return self.response

    async def __aexit__(self, *args) -> None:
        del args


class _FakeClient:
    response: _FakeResponse | Exception
    fallback_response: _FakeResponse | Exception | None = None
    last_json: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        del args

    async def post(self, *args, **kwargs):
        del args, kwargs
        response = self.fallback_response or self.response
        if isinstance(response, Exception):
            raise response
        return response

    def stream(self, *args, **kwargs):
        del args
        self.__class__.last_json = kwargs["json"]
        return _FakeStreamContext(self.response)


async def test_ai_provider_protocol_and_json_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ai_service.httpx, "AsyncClient", _FakeClient)
    _FakeClient.response = _FakeResponse(
        {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4},
        }
    )
    content, usage = await ai_service._provider_request(_config(), [])
    assert content == "answer"
    assert usage["input_tokens"] == 3
    assert usage["output_tokens"] == 4
    assert usage["provider_total_tokens"] == 7
    assert _FakeClient.last_json["stream"] is True

    _FakeClient.response = _FakeResponse(
        [
            'data: {"choices":[{"delta":{"content":"ans"}}]}',
            'data: {"choices":[{"delta":{"content":"wer"}}]}',
            ('data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":6}}'),
            "data: [DONE]",
        ]
    )
    deltas: list[str] = []

    async def capture_delta(delta: str) -> None:
        deltas.append(delta)

    content, usage = await ai_service._provider_request(_config(), [], on_delta=capture_delta)
    assert content == "answer"
    assert deltas == ["ans", "wer"]
    assert usage["input_tokens"] == 5
    assert usage["output_tokens"] == 6

    _FakeClient.response = _FakeResponse({}, status_code=400)
    _FakeClient.fallback_response = _FakeResponse(
        {
            "choices": [{"message": {"content": "fallback answer"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 8},
        }
    )
    content, usage = await ai_service._provider_request(_config(), [])
    assert content == "fallback answer"
    assert usage["input_tokens"] == 7
    assert usage["output_tokens"] == 8
    _FakeClient.fallback_response = None

    _FakeClient.response = _FakeResponse({}, status_code=401)
    with pytest.raises(RuntimeError, match="HTTP 401"):
        await ai_service._provider_request(_config(), [])
    _FakeClient.response = httpx.ReadTimeout("slow")
    with pytest.raises(RuntimeError, match="超时"):
        await ai_service._provider_request(_config(), [])
    _FakeClient.response = _FakeResponse(ValueError("bad json"))
    with pytest.raises(RuntimeError, match="不兼容"):
        await ai_service._provider_request(_config(), [])
    _FakeClient.response = httpx.ConnectError("offline")
    with pytest.raises(RuntimeError, match="无法连接"):
        await ai_service._provider_request(_config(), [])

    assert ai_service._completion_endpoint("https://x/v1").endswith("/chat/completions")
    full = "https://x/v1/chat/completions"
    assert ai_service._completion_endpoint(full) == full
    assert ai_service._extract_json('```json\n{"x": 1}\n```') == {"x": 1}
    assert ai_service._extract_json('{"text":"line one\nline two"}') == {
        "text": "line one\nline two"
    }
    with pytest.raises(ValueError, match="没有 JSON"):
        ai_service._extract_json("plain text")
    with pytest.raises(ValueError, match="没有 JSON"):
        ai_service._extract_json("[1, 2]")
    generated = {
        "id": "nested",
        "title": "nested",
        "description": "nested",
        "reference_solution": "print(1)",
        "cpp_solution": "int main(){}",
    }
    normalized = ai_service._normalize_generated_payload(generated)
    assert normalized["problem"]["id"] == "nested"
    assert normalized["reference_solution"] == "print(1)"
    assert normalized["reference_solution_cpp"] == "int main(){}"
    normalized_cases = ai_service._normalize_generated_payload(
        {
            "problem": {
                "samples": [{"input": 10, "output": 55}],
                "testcases": [{"input": 100000}],
            }
        }
    )["problem"]
    assert normalized_cases["samples"] == [{"input": "10", "output": "55"}]
    assert normalized_cases["testcases"] == [{"input": "100000", "output": ""}]

    total = ai_service._empty_usage(_config())
    ai_service._merge_usage(total, {"input_tokens": 100, "output_tokens": 50}, _config())
    assert total["total_tokens"] == 150
    assert total["cost"] == 0.4
    assert total["estimated"] is False

    detailed = ai_service._token_usage(
        {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 7},
        }
    )
    assert detailed["cached_input_tokens"] == 4
    assert detailed["reasoning_tokens"] == 7

    qwen = AIModelConfigPayload(
        provider_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen3.7-plus",
        api_key="secret",
    )
    _FakeClient.response = _FakeResponse(
        {"choices": [{"message": {"content": "answer"}}], "usage": {"total_tokens": 1}}
    )
    await ai_service._provider_request(qwen, [], json_mode=True)
    assert _FakeClient.last_json["enable_thinking"] is False
    assert _FakeClient.last_json["max_completion_tokens"] == 12_000
    assert _FakeClient.last_json["response_format"] == {"type": "json_object"}


async def test_private_ai_provider_rules(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    local = AIModelConfigPayload(
        provider_url="http://127.0.0.1:9999/v1",
        model="local",
        api_key="secret",
    )
    monkeypatch.setattr(
        ai_service,
        "get_settings",
        lambda: SimpleNamespace(
            allow_local_ai=True,
            session_secret="test-encryption-secret",
            ai_config_file=str(tmp_path / "fallback.enc"),
        ),
    )
    store = tmp_path / "models.enc"
    await ai_service.configure_model_store(store)
    public = await ai_service.set_model_config(123, local)
    assert public["api_key_configured"] is True
    assert "api_key" not in public
    assert await ai_service.get_model_config(123) == public
    encrypted = store.read_bytes()
    assert b"secret" not in encrypted
    assert b"local" not in encrypted

    # Simulate a process restart: clear memory, point at the same encrypted file,
    # and ensure the model can be selected without entering the key again.
    await ai_service.configure_model_store(store)
    restored = await ai_service.list_model_configs(123)
    assert restored["active"] == "默认模型"
    assert restored["models"][0]["model"] == "local"
    assert "api_key" not in restored["models"][0]
    assert await ai_service.get_model_config(999) is None


def test_process_memory_probe_handles_missing_process() -> None:
    assert judge._rss_tree_mb(os.getpid()) > 0
    assert judge._rss_tree_mb(999_999_999) == 0
