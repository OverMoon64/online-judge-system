from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import re
import uuid
from base64 import urlsafe_b64encode
from collections.abc import Awaitable, Callable
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from cryptography.fernet import Fernet, InvalidToken
from pydantic import ValidationError
from sqlalchemy import select

from app import db as database
from app.config import get_settings
from app.db import AIProblemTask, Problem, utc_now
from app.judge import ReferenceExecution, execute_reference_solution, normalize_output
from app.schemas import AIModelConfigPayload, AIProblemTaskPayload, GeneratedProblem
from app.stress_cases import build_optional_file_stress, stress_requested
from app.testcase_files import remove_file_testcases

_model_configs: dict[int, dict[str, AIModelConfigPayload]] = {}
_active_model_configs: dict[int, str] = {}
_model_usage: dict[int, dict[str, dict[str, Any]]] = {}
_model_store_path_override: Path | None = None
_model_store_loaded = False
_model_store_lock: asyncio.Lock | None = None
_ai_id_lock: asyncio.Lock | None = None
_reserved_ai_ids: set[str] = set()
_running_tasks: dict[str, asyncio.Task[None]] = {}
StreamCallback = Callable[[str], Awaitable[None]]


def _public_config(
    config: AIModelConfigPayload,
    *,
    usage: dict[str, Any] | None = None,
    active: bool = False,
) -> dict[str, Any]:
    return {
        "name": config.name,
        "provider_url": config.provider_url,
        "model": config.model,
        "api_key_configured": True,
        "input_price": config.input_price,
        "output_price": config.output_price,
        "price_unit": config.price_unit,
        "currency": config.currency,
        "disable_thinking": config.disable_thinking,
        "active": active,
        "usage": usage
        or {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "currency": config.currency,
        },
    }


def _store_path() -> Path:
    return _model_store_path_override or Path(get_settings().ai_config_file)


def _store_fernet() -> Fernet:
    digest = sha256(get_settings().session_secret.encode("utf-8")).digest()
    return Fernet(urlsafe_b64encode(digest))


def _get_store_lock() -> asyncio.Lock:
    global _model_store_lock
    if _model_store_lock is None:
        _model_store_lock = asyncio.Lock()
    return _model_store_lock


def _get_ai_id_lock() -> asyncio.Lock:
    global _ai_id_lock
    if _ai_id_lock is None:
        _ai_id_lock = asyncio.Lock()
    return _ai_id_lock


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


async def _persist_model_store() -> None:
    payload = {
        "version": 1,
        "users": {
            str(user_id): {
                "active": _active_model_configs.get(user_id),
                "models": {
                    name: config.model_dump(mode="json") for name, config in configs.items()
                },
                "usage": _model_usage.get(user_id, {}),
            }
            for user_id, configs in _model_configs.items()
        },
    }
    encrypted = _store_fernet().encrypt(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    await asyncio.to_thread(_atomic_write, _store_path(), encrypted)


async def _ensure_model_store_loaded() -> None:
    global _model_store_loaded
    if _model_store_loaded:
        return
    async with _get_store_lock():
        if _model_store_loaded:
            return
        path = _store_path()
        if path.exists():
            try:
                encrypted = await asyncio.to_thread(path.read_bytes)
                decoded = _store_fernet().decrypt(encrypted)
                payload = json.loads(decoded.decode("utf-8"))
                for user_id_text, stored in payload.get("users", {}).items():
                    user_id = int(user_id_text)
                    configs = {
                        name: AIModelConfigPayload.model_validate(
                            {
                                key: value
                                for key, value in config.items()
                                if key in AIModelConfigPayload.model_fields
                            }
                        )
                        for name, config in stored.get("models", {}).items()
                    }
                    if configs:
                        _model_configs[user_id] = configs
                        active = stored.get("active")
                        _active_model_configs[user_id] = (
                            active if active in configs else next(iter(configs))
                        )
                        _model_usage[user_id] = stored.get("usage", {})
            except (InvalidToken, ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "本地模型配置无法解密，请确认 OJ_SESSION_SECRET 未发生变化"
                ) from exc
        _model_store_loaded = True


async def configure_model_store(path: Path) -> None:
    global _model_store_path_override, _model_store_loaded, _model_store_lock
    _model_store_path_override = path
    _model_configs.clear()
    _active_model_configs.clear()
    _model_usage.clear()
    _model_store_loaded = False
    _model_store_lock = None


async def set_model_config(user_id: int, config: AIModelConfigPayload) -> dict[str, Any]:
    parsed = urlparse(config.provider_url)
    host = parsed.hostname or ""
    local_host = host.lower() in {"localhost", "localhost.localdomain"}
    with contextlib.suppress(ValueError):
        local_host = local_host or ipaddress.ip_address(host).is_private
    if (parsed.scheme != "https" or local_host) and not get_settings().allow_local_ai:
        raise ValueError("provider URL must use public HTTPS unless OJ_ALLOW_LOCAL_AI=true")
    await _ensure_model_store_loaded()
    async with _get_store_lock():
        configs = _model_configs.setdefault(user_id, {})
        configs[config.name] = config
        _active_model_configs[user_id] = config.name
        usage = _model_usage.setdefault(user_id, {}).setdefault(config.name, _empty_usage(config))
        await _persist_model_store()
    return _public_config(config, usage=usage, active=True)


async def get_model_config(user_id: int) -> dict[str, Any] | None:
    await _ensure_model_store_loaded()
    configs = _model_configs.get(user_id, {})
    active = _active_model_configs.get(user_id)
    config = configs.get(active or "")
    return (
        _public_config(
            config,
            usage=_model_usage.get(user_id, {}).get(config.name),
            active=True,
        )
        if config
        else None
    )


async def list_model_configs(user_id: int) -> dict[str, Any]:
    await _ensure_model_store_loaded()
    active = _active_model_configs.get(user_id)
    models = [
        _public_config(
            config,
            usage=_model_usage.get(user_id, {}).get(name),
            active=name == active,
        )
        for name, config in _model_configs.get(user_id, {}).items()
    ]
    return {"active": active, "models": models}


async def delete_model_config(user_id: int, name: str) -> None:
    await _ensure_model_store_loaded()
    async with _get_store_lock():
        configs = _model_configs.get(user_id, {})
        if name not in configs:
            raise ValueError("model config not found")
        configs.pop(name)
        _model_usage.get(user_id, {}).pop(name, None)
        if not configs:
            _model_configs.pop(user_id, None)
            _model_usage.pop(user_id, None)
            _active_model_configs.pop(user_id, None)
        elif _active_model_configs.get(user_id) == name:
            _active_model_configs[user_id] = next(iter(configs))
        await _persist_model_store()


def _completion_endpoint(base_url: str) -> str:
    return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"


async def _provider_request(
    config: AIModelConfigPayload,
    messages: list[dict[str, str]],
    *,
    on_delta: StreamCallback | None = None,
    json_mode: bool = False,
) -> tuple[str, dict[str, Any]]:
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.35,
    }
    provider_host = (urlparse(config.provider_url).hostname or "").lower()
    is_dashscope_qwen = config.model.lower().startswith("qwen") and (
        "aliyuncs.com" in provider_host
    )
    if is_dashscope_qwen:
        payload["max_completion_tokens"] = 12_000
        if config.disable_thinking:
            payload["enable_thinking"] = False
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
            endpoint = _completion_endpoint(config.provider_url)
            stream_payload = {
                **payload,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            async with client.stream(
                "POST",
                endpoint,
                headers=headers,
                json=stream_payload,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code not in {400, 404, 422}:
                        raise
                    fallback = await client.post(endpoint, headers=headers, json=payload)
                    fallback.raise_for_status()
                    content, usage = _parse_non_stream_response(fallback.json())
                    if on_delta:
                        await on_delta(content)
                    return content, usage
                return await _consume_stream_response(response, on_delta)
    except httpx.TimeoutException as exc:
        raise RuntimeError("模型服务请求超时") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"模型服务返回 HTTP {exc.response.status_code}") from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("模型服务返回了不兼容的响应格式") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("无法连接模型服务") from exc


async def _provider_request_with_retry(
    config: AIModelConfigPayload,
    messages: list[dict[str, str]],
    *,
    retry_count: int = 2,
    on_delta: StreamCallback | None = None,
    json_mode: bool = False,
) -> tuple[str, dict[str, Any]]:
    for attempt in range(retry_count + 1):
        try:
            content, usage = await _provider_request(
                config,
                messages,
                on_delta=on_delta,
                json_mode=json_mode,
            )
            usage = dict(usage)
            usage["request_count"] = attempt + 1
            if attempt:
                usage["estimated"] = True
                usage["unreported_attempts"] = attempt
            return content, usage
        except RuntimeError:
            if attempt >= retry_count:
                raise
            await asyncio.sleep(0.5 * (2**attempt))
    raise RuntimeError("模型服务请求失败")


def _token_usage(value: dict[str, Any] | None) -> dict[str, Any]:
    usage = value or {}
    input_tokens = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    output_tokens = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    input_details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    output_details = (
        usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider_total_tokens": int(
            usage.get("total_tokens", usage.get("total_token_count", input_tokens + output_tokens))
            or 0
        ),
        "cached_input_tokens": int(input_details.get("cached_tokens", 0) or 0),
        "reasoning_tokens": int(output_details.get("reasoning_tokens", 0) or 0),
        "request_count": 1,
        "estimated": not bool(value),
    }


def _parse_non_stream_response(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    choice = body["choices"][0]
    content = choice["message"]["content"]
    if not isinstance(content, str) or not content:
        raise ValueError("model response content is empty")
    usage = _token_usage(body.get("usage"))
    if choice.get("finish_reason"):
        usage["finish_reason"] = choice["finish_reason"]
    return content, usage


def _delta_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            str(item.get("text", ""))
            for item in value
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    return ""


async def _consume_stream_response(
    response: httpx.Response,
    on_delta: StreamCallback | None,
) -> tuple[str, dict[str, Any]]:
    content_parts: list[str] = []
    raw_lines: list[str] = []
    usage = _token_usage(None)
    finish_reason: str | None = None
    received_event = False
    async for line in response.aiter_lines():
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith("data:"):
            raw_lines.append(stripped)
            continue
        received_event = True
        event_data = stripped.removeprefix("data:").strip()
        if event_data == "[DONE]":
            continue
        event = json.loads(event_data)
        if event.get("usage"):
            usage = _token_usage(event["usage"])
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]
        delta = _delta_text((choice.get("delta") or {}).get("content"))
        if delta:
            content_parts.append(delta)
            if on_delta:
                await on_delta(delta)
    if not received_event:
        return _parse_non_stream_response(json.loads("\n".join(raw_lines)))
    content = "".join(content_parts)
    if not content:
        raise ValueError("model stream content is empty")
    if finish_reason:
        usage["finish_reason"] = finish_reason
    return content, usage


def _empty_usage(config: AIModelConfigPayload) -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "provider_total_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_tokens": 0,
        "request_count": 0,
        "cost": 0.0,
        "currency": config.currency,
        "price_unit": config.price_unit,
        "estimated": False,
    }


def _merge_usage(
    total: dict[str, Any], current: dict[str, Any], config: AIModelConfigPayload
) -> None:
    for field in (
        "input_tokens",
        "output_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
        "request_count",
    ):
        total[field] = int(total.get(field, 0)) + int(current.get(field, 0) or 0)
    current_provider_total = int(
        current.get(
            "provider_total_tokens",
            int(current.get("input_tokens", 0)) + int(current.get("output_tokens", 0)),
        )
        or 0
    )
    total["provider_total_tokens"] = int(total.get("provider_total_tokens", 0)) + int(
        current_provider_total
    )
    total["total_tokens"] = total["input_tokens"] + total["output_tokens"]
    total["cost"] = round(
        total["input_tokens"] / config.price_unit * config.input_price
        + total["output_tokens"] / config.price_unit * config.output_price,
        8,
    )
    total["estimated"] = (
        bool(total.get("estimated")) or bool(current.get("estimated")) or total["total_tokens"] == 0
    )
    if current.get("unreported_attempts"):
        total["unreported_attempts"] = int(total.get("unreported_attempts", 0)) + int(
            current["unreported_attempts"]
        )


def _merge_stage_usage(
    total: dict[str, Any], current: dict[str, Any], config: AIModelConfigPayload, stage: str
) -> None:
    _merge_usage(total, current, config)
    total.setdefault("calls", []).append(
        {
            "stage": stage,
            "input_tokens": int(current.get("input_tokens", 0) or 0),
            "output_tokens": int(current.get("output_tokens", 0) or 0),
            "reasoning_tokens": int(current.get("reasoning_tokens", 0) or 0),
            "request_count": int(current.get("request_count", 1) or 1),
            "finish_reason": current.get("finish_reason"),
        }
    )


async def _record_model_usage(
    user_id: int,
    config_name: str,
    current: dict[str, Any],
    config: AIModelConfigPayload,
) -> None:
    await _ensure_model_store_loaded()
    async with _get_store_lock():
        usage = _model_usage.setdefault(user_id, {}).setdefault(config_name, _empty_usage(config))
        _merge_usage(usage, current, config)
        await _persist_model_store()


async def _update_task(
    task_id: str,
    *,
    status: str | None = None,
    progress: str | None = None,
    progress_percent: int | None = None,
    usage: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    async with database.session_factory() as session:
        task = await session.get(AIProblemTask, task_id)
        if task is None:
            return
        if status is not None:
            task.status = status
        if progress is not None:
            task.progress = progress
        if progress_percent is not None:
            task.progress_percent = progress_percent
        if usage is not None:
            task.usage = dict(usage)
        if result is not None:
            task.result = result
        if error is not None:
            task.error = error[:4000]
        task.updated_at = utc_now()
        await session.commit()


def _stream_preview_callback(task_id: str, stage: str) -> StreamCallback:
    streamed_text = ""
    pending_characters = 0
    last_update = 0.0

    async def update_preview(delta: str) -> None:
        nonlocal streamed_text, pending_characters, last_update
        streamed_text += delta
        pending_characters += len(delta)
        now = asyncio.get_running_loop().time()
        if pending_characters < 160 and now - last_update < 0.5:
            return
        pending_characters = 0
        last_update = now
        await _update_task(
            task_id,
            result={
                "stream_stage": stage,
                "stream_preview": streamed_text[-4000:],
            },
        )

    return update_preview


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip())
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("模型响应中没有 JSON 对象")
    value, _ = json.JSONDecoder(strict=False).raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("模型响应根节点必须是对象")
    return _normalize_generated_payload(value)


def _normalize_generated_payload(value: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(value)
    problem = normalized.get("problem")
    solution_fields = {
        "reference_solution": ("python_solution", "reference_solution_python"),
        "reference_solution_cpp": ("cpp_solution", "cxx_solution"),
        "solution_explanation": ("explanation",),
    }
    if not isinstance(problem, dict) and {"id", "title", "description"}.issubset(normalized):
        problem = {
            key: field_value
            for key, field_value in normalized.items()
            if key not in solution_fields
            and not any(key in aliases for aliases in solution_fields.values())
        }
        normalized = {"problem": problem}
    if isinstance(problem, dict):
        problem = dict(problem)
        normalized["problem"] = problem
        for case_field in ("samples", "testcases"):
            cases = problem.get(case_field)
            if not isinstance(cases, list):
                continue
            normalized_cases = []
            for case in cases:
                if not isinstance(case, dict):
                    normalized_cases.append(case)
                    continue
                normalized_case = dict(case)
                if isinstance(normalized_case.get("input"), (int, float)):
                    normalized_case["input"] = str(normalized_case["input"])
                if "output" not in normalized_case:
                    normalized_case["output"] = ""
                elif isinstance(normalized_case.get("output"), (int, float)):
                    normalized_case["output"] = str(normalized_case["output"])
                normalized_cases.append(normalized_case)
            problem[case_field] = normalized_cases
        for canonical, aliases in solution_fields.items():
            if canonical not in normalized:
                for candidate in (canonical, *aliases):
                    if candidate in problem:
                        normalized[canonical] = problem.pop(candidate)
                        break
                    if candidate in value:
                        normalized[canonical] = value[candidate]
                        break
    return normalized


_time_limit_pattern = re.compile(
    r"(?:时间限制|限制时间|时限|运行时间(?:限制)?|time\s*limit)"
    r"[^0-9]{0,16}([0-9]+(?:\.[0-9]+)?)\s*(?:秒|s\b)",
    flags=re.IGNORECASE,
)
_memory_limit_pattern = re.compile(
    r"(?:内存限制|限制内存|空间限制|内存|memory\s*limit)"
    r"[^0-9]{0,16}([0-9]+(?:\.[0-9]+)?)\s*(gb|gib|mb|mib|兆|吉字节)",
    flags=re.IGNORECASE,
)
_test_input_placeholder_pattern = re.compile(
    r"(?:^|\s)\.\.\.(?:\s|$)|…|\bomitted\b|\bplaceholder\b|"
    r"\bgenerate_[a-z0-9_]+\b|generated\s+by\s+(?:the\s+)?ref|省略|占位|待生成",
    flags=re.IGNORECASE,
)


def _explicit_resource_limits(requirement: str) -> tuple[float | None, int | None]:
    time_match = _time_limit_pattern.search(requirement)
    memory_match = _memory_limit_pattern.search(requirement)
    time_limit = float(time_match.group(1)) if time_match else None
    memory_limit: int | None = None
    if memory_match:
        amount = float(memory_match.group(1))
        unit = memory_match.group(2).lower()
        if unit in {"gb", "gib", "吉字节"}:
            amount *= 1024
        memory_limit = int(round(amount))
    if time_limit is not None and not 0 < time_limit <= 60:
        raise ValueError("命题需求中的时间限制必须在 0 到 60 秒之间")
    if memory_limit is not None and not 0 < memory_limit <= 4096:
        raise ValueError("命题需求中的内存限制必须在 1 到 4096 MB 之间")
    return time_limit, memory_limit


def _recommended_resource_limits(
    request: AIProblemTaskPayload, generated: GeneratedProblem | None = None
) -> tuple[float, int, list[str]]:
    difficulty_profiles = {
        "入门": (1.0, 64),
        "简单": (1.5, 128),
        "中等": (2.0, 256),
        "困难": (3.0, 512),
    }
    difficulty = _effective_difficulty(request, generated)
    time_limit, memory_limit = difficulty_profiles[difficulty]
    reasons = [f"按{difficulty}难度设置"]
    context_parts = [request.requirement, request.difficulty, *request.knowledge_points]
    if generated is not None:
        context_parts.extend(
            [
                generated.problem.title,
                generated.problem.description,
                generated.problem.constraints,
                generated.solution_explanation,
                *generated.problem.tags,
            ]
        )
    context = " ".join(context_parts).lower()
    time_heavy = (
        "回溯",
        "状态压缩",
        "网络流",
        "最短路",
        "复杂搜索",
        "大规模",
        "百万",
        "线段树",
        "后缀",
    )
    memory_heavy = (
        "动态规划",
        "状态压缩",
        "图",
        "网格",
        "矩阵",
        "bfs",
        "dfs",
        "搜索",
        "并查集",
        "最短路",
    )
    if any(keyword in context for keyword in time_heavy):
        time_limit = max(time_limit, 4.0)
        reasons.append("算法或数据规模偏重")
    elif any(
        keyword in context
        for keyword in ("动态规划", "滑动窗口", "双指针", "排序", "bfs", "dfs", "搜索")
    ):
        time_limit = max(time_limit, 2.0)
        reasons.append("需要遍历较大状态或数据结构")
    if any(keyword in context for keyword in memory_heavy):
        memory_limit = max(memory_limit, 512)
        reasons.append("需要维护图、网格或状态结构")
    return time_limit, memory_limit, reasons


def _effective_difficulty(
    request: AIProblemTaskPayload, generated: GeneratedProblem | None = None
) -> str:
    allowed = {"入门", "简单", "中等", "困难"}
    if request.difficulty != "自动":
        return request.difficulty
    if generated is not None and generated.problem.difficulty in allowed:
        return generated.problem.difficulty
    return "中等"


def _apply_difficulty_policy(
    generated: GeneratedProblem, request: AIProblemTaskPayload
) -> dict[str, str]:
    original = generated.problem.difficulty
    if request.difficulty == "自动":
        final = _effective_difficulty(request, generated)
        source = "model" if original == final else "fallback"
    else:
        final = request.difficulty
        source = "requested"
    generated.problem.difficulty = final
    return {"source": source, "model": original, "final": final}


def _target_testcase_count(
    request: AIProblemTaskPayload, generated: GeneratedProblem | None = None
) -> int:
    if request.testcase_count is not None:
        return request.testcase_count
    return {"入门": 2, "简单": 2, "中等": 3, "困难": 5}[_effective_difficulty(request, generated)]


def _apply_resource_limit_policy(
    generated: GeneratedProblem, request: AIProblemTaskPayload
) -> dict[str, Any]:
    explicit_time, explicit_memory = _explicit_resource_limits(request.requirement)
    recommended_time, recommended_memory, reasons = _recommended_resource_limits(request, generated)
    original_time = generated.problem.time_limit
    original_memory = generated.problem.memory_limit
    final_time = explicit_time if explicit_time is not None else recommended_time
    final_memory = explicit_memory if explicit_memory is not None else recommended_memory
    generated.problem.time_limit = round(final_time, 2)
    generated.problem.memory_limit = int(final_memory)
    source = (
        "explicit"
        if explicit_time is not None and explicit_memory is not None
        else "mixed"
        if explicit_time is not None or explicit_memory is not None
        else "automatic"
    )
    if explicit_time is not None:
        reasons.append("采用需求中明确指定的时间限制")
    if explicit_memory is not None:
        reasons.append("采用需求中明确指定的内存限制")
    return {
        "source": source,
        "requested": {"time_limit": explicit_time, "memory_limit": explicit_memory},
        "model": {"time_limit": original_time, "memory_limit": original_memory},
        "automatic": {
            "time_limit": recommended_time,
            "memory_limit": recommended_memory,
        },
        "final": {
            "time_limit": generated.problem.time_limit,
            "memory_limit": generated.problem.memory_limit,
        },
        "reasons": list(dict.fromkeys(reasons)),
    }


def _resource_limit_instruction(request: AIProblemTaskPayload) -> str:
    explicit_time, explicit_memory = _explicit_resource_limits(request.requirement)
    recommended_time, recommended_memory, _ = _recommended_resource_limits(request)
    time_value = explicit_time if explicit_time is not None else recommended_time
    memory_value = explicit_memory if explicit_memory is not None else recommended_memory
    requirements: list[str] = []
    if explicit_time is not None:
        requirements.append(f"time_limit 必须严格为 {explicit_time:g} 秒")
    else:
        requirements.append(f"系统将按最终难度自动设置时间，当前参考值 {recommended_time:g} 秒")
    if explicit_memory is not None:
        requirements.append(f"memory_limit 必须严格为 {explicit_memory} MB")
    else:
        requirements.append(f"系统将按最终难度自动设置内存，当前参考值 {recommended_memory} MB")
    return (
        "；".join(requirements)
        + f"。JSON 填写 time_limit={time_value:g}、memory_limit={memory_value}，"
        "最终值由系统统一校正。"
    )


async def _allocate_ai_problem_id() -> str:
    async with _get_ai_id_lock():
        async with database.session_factory() as session:
            ids = set((await session.scalars(select(Problem.id))).all())
            completed_results = list(
                (
                    await session.scalars(
                        select(AIProblemTask.result).where(AIProblemTask.status == "completed")
                    )
                ).all()
            )
        ids.update(_reserved_ai_ids)
        for result in completed_results:
            if isinstance(result, dict):
                problem = result.get("problem")
                if isinstance(problem, dict) and isinstance(problem.get("id"), str):
                    ids.add(problem["id"])
        sequence = [
            int(match.group(1))
            for problem_id in ids
            if (match := re.fullmatch(r"AI(\d{4,})", problem_id, flags=re.IGNORECASE))
        ]
        allocated = f"AI{max(sequence, default=0) + 1:04d}"
        _reserved_ai_ids.add(allocated)
        return allocated


def _generation_prompt(
    request: AIProblemTaskPayload, blueprint: str, existing: dict[str, Any] | None
) -> str:
    existing_text = json.dumps(existing, ensure_ascii=False)[:8_000] if existing else "无"
    if request.testcase_count is not None:
        testcase_requirement = f"生成至少 {request.testcase_count} 个互不重复的隐藏测试点"
    elif request.difficulty == "自动":
        testcase_requirement = (
            "先判定 difficulty，再按入门/简单至少 2 个、中等至少 3 个、困难至少 5 个生成隐藏测试点"
        )
    else:
        testcase_requirement = f"生成至少 {_target_testcase_count(request)} 个互不重复的隐藏测试点"
    resource_requirement = _resource_limit_instruction(request)
    prompt_time, prompt_memory, _ = _recommended_resource_limits(request)
    explicit_time, explicit_memory = _explicit_resource_limits(request.requirement)
    prompt_time = explicit_time if explicit_time is not None else prompt_time
    prompt_memory = explicit_memory if explicit_memory is not None else prompt_memory
    prompt_difficulty = "中等" if request.difficulty == "自动" else request.difficulty
    prompt_id = str(existing.get("id")) if existing else "AUTO"
    id_instruction = (
        f"本任务修改已有题目，id 必须保持为 {prompt_id}。"
        if existing
        else "新题 id 统一由系统在校验通过后分配为 AI0001、AI0002……；此处固定填写 AUTO。"
    )
    difficulty_instruction = (
        "请根据预期算法和数据范围，从入门、简单、中等、困难中选择 difficulty。"
        if request.difficulty == "自动"
        else f"difficulty 必须填写 {request.difficulty}。"
    )
    return f"""
你是程序设计训练课程的严谨命题专家。请根据命题需求和分析草案生成一道可直接导入 OJ 的题目。

命题需求：{request.requirement}
知识点：{", ".join(request.knowledge_points) or "由需求推断"}
难度：{request.difficulty}。{difficulty_instruction}
{testcase_requirement}，覆盖最小值、最大值、特殊结构和能区分错误算法的数据。
资源限制：{resource_requirement}
参考已有题目：{existing_text}
分析草案：{blueprint}

只返回一个 JSON 对象，禁止 Markdown。格式必须为：
{{
  "problem": {{
    "id": "{prompt_id}", "title": "标题", "description": "题面",
    "input_description": "输入格式", "output_description": "输出格式",
    "samples": [{{"input": "...", "output": "..."}}],
    "constraints": "数据范围", "testcases": [{{"input": "...", "output": "..."}}],
    "hint": "", "source": "AI 辅助命题", "tags": ["知识点"],
    "time_limit": {prompt_time:g}, "memory_limit": {prompt_memory}, "author": "AI Assistant",
    "difficulty": "{prompt_difficulty}"
  }},
    "reference_solution": "完整 Python 3 程序，换行必须写成 JSON 转义字符 \\n",
    "reference_solution_cpp": "完整 C++14 程序，换行必须写成 JSON 转义字符 \\n",
  "solution_explanation": "正确性和复杂度说明"
}}
{id_instruction}
两个参考程序必须使用相同算法并对所有 output 给出一致结果；Python 只能使用标准库，C++ 使用 C++14。
若题目涉及递推数列，必须在题面和解法中明确 F(0)/F(1) 或 F(1)/F(2) 等下标定义。
至少提供一个答案可人工核对的小规模样例，且该样例 output 必须准确且非空，作为题意安全锚点。
大规模隐藏测试点不要求手算，无法可靠计算时 output 可以暂填空字符串，系统会交叉运行参考程序后校准。
Python 与 C++ 必须按题意独立计算，严禁硬编码样例、隐藏测试输入或对应答案。
测试点必须可以直接作为标准输入；除非输入格式明确允许，否则禁止使用空输入。
input 字段必须包含完整的字面输入数据，严禁使用省略号、GENERATE_*、placeholder、
“omitted for brevity”“由参考程序生成”等描述或宏。若完整大矩阵/长序列超过长度限制，
应使用规模较小但结构有效的完整数据；只有题目输入格式本身定义了种子或紧凑表示时才能使用它。
每个测试点的 input 和 output 均不得超过 2000 字符。大规模边界应使用紧凑的数字输入，
不要展开数千个重复字符；优先选择边界值、特殊结构和能区分常见错误算法的紧凑输入。
整个 JSON 应简洁完整。
所有字符串中的换行、制表符和控制字符都必须按 JSON 规则转义，禁止输出字面控制字符。
""".strip()


def _solution_repair_prompt(
    generated: GeneratedProblem,
    *,
    language: str,
    errors: list[str],
) -> str:
    key = "reference_solution_cpp" if language == "C++" else "reference_solution"
    problem = generated.problem.model_dump(mode="json")
    return (
        f"只修复 {language} 参考程序。只返回严格 JSON 对象，唯一根字段为 {key}。\n"
        "程序必须按题面读取标准输入，对每个测试点都输出结果；不得修改题面或测试点。\n"
        f"题目和测试点：{json.dumps(problem, ensure_ascii=False)}\n"
        f"校验错误：{'; '.join(errors[:20])}"
    )


def _test_data_repair_prompt(
    generated: GeneratedProblem,
    request: AIProblemTaskPayload,
    *,
    errors: list[str],
) -> str:
    problem_contract = generated.problem.model_dump(mode="json")
    problem_contract.pop("samples", None)
    problem_contract.pop("testcases", None)
    target_count = _target_testcase_count(request, generated)
    return (
        "只修复样例和隐藏测试数据，不得修改题意、输入格式、参考程序或资源限制。"
        "只返回严格 JSON 对象，且只能有 samples 和 testcases 两个根字段。\n"
        f"题目契约：{json.dumps(problem_contract, ensure_ascii=False)}\n"
        f"Python 参考程序：{generated.reference_solution}\n"
        f"C++ 参考程序：{generated.reference_solution_cpp}\n"
        f"需要 {target_count} 个互不重复的隐藏测试点，并保留至少一个答案准确、可人工核对的小样例。\n"
        "每个 input 必须是可直接传给 stdin 的完整字面数据且不超过 2000 字符；严禁省略号、"
        "GENERATE_*、placeholder、重复次数描述或任何待展开宏。矩阵、网格或长序列无法完整展开时，"
        "应改用规模较小但结构完整的输入。隐藏 output 无法可靠手算时可以留空，系统会用双解共识回填。\n"
        f"校验错误：{'; '.join(errors[:20])}"
    )


def _static_validation_errors(
    generated: GeneratedProblem,
    request: AIProblemTaskPayload,
    existing: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    target_count = _target_testcase_count(request, generated)
    if not 2 <= len(generated.problem.testcases) <= 10:
        errors.append("测试数据 隐藏测试点数量必须在 2 到 10 之间")
    elif len(generated.problem.testcases) < target_count:
        errors.append(f"测试数据 测试点少于要求的 {target_count} 个")
    inputs = [case.input for case in generated.problem.testcases]
    if len(set(inputs)) != len(inputs):
        errors.append("测试数据 隐藏测试点输入存在重复")
    if not generated.problem.samples:
        errors.append("测试数据 至少需要一个可人工核对的样例")
    elif not any(case.output.strip() for case in generated.problem.samples):
        errors.append("测试数据 至少一个样例必须提供非空的正确答案作为安全锚点")
    total_test_data_length = 0
    for kind, cases in (
        ("样例", generated.problem.samples),
        ("测试点", generated.problem.testcases),
    ):
        for index, case in enumerate(cases, start=1):
            total_test_data_length += len(case.input) + len(case.output)
            if len(case.input) > 2_000:
                errors.append(f"测试数据 {kind} {index} 输入超过 2000 字符，请改为紧凑数据")
            if len(case.output) > 4_000:
                errors.append(f"测试数据 {kind} {index} 输出过长，请简化题目或测试数据")
            placeholder = _test_input_placeholder_pattern.search(case.input)
            if placeholder:
                errors.append(
                    f"测试数据 {kind} {index} 含占位或省略内容 {placeholder.group(0)!r}；"
                    "input 必须是可以直接传给标准输入的完整字面数据"
                )
    if total_test_data_length > 16_000:
        errors.append("测试数据 总长度超过 16000 字符，请减少冗余")
    if existing and generated.problem.id != existing.get("id"):
        errors.append("修改已有题目时不得改变题目 id")
    return errors


async def _validate_generated(
    generated: GeneratedProblem,
    request: AIProblemTaskPayload,
    existing: dict[str, Any] | None,
) -> tuple[list[str], dict[str, Any]]:
    calibration: dict[str, Any] = {"applied": False, "count": 0, "items": []}
    errors = _static_validation_errors(generated, request, existing)
    if errors:
        return errors, calibration

    cases = [
        *(("sample", index, case) for index, case in enumerate(generated.problem.samples, 1)),
        *(("testcase", index, case) for index, case in enumerate(generated.problem.testcases, 1)),
    ]
    case_inputs = [case.input for _, _, case in cases]
    python_execution = await execute_reference_solution(
        generated.problem,
        generated.reference_solution,
        inputs=case_inputs,
    )
    cpp_execution = await execute_reference_solution(
        generated.problem,
        generated.reference_solution_cpp,
        language="cpp",
        inputs=case_inputs,
    )
    python_errors = _reference_execution_errors(python_execution, cases, "Python")
    cpp_errors = _reference_execution_errors(cpp_execution, cases, "C++")
    if python_errors or cpp_errors:
        if python_errors and cpp_errors:
            return [*python_errors, *cpp_errors], calibration
        return (python_errors or cpp_errors), calibration

    python_outputs = [normalize_output(result.stdout) for result in python_execution.results]
    cpp_outputs = [normalize_output(result.stdout) for result in cpp_execution.results]
    disagreements = [
        index
        for index, (python_output, cpp_output) in enumerate(
            zip(python_outputs, cpp_outputs, strict=True)
        )
        if python_output != cpp_output
    ]
    if disagreements:
        originals = [normalize_output(case.output) for _, _, case in cases]
        nonblank = [index for index, value in enumerate(originals) if value]
        python_matches = bool(nonblank) and all(
            originals[index] == python_outputs[index] for index in nonblank
        )
        cpp_matches = bool(nonblank) and all(
            originals[index] == cpp_outputs[index] for index in nonblank
        )
        if python_matches != cpp_matches:
            failing_language = "C++" if python_matches else "Python"
            failing_outputs = cpp_outputs if python_matches else python_outputs
            trusted_outputs = python_outputs if python_matches else cpp_outputs
            targeted_errors = []
            for case_index in disagreements:
                kind, index, case = cases[case_index]
                targeted_errors.append(
                    f"{failing_language} {_case_label(kind, index)}: 与另一语言和原答案不一致; "
                    f"input={case.input[:240]!r}; actual={failing_outputs[case_index][:240]!r}; "
                    f"trusted={trusted_outputs[case_index][:240]!r}"
                )
            return targeted_errors, calibration
        disagreement_errors = []
        for case_index in disagreements:
            kind, index, case = cases[case_index]
            disagreement_errors.append(
                f"双解输出不一致 {_case_label(kind, index)}; input={case.input[:240]!r}; "
                f"Python stdout={python_outputs[case_index][:240]!r}; "
                f"C++ stdout={cpp_outputs[case_index][:240]!r}"
            )
        return disagreement_errors, calibration

    originals = [normalize_output(case.output) for _, _, case in cases]
    anchor_exists = any(
        original and original == consensus
        for original, consensus in zip(originals, python_outputs, strict=True)
    )
    changes = [
        case_index
        for case_index, (original, consensus) in enumerate(
            zip(originals, python_outputs, strict=True)
        )
        if original != consensus
    ]
    if changes and not anchor_exists:
        diagnostics = []
        for case_index in changes[:6]:
            kind, index, case = cases[case_index]
            diagnostics.append(
                f"{_case_label(kind, index)} input={case.input[:240]!r}; "
                f"original={case.output[:240]!r}; consensus={python_outputs[case_index][:240]!r}"
            )
        return [
            "Python/C++ 输出一致，但不存在与双解结果一致的非空安全锚点，禁止自动回填；"
            + "; ".join(diagnostics)
        ], calibration

    oversized = [index for index in changes if len(python_outputs[index]) > 4_000]
    if oversized:
        kind, index, case = cases[oversized[0]]
        return [
            f"{_case_label(kind, index)} 双解输出超过 4000 字符，禁止自动回填; "
            f"input={case.input[:240]!r}"
        ], calibration

    items: list[dict[str, Any]] = []
    for case_index in changes:
        kind, index, case = cases[case_index]
        calibrated_output = python_outputs[case_index]
        items.append(
            {
                "kind": kind,
                "index": index,
                "input": case.input,
                "original_output": case.output,
                "calibrated_output": calibrated_output,
            }
        )
        case.output = calibrated_output
    calibration = {"applied": bool(items), "count": len(items), "items": items}
    return [], calibration


def _case_label(kind: str, index: int) -> str:
    return f"{'sample' if kind == 'sample' else 'testcase'} {index}"


def _reference_execution_errors(
    execution: ReferenceExecution,
    cases: list[tuple[str, int, Any]],
    language: str,
) -> list[str]:
    if execution.setup_error:
        return [f"{language} {execution.setup_error}"]
    errors: list[str] = []
    for (kind, index, case), result in zip(cases, execution.results, strict=True):
        if result.status == "OK":
            continue
        detail = result.stderr or result.stdout or result.status
        errors.append(
            f"{language} {_case_label(kind, index)}: {result.status}; "
            f"input={case.input[:240]!r}; stdout={result.stdout[:240]!r}; "
            f"error={detail[:240]!r}"
        )
    return errors


async def _run_problem_task(
    task_id: str,
    user_id: int,
    request: AIProblemTaskPayload,
    config: AIModelConfigPayload,
    config_name: str,
) -> None:
    usage = _empty_usage(config)
    usage.update({"model_config_name": config_name, "model": config.model, "calls": []})
    staged_file_problem_id: str | None = None
    try:
        existing: dict[str, Any] | None = None
        if request.problem_id:
            async with database.session_factory() as session:
                problem = await session.get(Problem, request.problem_id)
                if problem:
                    existing = problem_to_dict(problem)

        await _update_task(
            task_id,
            status="running",
            progress="正在分析知识点、难度和边界条件",
            progress_percent=10,
            usage=usage,
            result={"stream_stage": "需求分析", "stream_preview": ""},
        )
        analysis_prompt = (
            "请为下面的程序设计题命题需求制定简洁但严格的命题蓝图，重点分析知识点、"
            "预期算法、复杂度、边界条件和用于淘汰错误/低效算法的测试策略。\n"
            f"需求：{request.requirement}\n难度：{request.difficulty}\n"
            f"知识点：{request.knowledge_points}\n"
            f"测试点策略：{request.testcase_count or '按复杂度和运行时长自动决定 2-10 个'}"
        )
        blueprint, current_usage = await _provider_request_with_retry(
            config,
            [
                {"role": "system", "content": "你是严谨的算法课程命题专家。"},
                {"role": "user", "content": analysis_prompt},
            ],
            on_delta=_stream_preview_callback(task_id, "需求分析"),
        )
        _merge_stage_usage(usage, current_usage, config, "需求分析")
        await _record_model_usage(user_id, config_name, current_usage, config)

        await _update_task(
            task_id,
            progress="正在生成题面、参考解法和分层测试点",
            progress_percent=45,
            usage=usage,
            result={"stream_stage": "题面、解法与测试点", "stream_preview": ""},
        )
        content, current_usage = await _provider_request_with_retry(
            config,
            [
                {"role": "system", "content": "你必须只输出严格 JSON，所有测试答案必须正确。"},
                {"role": "user", "content": _generation_prompt(request, blueprint, existing)},
            ],
            on_delta=_stream_preview_callback(task_id, "题面、解法与测试点"),
            json_mode=True,
        )
        _merge_stage_usage(usage, current_usage, config, "题面、解法与测试点")
        await _record_model_usage(user_id, config_name, current_usage, config)

        await _update_task(
            task_id,
            progress="正在交叉运行 Python/C++ 参考解法并校准答案",
            progress_percent=75,
            usage=usage,
        )
        generated: GeneratedProblem | None = None
        repair_count = 0
        errors: list[str] = []
        calibration: dict[str, Any] = {"applied": False, "count": 0, "items": []}
        resource_limits: dict[str, Any] = {}
        difficulty_policy: dict[str, str] = {}
        candidate_content = content
        for validation_attempt in range(3):
            try:
                generated = GeneratedProblem.model_validate(_extract_json(candidate_content))
                difficulty_policy = _apply_difficulty_policy(generated, request)
                resource_limits = _apply_resource_limit_policy(generated, request)
                errors, calibration = await _validate_generated(generated, request, existing)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                errors = [str(exc)]
                calibration = {"applied": False, "count": 0, "items": []}
                resource_limits = {}
                difficulty_policy = {}
                generated = None
            if not errors:
                break
            if validation_attempt >= 2:
                raise RuntimeError("两次自动修正后仍未通过校验：" + "; ".join(errors[:20]))
            repair_count += 1
            validation_message = "; ".join(errors[:20])
            repair_language: str | None = None
            repair_test_data = generated is not None and all(
                error.startswith("测试数据 ") for error in errors
            )
            if generated is not None and all(error.startswith("Python ") for error in errors):
                repair_language = "Python"
            elif generated is not None and all(error.startswith("C++ ") for error in errors):
                repair_language = "C++"
            repair_stage = (
                f"定向修复测试数据 {repair_count}/2"
                if repair_test_data
                else f"定向修复 {repair_language} {repair_count}/2"
                if repair_language
                else f"自动修正 {repair_count}/2"
            )
            await _update_task(
                task_id,
                progress=f"初稿未通过校验，正在进行第 {repair_count}/2 次修正",
                progress_percent=80 + repair_count * 7,
                usage=usage,
                result={
                    "stream_stage": repair_stage,
                    "stream_preview": "",
                },
            )
            if repair_test_data and generated is not None:
                repair_prompt = _test_data_repair_prompt(
                    generated,
                    request,
                    errors=errors,
                )
                system_message = (
                    "你是 OJ 测试数据审查员，只能返回 samples 和 testcases 的严格 JSON 对象。"
                )
            elif repair_language and generated is not None:
                repair_prompt = _solution_repair_prompt(
                    generated,
                    language=repair_language,
                    errors=errors,
                )
                system_message = (
                    "你是 OJ 参考程序审查员，只能返回包含指定参考程序的严格 JSON 对象。"
                )
            else:
                repair_prompt = (
                    _generation_prompt(request, blueprint, existing)
                    + "\n\n上一稿未通过校验，请根据错误从头生成完整 JSON，"
                    "不要省略任何根字段。" + f"\n校验错误：{validation_message}"
                )
                system_message = "你是 OJ 题目质量审查员，只能返回完整、严格的 JSON 对象。"
            candidate_content, current_usage = await _provider_request_with_retry(
                config,
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": repair_prompt},
                ],
                on_delta=_stream_preview_callback(task_id, repair_stage),
                json_mode=True,
            )
            _merge_stage_usage(usage, current_usage, config, repair_stage)
            await _record_model_usage(user_id, config_name, current_usage, config)
            if repair_test_data and generated is not None:
                replacement = _extract_json(candidate_content)
                samples = replacement.get("samples")
                testcases = replacement.get("testcases")
                if not isinstance(samples, list) or not isinstance(testcases, list):
                    raise RuntimeError("测试数据定向修复未返回 samples 和 testcases 数组")
                merged = generated.model_dump(mode="json")
                merged["problem"]["samples"] = samples
                merged["problem"]["testcases"] = testcases
                candidate_content = json.dumps(merged, ensure_ascii=False)
            elif repair_language and generated is not None:
                replacement = _extract_json(candidate_content).get(
                    "reference_solution_cpp" if repair_language == "C++" else "reference_solution"
                )
                if not isinstance(replacement, str) or not replacement.strip():
                    raise RuntimeError(f"{repair_language} 定向修复未返回参考程序")
                if repair_language == "C++":
                    generated.reference_solution_cpp = replacement
                else:
                    generated.reference_solution = replacement
                candidate_content = json.dumps(
                    generated.model_dump(mode="json"), ensure_ascii=False
                )

        assert generated is not None
        original_problem_id = generated.problem.id
        assigned_problem_id = str(existing["id"]) if existing else await _allocate_ai_problem_id()
        if existing is None:
            staged_file_problem_id = assigned_problem_id
        generated.problem.id = assigned_problem_id
        generated.problem.author = config.model
        file_stress: dict[str, Any] = {
            "requested": stress_requested(request),
            "applied": False,
            "count": 0,
        }
        if file_stress["requested"]:
            await _update_task(
                task_id,
                progress="正在尝试生成文件压力测试点",
                progress_percent=96,
                usage=usage,
            )
            try:
                file_stress = await asyncio.wait_for(
                    build_optional_file_stress(generated, request), timeout=20
                )
            except TimeoutError:
                file_stress["reason"] = "文件压力点生成超时，已保留普通测试点"
        result = generated.model_dump(mode="json")
        result["validation"] = {
            "passed": True,
            "testcase_count": len(generated.problem.testcases),
            "reference_solution_executed": True,
            "reference_languages": ["python", "cpp"],
            "automatic_repair_used": repair_count > 0,
            "automatic_repair_count": repair_count,
            "output_calibration": calibration,
            "resource_limits": resource_limits,
            "difficulty": difficulty_policy,
            "file_stress": file_stress,
            "id_assignment": {
                "source": "existing" if existing else "automatic",
                "model_id": original_problem_id,
                "final_id": assigned_problem_id,
            },
        }
        await _update_task(
            task_id,
            status="completed",
            progress="命题完成，已通过双语言交叉校验",
            progress_percent=100,
            usage=usage,
            result=result,
        )
    except asyncio.CancelledError:
        if staged_file_problem_id:
            with contextlib.suppress(Exception):
                await remove_file_testcases(staged_file_problem_id)
        await _update_task(
            task_id,
            status="cancelled",
            progress="任务已中断",
            progress_percent=100,
            usage=usage,
        )
        raise
    except Exception as exc:
        if staged_file_problem_id:
            with contextlib.suppress(Exception):
                await remove_file_testcases(staged_file_problem_id)
        await _update_task(
            task_id,
            status="failed",
            progress="命题失败",
            progress_percent=100,
            usage=usage,
            error=str(exc),
        )
    finally:
        _running_tasks.pop(task_id, None)


def problem_to_dict(problem: Problem) -> dict[str, Any]:
    return {
        "id": problem.id,
        "title": problem.title,
        "description": problem.description,
        "input_description": problem.input_description,
        "output_description": problem.output_description,
        "samples": problem.samples,
        "constraints": problem.constraints,
        "testcases": problem.testcases,
        "hint": problem.hint,
        "source": problem.source,
        "tags": problem.tags,
        "time_limit": problem.time_limit,
        "memory_limit": problem.memory_limit,
        "author": problem.author,
        "difficulty": problem.difficulty,
    }


async def create_problem_task(user_id: int, request: AIProblemTaskPayload) -> AIProblemTask:
    _explicit_resource_limits(request.requirement)
    await _ensure_model_store_loaded()
    configs = _model_configs.get(user_id, {})
    config_name = request.model_config_name or _active_model_configs.get(user_id)
    config = configs.get(config_name or "")
    if config is None or config_name is None:
        raise ValueError("请先配置模型")
    task_id = f"ai-task-{uuid.uuid4().hex}"
    task = AIProblemTask(
        task_id=task_id,
        user_id=user_id,
        status="pending",
        progress="等待处理",
        progress_percent=0,
        usage={
            **_empty_usage(config),
            "model_config_name": config_name,
            "model": config.model,
        },
    )
    async with database.session_factory() as session:
        session.add(task)
        await session.commit()
    _running_tasks[task_id] = asyncio.create_task(
        _run_problem_task(task_id, user_id, request, config, config_name)
    )
    return task


async def cancel_problem_task(task_id: str) -> None:
    task = _running_tasks.get(task_id)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    else:
        await _update_task(
            task_id,
            status="cancelled",
            progress="任务已中断",
            progress_percent=100,
        )


async def cancel_all_ai_tasks(*, clear_configs: bool = False, purge_configs: bool = False) -> None:
    global _model_store_loaded
    tasks = list(_running_tasks.values())
    _running_tasks.clear()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task
    if clear_configs:
        _model_configs.clear()
        _active_model_configs.clear()
        _model_usage.clear()
        _reserved_ai_ids.clear()
        _model_store_loaded = True
    if purge_configs:
        path = _store_path()
        if path.exists():
            await asyncio.to_thread(path.unlink)
