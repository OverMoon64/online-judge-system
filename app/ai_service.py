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

from app import db as database
from app.config import get_settings
from app.db import AIProblemTask, Problem, utc_now
from app.judge import validate_reference_solution
from app.schemas import AIModelConfigPayload, AIProblemTaskPayload, GeneratedProblem

_model_configs: dict[int, dict[str, AIModelConfigPayload]] = {}
_active_model_configs: dict[int, str] = {}
_model_usage: dict[int, dict[str, dict[str, Any]]] = {}
_model_store_path_override: Path | None = None
_model_store_loaded = False
_model_store_lock: asyncio.Lock | None = None
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
        "price_source": config.price_source,
        "price_source_url": config.price_source_url,
        "pricing_note": config.pricing_note,
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
                        name: AIModelConfigPayload.model_validate(config)
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


def suggest_model_pricing(provider_url: str, model: str) -> dict[str, Any]:
    host = (urlparse(provider_url).hostname or "").lower()
    normalized_model = model.strip().lower()
    if "aliyuncs.com" not in host:
        raise ValueError("该 OpenAI-compatible 服务没有可识别的公开计价表，请手动填写")

    international = "dashscope-intl" in host
    source_url = "https://help.aliyun.com/zh/model-studio/model-pricing"
    common = {
        "price_unit": 1_000_000,
        "currency": "CNY",
        "price_source": "阿里云百炼官方模型价格",
        "price_source_url": source_url,
        "as_of": "2026-09-04",
    }
    if normalized_model.startswith("qwen3.7-plus"):
        if international:
            input_price, output_price = 2.998, 11.991
            region = "国际站"
        elif normalized_model == "qwen3.7-plus":
            input_price, output_price = 1.6, 6.4
            region = "中国站限时折扣"
        else:
            input_price, output_price = 2.0, 8.0
            region = "中国站快照版"
        tier = "单次请求输入不超过 256K Token"
    elif normalized_model.startswith("qwen3.7-flash"):
        if international:
            input_price, output_price = 0.225, 0.974
            region = "国际站"
        else:
            input_price, output_price = 0.2, 0.8
            region = "中国站"
        tier = "单次请求输入不超过 32K Token"
    else:
        raise ValueError("该模型暂无内置官方价格预设，请按服务商控制台手动填写")
    return {
        **common,
        "input_price": input_price,
        "output_price": output_price,
        "pricing_note": f"{region}，{tier}；阶梯、缓存和后续调价请以官方控制台为准。",
    }


async def _provider_request(
    config: AIModelConfigPayload,
    messages: list[dict[str, str]],
    *,
    on_delta: StreamCallback | None = None,
) -> tuple[str, dict[str, Any]]:
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.35,
    }
    provider_host = (urlparse(config.provider_url).hostname or "").lower()
    if (
        config.disable_thinking
        and config.model.lower().startswith("qwen")
        and "aliyuncs.com" in provider_host
    ):
        payload["enable_thinking"] = False
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
) -> tuple[str, dict[str, Any]]:
    for attempt in range(retry_count + 1):
        try:
            content, usage = await _provider_request(config, messages, on_delta=on_delta)
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
    content = body["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content:
        raise ValueError("model response content is empty")
    return content, _token_usage(body.get("usage"))


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
        delta = _delta_text((choices[0].get("delta") or {}).get("content"))
        if delta:
            content_parts.append(delta)
            if on_delta:
                await on_delta(delta)
    if not received_event:
        return _parse_non_stream_response(json.loads("\n".join(raw_lines)))
    content = "".join(content_parts)
    if not content:
        raise ValueError("model stream content is empty")
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


def _generation_prompt(
    request: AIProblemTaskPayload, blueprint: str, existing: dict[str, Any] | None
) -> str:
    existing_text = json.dumps(existing, ensure_ascii=False) if existing else "无"
    testcase_requirement = (
        f"生成至少 {request.testcase_count} 个互不重复的隐藏测试点"
        if request.testcase_count
        else "根据算法复杂度、边界规模和预计运行时长，生成 2 到 10 个互不重复的隐藏测试点"
    )
    return f"""
你是程序设计训练课程的严谨命题专家。请根据命题需求和分析草案生成一道可直接导入 OJ 的题目。

命题需求：{request.requirement}
知识点：{", ".join(request.knowledge_points) or "由需求推断"}
难度：{request.difficulty}
{testcase_requirement}，必须包含最小值、最大值、特殊结构和能区分低效算法的数据。
参考已有题目：{existing_text}
分析草案：{blueprint}

只返回一个 JSON 对象，禁止 Markdown。格式必须为：
{{
  "problem": {{
    "id": "唯一英文或数字编号", "title": "标题", "description": "题面",
    "input_description": "输入格式", "output_description": "输出格式",
    "samples": [{{"input": "...", "output": "..."}}],
    "constraints": "数据范围", "testcases": [{{"input": "...", "output": "..."}}],
    "hint": "", "source": "AI 辅助命题", "tags": ["知识点"],
    "time_limit": 1.0, "memory_limit": 128, "author": "AI Assistant",
    "difficulty": "{request.difficulty}"
  }},
    "reference_solution": "完整 Python 3 程序，换行必须写成 JSON 转义字符 \\n",
    "reference_solution_cpp": "完整 C++14 程序，换行必须写成 JSON 转义字符 \\n",
  "solution_explanation": "正确性和复杂度说明"
}}
两个参考程序必须使用相同算法并对所有 output 给出一致结果；Python 只能使用标准库，C++ 使用 C++14。
所有字符串中的换行、制表符和控制字符都必须按 JSON 规则转义，禁止输出字面控制字符。
""".strip()


async def _validate_generated(
    generated: GeneratedProblem,
    request: AIProblemTaskPayload,
    existing: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if not 2 <= len(generated.problem.testcases) <= 10:
        errors.append("隐藏测试点数量必须在 2 到 10 之间")
    elif request.testcase_count and len(generated.problem.testcases) < request.testcase_count:
        errors.append(f"测试点少于要求的 {request.testcase_count} 个")
    inputs = [case.input for case in generated.problem.testcases]
    if len(set(inputs)) != len(inputs):
        errors.append("测试点输入存在重复")
    if existing and generated.problem.id != existing.get("id"):
        errors.append("修改已有题目时不得改变题目 id")
    errors.extend(
        await validate_reference_solution(generated.problem, generated.reference_solution)
    )
    cpp_errors = await validate_reference_solution(
        generated.problem,
        generated.reference_solution_cpp,
        language="cpp",
    )
    errors.extend(f"C++ {error}" for error in cpp_errors)
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
        )
        _merge_stage_usage(usage, current_usage, config, "题面、解法与测试点")
        await _record_model_usage(user_id, config_name, current_usage, config)

        await _update_task(
            task_id,
            progress="正在校验字段、测试点和参考解法",
            progress_percent=75,
            usage=usage,
        )
        generated: GeneratedProblem | None = None
        repair_count = 0
        errors: list[str] = []
        candidate_content = content
        for validation_attempt in range(3):
            try:
                generated = GeneratedProblem.model_validate(_extract_json(candidate_content))
                errors = await _validate_generated(generated, request, existing)
            except (ValidationError, ValueError, json.JSONDecodeError) as exc:
                errors = [str(exc)]
                generated = None
            if not errors:
                break
            if validation_attempt >= 2:
                raise RuntimeError("两次自动修正后仍未通过校验：" + "; ".join(errors[:20]))
            repair_count += 1
            validation_message = "; ".join(errors[:20])
            await _update_task(
                task_id,
                progress=f"初稿未通过校验，正在进行第 {repair_count}/2 次自动修正",
                progress_percent=80 + repair_count * 7,
                usage=usage,
                result={
                    "stream_stage": f"自动修正 {repair_count}/2",
                    "stream_preview": "",
                },
            )
            repair_prompt = (
                _generation_prompt(request, blueprint, existing)
                + "\n\n上一稿未通过校验，请根据错误从头生成完整 JSON，不要省略任何根字段。"
                + f"\n校验错误：{validation_message}"
            )
            candidate_content, current_usage = await _provider_request_with_retry(
                config,
                [
                    {
                        "role": "system",
                        "content": "你是 OJ 题目质量审查员，只能返回完整、严格的 JSON 对象。",
                    },
                    {"role": "user", "content": repair_prompt},
                ],
                on_delta=_stream_preview_callback(task_id, f"自动修正 {repair_count}/2"),
            )
            _merge_stage_usage(usage, current_usage, config, f"自动修正 {repair_count}/2")
            await _record_model_usage(user_id, config_name, current_usage, config)

        assert generated is not None
        generated.problem.author = config.model
        result = generated.model_dump(mode="json")
        result["validation"] = {
            "passed": True,
            "testcase_count": len(generated.problem.testcases),
            "reference_solution_executed": True,
            "reference_languages": ["python", "cpp"],
            "automatic_repair_used": repair_count > 0,
            "automatic_repair_count": repair_count,
        }
        await _update_task(
            task_id,
            status="completed",
            progress="命题完成，已通过结构和参考解法校验",
            progress_percent=100,
            usage=usage,
            result=result,
        )
    except asyncio.CancelledError:
        await _update_task(
            task_id,
            status="cancelled",
            progress="任务已中断",
            progress_percent=100,
            usage=usage,
        )
        raise
    except Exception as exc:
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
        _model_store_loaded = True
    if purge_configs:
        path = _store_path()
        if path.exists():
            await asyncio.to_thread(path.unlink)
