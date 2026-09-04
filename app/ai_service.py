from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import json
import re
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from app import db as database
from app.config import get_settings
from app.db import AIProblemTask, Problem, utc_now
from app.judge import validate_reference_solution
from app.schemas import AIModelConfigPayload, AIProblemTaskPayload, GeneratedProblem

_model_configs: dict[int, AIModelConfigPayload] = {}
_running_tasks: dict[str, asyncio.Task[None]] = {}


def _public_config(config: AIModelConfigPayload) -> dict[str, Any]:
    return {
        "provider_url": config.provider_url,
        "model": config.model,
        "api_key_configured": True,
        "input_price": config.input_price,
        "output_price": config.output_price,
        "price_unit": config.price_unit,
        "currency": config.currency,
    }


def set_model_config(user_id: int, config: AIModelConfigPayload) -> dict[str, Any]:
    parsed = urlparse(config.provider_url)
    host = parsed.hostname or ""
    local_host = host.lower() in {"localhost", "localhost.localdomain"}
    with contextlib.suppress(ValueError):
        local_host = local_host or ipaddress.ip_address(host).is_private
    if (parsed.scheme != "https" or local_host) and not get_settings().allow_local_ai:
        raise ValueError("provider URL must use public HTTPS unless OJ_ALLOW_LOCAL_AI=true")
    _model_configs[user_id] = config
    return _public_config(config)


def get_model_config(user_id: int) -> dict[str, Any] | None:
    config = _model_configs.get(user_id)
    return _public_config(config) if config else None


def _completion_endpoint(base_url: str) -> str:
    return base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"


async def _provider_request(
    config: AIModelConfigPayload, messages: list[dict[str, str]]
) -> tuple[str, dict[str, int]]:
    headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": 0.35,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=15.0)) as client:
            response = await client.post(
                _completion_endpoint(config.provider_url), headers=headers, json=payload
            )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        usage = body.get("usage") or {}
        return str(content), {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        }
    except httpx.TimeoutException as exc:
        raise RuntimeError("模型服务请求超时") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"模型服务返回 HTTP {exc.response.status_code}") from exc
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("模型服务返回了不兼容的响应格式") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("无法连接模型服务") from exc


def _empty_usage(config: AIModelConfigPayload) -> dict[str, Any]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "currency": config.currency,
        "price_unit": config.price_unit,
        "estimated": False,
    }


def _merge_usage(
    total: dict[str, Any], current: dict[str, int], config: AIModelConfigPayload
) -> None:
    total["input_tokens"] += current.get("input_tokens", 0)
    total["output_tokens"] += current.get("output_tokens", 0)
    total["total_tokens"] = total["input_tokens"] + total["output_tokens"]
    total["cost"] = round(
        total["input_tokens"] / config.price_unit * config.input_price
        + total["output_tokens"] / config.price_unit * config.output_price,
        8,
    )
    total["estimated"] = total["total_tokens"] == 0


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


def _extract_json(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("模型响应中没有 JSON 对象")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("模型响应根节点必须是对象")
    return value


def _generation_prompt(
    request: AIProblemTaskPayload, blueprint: str, existing: dict[str, Any] | None
) -> str:
    existing_text = json.dumps(existing, ensure_ascii=False) if existing else "无"
    return f"""
你是程序设计训练课程的严谨命题专家。请根据命题需求和分析草案生成一道可直接导入 OJ 的题目。

命题需求：{request.requirement}
知识点：{", ".join(request.knowledge_points) or "由需求推断"}
难度：{request.difficulty}
至少生成 {request.testcase_count} 个互不重复的测试点，必须包含最小值、最大值、特殊结构和能区分低效算法的数据。
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
  "reference_solution": "完整 Python 3 程序",
  "solution_explanation": "正确性和复杂度说明"
}}
所有 output 必须与 reference_solution 对应，且代码只能使用 Python 标准库。
""".strip()


async def _validate_generated(
    generated: GeneratedProblem,
    request: AIProblemTaskPayload,
    existing: dict[str, Any] | None,
) -> list[str]:
    errors: list[str] = []
    if len(generated.problem.testcases) < request.testcase_count:
        errors.append(f"测试点少于要求的 {request.testcase_count} 个")
    inputs = [case.input for case in generated.problem.testcases]
    if len(set(inputs)) != len(inputs):
        errors.append("测试点输入存在重复")
    if existing and generated.problem.id != existing.get("id"):
        errors.append("修改已有题目时不得改变题目 id")
    errors.extend(
        await validate_reference_solution(generated.problem, generated.reference_solution)
    )
    return errors


async def _run_problem_task(
    task_id: str,
    user_id: int,
    request: AIProblemTaskPayload,
    config: AIModelConfigPayload,
) -> None:
    usage = _empty_usage(config)
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
        )
        analysis_prompt = (
            "请为下面的程序设计题命题需求制定简洁但严格的命题蓝图，重点分析知识点、"
            "预期算法、复杂度、边界条件和用于淘汰错误/低效算法的测试策略。\n"
            f"需求：{request.requirement}\n难度：{request.difficulty}\n"
            f"知识点：{request.knowledge_points}\n测试点数量：{request.testcase_count}"
        )
        blueprint, current_usage = await _provider_request(
            config,
            [
                {"role": "system", "content": "你是严谨的算法课程命题专家。"},
                {"role": "user", "content": analysis_prompt},
            ],
        )
        _merge_usage(usage, current_usage, config)

        await _update_task(
            task_id,
            progress="正在生成题面、参考解法和分层测试点",
            progress_percent=45,
            usage=usage,
        )
        content, current_usage = await _provider_request(
            config,
            [
                {"role": "system", "content": "你必须只输出严格 JSON，所有测试答案必须正确。"},
                {"role": "user", "content": _generation_prompt(request, blueprint, existing)},
            ],
        )
        _merge_usage(usage, current_usage, config)

        await _update_task(
            task_id,
            progress="正在校验字段、测试点和参考解法",
            progress_percent=75,
            usage=usage,
        )
        validation_message = ""
        try:
            generated = GeneratedProblem.model_validate(_extract_json(content))
            errors = await _validate_generated(generated, request, existing)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            errors = [str(exc)]
            generated = None

        if errors:
            validation_message = "; ".join(errors[:20])
            await _update_task(
                task_id,
                progress="初稿未通过校验，正在进行一次自动修正",
                progress_percent=85,
                usage=usage,
            )
            repair_prompt = (
                "请修正下面的 OJ 题目 JSON。只返回完整、严格 JSON，不要解释。\n"
                f"校验错误：{validation_message}\n原始内容：{content}"
            )
            repaired, current_usage = await _provider_request(
                config,
                [
                    {"role": "system", "content": "你是 OJ 题目质量审查员。"},
                    {"role": "user", "content": repair_prompt},
                ],
            )
            _merge_usage(usage, current_usage, config)
            generated = GeneratedProblem.model_validate(_extract_json(repaired))
            errors = await _validate_generated(generated, request, existing)
            if errors:
                raise RuntimeError("自动修正后仍未通过校验：" + "; ".join(errors[:20]))

        assert generated is not None
        result = generated.model_dump(mode="json")
        result["validation"] = {
            "passed": True,
            "testcase_count": len(generated.problem.testcases),
            "reference_solution_executed": True,
            "automatic_repair_used": bool(validation_message),
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
    config = _model_configs.get(user_id)
    if config is None:
        raise ValueError("请先配置模型")
    task_id = f"ai-task-{uuid.uuid4().hex}"
    task = AIProblemTask(
        task_id=task_id,
        user_id=user_id,
        status="pending",
        progress="等待处理",
        progress_percent=0,
        usage=_empty_usage(config),
    )
    async with database.session_factory() as session:
        session.add(task)
        await session.commit()
    _running_tasks[task_id] = asyncio.create_task(
        _run_problem_task(task_id, user_id, request, config)
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


async def cancel_all_ai_tasks(*, clear_configs: bool = False) -> None:
    tasks = list(_running_tasks.values())
    _running_tasks.clear()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task
    if clear_configs:
        _model_configs.clear()
