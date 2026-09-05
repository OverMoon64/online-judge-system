from __future__ import annotations

import asyncio
import contextlib
import os
import re
import shlex
import signal
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
from sqlalchemy import select

from app import db as database
from app.config import get_settings
from app.db import Language, Problem, Submission, utc_now
from app.schemas import ProblemPayload
from app.testcase_files import load_file_testcases

_running_tasks: dict[int, asyncio.Task[None]] = {}
_unsafe_command_chars = re.compile(r"[;&|><`\n\r\x00]")
_placeholder = re.compile(r"\{([^{}]+)\}")


@dataclass(slots=True)
class ProcessResult:
    status: str
    returncode: int
    stdout: str
    stderr: str
    elapsed: float
    peak_memory_mb: float


@dataclass(slots=True)
class ReferenceExecution:
    language: str
    setup_error: str | None
    results: list[ProcessResult]


class OutputLimitExceeded(Exception):
    pass


def validate_language_commands(language: Language | Any) -> None:
    settings = get_settings()
    commands = [command for command in (language.compile_cmd, language.run_cmd) if command]
    if not commands or not language.run_cmd:
        raise ValueError("run_cmd is required")

    for command in commands:
        if _unsafe_command_chars.search(command):
            raise ValueError("command contains forbidden shell characters")
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError as exc:
            raise ValueError("command has invalid quoting") from exc
        if not tokens:
            raise ValueError("command must not be empty")
        placeholders = set(_placeholder.findall(command))
        if not placeholders.issubset({"src", "exe"}):
            raise ValueError("only {src} and {exe} placeholders are allowed")
        executable = tokens[0]
        if executable != "{exe}" and Path(executable).name not in settings.executable_allowlist:
            raise ValueError(f"executable '{Path(executable).name}' is not allowed")

    if language.compile_cmd and "{src}" not in language.compile_cmd:
        raise ValueError("compile_cmd must contain {src}")
    if "{src}" not in language.run_cmd and "{exe}" not in language.run_cmd:
        raise ValueError("run_cmd must contain {src} or {exe}")


def render_command(command: str, source: Path, executable: Path) -> list[str]:
    tokens = shlex.split(command, posix=True)
    return [
        token.replace("{src}", str(source)).replace("{exe}", str(executable)) for token in tokens
    ]


def normalize_output(value: str) -> str:
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[-1].rstrip():
        lines.pop()
    return "\n".join(line.rstrip() for line in lines)


async def _read_limited(stream: asyncio.StreamReader | None, limit: int) -> bytes:
    if stream is None:
        return b""
    data = bytearray()
    while True:
        chunk = await stream.read(65_536)
        if not chunk:
            return bytes(data)
        data.extend(chunk)
        if len(data) > limit:
            raise OutputLimitExceeded


def _rss_tree_mb(pid: int) -> float:
    try:
        parent = psutil.Process(pid)
        processes = [parent, *parent.children(recursive=True)]
        return sum(process.memory_info().rss for process in processes if process.is_running()) / (
            1024 * 1024
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def _kill_process_tree(pid: int) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        return
    with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                child.kill()
        parent.kill()


async def run_process(
    command: list[str],
    input_text: str,
    timeout_seconds: float,
    memory_limit_mb: int,
) -> ProcessResult:
    settings = get_settings()
    started = time.monotonic()
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
    except (FileNotFoundError, PermissionError, OSError) as exc:
        return ProcessResult("UNK", -1, "", str(exc), 0.0, 0.0)

    async def feed_stdin() -> None:
        if process.stdin is None:
            return
        try:
            process.stdin.write(input_text.encode("utf-8"))
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()

    stdin_task = asyncio.create_task(feed_stdin())
    stdout_task = asyncio.create_task(_read_limited(process.stdout, settings.max_output_bytes))
    stderr_task = asyncio.create_task(_read_limited(process.stderr, settings.max_output_bytes))
    forced_status: str | None = None
    peak_memory = 0.0

    finished_at = started
    try:
        deadline = started + timeout_seconds
        wait_task = asyncio.create_task(process.wait())
        while not wait_task.done():
            if stdout_task.done() and isinstance(stdout_task.exception(), OutputLimitExceeded):
                forced_status = "RE"
                break
            if stderr_task.done() and isinstance(stderr_task.exception(), OutputLimitExceeded):
                forced_status = "RE"
                break
            peak_memory = max(peak_memory, await asyncio.to_thread(_rss_tree_mb, process.pid))
            if peak_memory > memory_limit_mb:
                forced_status = "MLE"
                break
            if time.monotonic() >= deadline:
                forced_status = "TLE"
                break
            remaining = max(0.0, deadline - time.monotonic())
            try:
                await asyncio.wait_for(
                    asyncio.shield(wait_task),
                    timeout=min(0.01, remaining),
                )
            except asyncio.TimeoutError:
                pass

        if forced_status:
            await asyncio.to_thread(_kill_process_tree, process.pid)
        await wait_task
        finished_at = time.monotonic()
    except asyncio.CancelledError:
        await asyncio.to_thread(_kill_process_tree, process.pid)
        with contextlib.suppress(Exception):
            await process.wait()
        raise
    finally:
        with contextlib.suppress(Exception):
            await stdin_task

    output_error = False
    try:
        stdout_bytes = await stdout_task
    except OutputLimitExceeded:
        stdout_bytes = b""
        output_error = True
    try:
        stderr_bytes = await stderr_task
    except OutputLimitExceeded:
        stderr_bytes = b"output exceeded the configured limit"
        output_error = True

    elapsed = finished_at - started
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if output_error and not forced_status:
        forced_status = "RE"
    if not forced_status and elapsed > timeout_seconds:
        forced_status = "TLE"
    status = forced_status or ("OK" if process.returncode == 0 else "RE")
    return ProcessResult(status, process.returncode or 0, stdout, stderr, elapsed, peak_memory)


def _sanitize_message(message: str, temporary_directory: str | None = None) -> str:
    cleaned = message
    if temporary_directory:
        cleaned = cleaned.replace(temporary_directory, "<judge>")
    cleaned = cleaned.replace("\\", "/")
    return cleaned[:4000]


async def _judge_submission(submission_id: int) -> None:
    try:
        async with database.session_factory() as session:
            submission = await session.get(Submission, submission_id)
            if submission is None:
                return
            problem = await session.get(Problem, submission.problem_id)
            language = await session.get(Language, submission.language)
            if problem is None or language is None:
                submission.status = "error"
                submission.error_info = "problem or language no longer exists"
                await session.commit()
                return
            code = submission.code
            file_testcases = await load_file_testcases(problem)
            problem_data = {
                "time_limit": problem.time_limit or language.time_limit or 3.0,
                "memory_limit": problem.memory_limit or language.memory_limit or 128,
                "testcases": [
                    *list(problem.testcases),
                    *(
                        {
                            "input": case["input"],
                            "output": case["output"],
                            "source": "file",
                            "label": case["label"],
                        }
                        for case in file_testcases
                    ),
                ],
            }
            language_data = {
                "file_ext": language.file_ext,
                "compile_cmd": language.compile_cmd,
                "run_cmd": language.run_cmd,
            }

        with tempfile.TemporaryDirectory(prefix=f"oj-{submission_id}-") as temp_name:
            temp_dir = Path(temp_name)
            source = temp_dir / f"main{language_data['file_ext']}"
            executable = temp_dir / "program"
            await asyncio.to_thread(source.write_text, code, encoding="utf-8")

            compile_info: dict[str, str] | None = None
            if language_data["compile_cmd"]:
                compile_result = await run_process(
                    render_command(language_data["compile_cmd"], source, executable),
                    "",
                    get_settings().compile_timeout_seconds,
                    512,
                )
                if compile_result.status != "OK":
                    message = _sanitize_message(
                        compile_result.stderr or compile_result.status, temp_name
                    )
                    compile_info = {"result": "failed", "message": message}
                    details = [
                        {
                            "id": index,
                            "result": "CE",
                            "time": 0.0,
                            "memory": 0.0,
                            "source": testcase.get("source", "inline"),
                        }
                        for index, testcase in enumerate(problem_data["testcases"], start=1)
                    ]
                    await _store_result(
                        submission_id,
                        score=0,
                        counts=len(details) * 10,
                        compile_info=compile_info,
                        run_info=None,
                        error_info=message,
                        details=details,
                    )
                    return
                compile_info = {
                    "result": "success",
                    "message": _sanitize_message(compile_result.stderr, temp_name),
                }

            details: list[dict[str, Any]] = []
            score = 0
            first_error = ""
            run_command = render_command(language_data["run_cmd"], source, executable)
            for index, testcase in enumerate(problem_data["testcases"], start=1):
                result = await run_process(
                    run_command,
                    testcase["input"],
                    float(problem_data["time_limit"]),
                    int(problem_data["memory_limit"]),
                )
                outcome = result.status
                if outcome == "OK":
                    outcome = (
                        "AC"
                        if normalize_output(result.stdout) == normalize_output(testcase["output"])
                        else "WA"
                    )
                if outcome == "AC":
                    score += 10
                elif not first_error and outcome in {"RE", "TLE", "MLE", "UNK"}:
                    first_error = _sanitize_message(result.stderr or outcome, temp_name)
                details.append(
                    {
                        "id": index,
                        "result": outcome if outcome in {"AC", "WA", "TLE", "MLE", "RE"} else "UNK",
                        "time": round(result.elapsed, 4),
                        "memory": round(result.peak_memory_mb, 3),
                        "source": testcase.get("source", "inline"),
                    }
                )

            await _store_result(
                submission_id,
                score=score,
                counts=len(details) * 10,
                compile_info=compile_info,
                run_info={
                    "result": "finished",
                    "message": f"{len(details)} test cases finished",
                },
                error_info=first_error,
                details=details,
            )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # The task boundary must convert all engine failures to status=error.
        async with database.session_factory() as session:
            submission = await session.get(Submission, submission_id)
            if submission is not None:
                submission.status = "error"
                submission.error_info = _sanitize_message(str(exc)) or "unknown judge error"
                submission.updated_at = utc_now()
                await session.commit()
    finally:
        _running_tasks.pop(submission_id, None)


async def _store_result(
    submission_id: int,
    *,
    score: int,
    counts: int,
    compile_info: dict[str, Any] | None,
    run_info: dict[str, Any] | None,
    error_info: str,
    details: list[dict[str, Any]],
) -> None:
    async with database.session_factory() as session:
        submission = await session.get(Submission, submission_id)
        if submission is None:
            return
        submission.status = "success"
        submission.score = score
        submission.counts = counts
        submission.compile_info = compile_info
        submission.run_info = run_info
        submission.error_info = error_info
        submission.details = details
        submission.updated_at = utc_now()
        await session.commit()


def start_judging(submission_id: int) -> None:
    previous = _running_tasks.pop(submission_id, None)
    if previous and not previous.done():
        previous.cancel()
    _running_tasks[submission_id] = asyncio.create_task(_judge_submission(submission_id))


async def cancel_submission_task(submission_id: int) -> None:
    task = _running_tasks.pop(submission_id, None)
    if task and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def cancel_all_submission_tasks() -> None:
    tasks = list(_running_tasks.values())
    _running_tasks.clear()
    for task in tasks:
        task.cancel()
    for task in tasks:
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def wait_for_submission(submission_id: int, timeout: float = 10.0) -> None:
    task = _running_tasks.get(submission_id)
    if task:
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)


async def validate_reference_solution(
    problem: ProblemPayload, reference_solution: str, language: str = "python"
) -> list[str]:
    execution = await execute_reference_solution(problem, reference_solution, language)
    if execution.setup_error:
        return [execution.setup_error]
    errors: list[str] = []
    for index, (testcase, result) in enumerate(
        zip(problem.testcases, execution.results, strict=True), start=1
    ):
        if result.status != "OK":
            errors.append(f"testcase {index}: {result.status}")
        elif normalize_output(result.stdout) != normalize_output(testcase.output):
            errors.append(f"testcase {index}: reference solution output mismatch")
    return errors


async def execute_reference_solution(
    problem: ProblemPayload,
    reference_solution: str,
    language: str = "python",
    *,
    inputs: list[str] | None = None,
) -> ReferenceExecution:
    """Compile once and execute a reference solution for every supplied input."""

    if language not in {"python", "cpp"}:
        raise ValueError("reference solution language must be python or cpp")
    case_inputs = inputs if inputs is not None else [case.input for case in problem.testcases]
    with tempfile.TemporaryDirectory(prefix="oj-ai-validation-") as temp_name:
        source = Path(temp_name) / ("solution.cpp" if language == "cpp" else "solution.py")
        executable = Path(temp_name) / "solution"
        await asyncio.to_thread(source.write_text, reference_solution, encoding="utf-8")
        if language == "cpp":
            compile_result = await run_process(
                ["g++", "-std=c++14", "-O2", "-pipe", str(source), "-o", str(executable)],
                "",
                get_settings().compile_timeout_seconds,
                max(problem.memory_limit, 256),
            )
            if compile_result.status != "OK":
                return ReferenceExecution(
                    language,
                    "compile: "
                    + _sanitize_message(
                        compile_result.stderr or compile_result.status,
                        temp_name,
                    ),
                    [],
                )
            run_command = [str(executable)]
        else:
            run_command = ["python3", str(source)]

        results: list[ProcessResult] = []
        for input_text in case_inputs:
            result = await run_process(
                run_command,
                input_text,
                problem.time_limit,
                problem.memory_limit,
            )
            result.stderr = _sanitize_message(result.stderr, temp_name)
            results.append(result)
        return ReferenceExecution(language, None, results)


async def submission_exists(submission_id: int) -> bool:
    async with database.session_factory() as session:
        return (
            await session.scalar(select(Submission.id).where(Submission.id == submission_id))
        ) is not None
