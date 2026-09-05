from __future__ import annotations

import secrets
from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Request
from sqlalchemy import delete, distinct, func, select
from sqlalchemy.exc import IntegrityError

from app import ai_service
from app.ai_service import problem_to_dict
from app.config import get_settings
from app.db import (
    AccessLog,
    AIProblemTask,
    Language,
    Problem,
    Submission,
    User,
    reset_database,
    utc_now,
)
from app.dependencies import AdminUser, CurrentUser, SessionDep
from app.errors import (
    ApiError,
    parse_body,
    parse_pagination,
    parse_resource_id,
    success,
)
from app.judge import (
    cancel_all_submission_tasks,
    cancel_submission_task,
    start_judging,
    validate_language_commands,
)
from app.schemas import (
    AIModelConfigPayload,
    AIProblemTaskPayload,
    ChangePasswordPayload,
    LanguagePayload,
    LoginPayload,
    LogVisibilityPayload,
    ProblemPayload,
    RegisterPayload,
    RolePayload,
    SubmissionPayload,
)
from app.security import hash_password, verify_password
from app.testcase_files import (
    clear_testcase_store,
    count_file_testcases,
    problem_fingerprint,
    remove_file_testcases,
)

router = APIRouter()


async def _problem_data(problem: Problem) -> dict[str, Any]:
    data = problem_to_dict(problem)
    data["public_cases"] = problem.public_cases
    data["file_testcase_count"] = await count_file_testcases(problem)
    return data


async def _user_data(session: SessionDep, user: User) -> dict[str, Any]:
    submit_count = await session.scalar(
        select(func.count(Submission.id)).where(Submission.user_id == user.id)
    )
    resolve_count = await session.scalar(
        select(func.count(distinct(Submission.problem_id))).where(
            Submission.user_id == user.id,
            Submission.status == "success",
            Submission.counts > 0,
            Submission.score == Submission.counts,
        )
    )
    return {
        "user_id": str(user.id),
        "username": user.username,
        "join_time": user.join_time.date().isoformat(),
        "role": user.role,
        "submit_count": int(submit_count or 0),
        "resolve_count": int(resolve_count or 0),
    }


def _submission_result(submission: Submission) -> str:
    if submission.status == "pending":
        return "pending"
    if submission.status == "error":
        return "UNK"

    supported_results = {"AC", "WA", "TLE", "MLE", "RE", "CE", "UNK"}
    details = submission.details or []
    for case in details:
        result = str(case.get("result", "UNK")).upper()
        if result != "AC":
            return result if result in supported_results else "UNK"
    if details:
        return "AC"
    if submission.compile_info and submission.compile_info.get("result") == "failed":
        return "CE"
    if submission.counts and submission.score == submission.counts:
        return "AC"
    return "UNK"


def _submission_detail(submission: Submission, *, include_code: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "submission_id": str(submission.id),
        "status": submission.status,
        "problem_id": submission.problem_id,
        "language": submission.language,
        "created_at": submission.created_at.isoformat(),
        "result": _submission_result(submission),
    }
    if include_code:
        data["code"] = submission.code
    if submission.status != "pending":
        data.update(
            {
                "score": submission.score,
                "counts": submission.counts,
                "compile_info": submission.compile_info,
                "run_info": submission.run_info,
                "error_info": submission.error_info or "",
            }
        )
    return data


def _ai_task_data(task: AIProblemTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "progress_percent": task.progress_percent,
        "result": task.result,
        "usage": task.usage,
        "error": task.error,
    }


@router.get("/")
async def root() -> dict[str, Any]:
    return success({"service": "Async Online Judge", "docs": "/docs"})


@router.get("/health")
async def health() -> dict[str, Any]:
    return success({"status": "ok"})


@router.post("/api/users/")
async def register(request: Request, session: SessionDep) -> dict[str, Any]:
    payload = await parse_body(request, RegisterPayload)
    existing = await session.scalar(select(User).where(User.username == payload.username))
    if existing:
        raise ApiError(400, "username already exists")
    user = User(
        username=payload.username,
        password_hash=await hash_password(payload.password),
        role="user",
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise ApiError(400, "username already exists") from exc
    await session.refresh(user)
    return success(await _user_data(session, user), "register success")


@router.post("/api/users/admin")
async def create_admin(request: Request, session: SessionDep, _: AdminUser) -> dict[str, Any]:
    payload = await parse_body(request, RegisterPayload)
    existing = await session.scalar(select(User).where(User.username == payload.username))
    if existing:
        raise ApiError(400, "username already exists")
    user = User(
        username=payload.username,
        password_hash=await hash_password(payload.password),
        role="admin",
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return success({"user_id": str(user.id), "username": user.username})


@router.post("/api/auth/login")
async def login(request: Request, session: SessionDep) -> dict[str, Any]:
    payload = await parse_body(request, LoginPayload)
    user = await session.scalar(select(User).where(User.username == payload.username))
    if user is None or not await verify_password(payload.password, user.password_hash):
        raise ApiError(401, "invalid username or password")
    if user.role == "banned":
        raise ApiError(403, "user is banned")
    request.session.clear()
    user.session_token = secrets.token_urlsafe(32)
    request.session["user_id"] = user.id
    request.session["session_token"] = user.session_token
    await session.commit()
    return success(
        {"user_id": str(user.id), "username": user.username, "role": user.role},
        "login success",
    )


@router.post("/api/auth/logout")
async def logout(
    request: Request, session: SessionDep, current_user: CurrentUser
) -> dict[str, Any]:
    current_user.session_token = None
    await session.commit()
    request.session.clear()
    return success(None, "logout success")


@router.get("/api/auth/session")
async def get_login_session(session: SessionDep, current_user: CurrentUser) -> dict[str, Any]:
    return success(
        {
            "user_id": str(current_user.id),
            "username": current_user.username,
            "role": current_user.role,
        }
    )


@router.get("/api/users/")
async def list_users(request: Request, session: SessionDep, _: AdminUser) -> dict[str, Any]:
    page, page_size = parse_pagination(
        request.query_params.get("page"), request.query_params.get("page_size")
    )
    total = int(await session.scalar(select(func.count(User.id))) or 0)
    statement = select(User).order_by(User.id)
    if page_size:
        statement = statement.offset((page - 1) * page_size).limit(page_size)
    users = list((await session.scalars(statement)).all())
    return success({"total": total, "users": [await _user_data(session, user) for user in users]})


@router.get("/api/users/{user_id}")
async def get_user(user_id: str, session: SessionDep, current_user: CurrentUser) -> dict[str, Any]:
    parsed_id = parse_resource_id(user_id, "user_id")
    if current_user.role != "admin" and current_user.id != parsed_id:
        raise ApiError(403, "permission denied")
    user = await session.get(User, parsed_id)
    if user is None:
        raise ApiError(404, "user not found")
    return success(await _user_data(session, user))


@router.put("/api/users/{user_id}/password")
async def change_password(
    user_id: str, request: Request, session: SessionDep, current_user: CurrentUser
) -> dict[str, Any]:
    parsed_id = parse_resource_id(user_id, "user_id")
    if current_user.id != parsed_id:
        raise ApiError(403, "permission denied")
    payload = await parse_body(request, ChangePasswordPayload)
    if not await verify_password(payload.current_password, current_user.password_hash):
        raise ApiError(400, "invalid current password")
    if payload.current_password == payload.new_password:
        raise ApiError(400, "new password must be different")
    current_user.password_hash = await hash_password(payload.new_password)
    await session.commit()
    return success(None, "password updated")


@router.put("/api/users/{user_id}/role")
async def update_user_role(
    user_id: str, request: Request, session: SessionDep, _: AdminUser
) -> dict[str, Any]:
    parsed_id = parse_resource_id(user_id, "user_id")
    payload = await parse_body(request, RolePayload)
    user = await session.get(User, parsed_id)
    if user is None:
        raise ApiError(404, "user not found")
    if user.role == "admin" and payload.role != "admin":
        admin_count = int(
            await session.scalar(select(func.count(User.id)).where(User.role == "admin")) or 0
        )
        if admin_count <= 1:
            raise ApiError(409, "at least one admin is required")
    user.role = payload.role
    await session.commit()
    return success({"user_id": str(user.id), "role": user.role}, "role updated")


@router.get("/api/problems/")
async def list_problems(session: SessionDep, _: CurrentUser) -> dict[str, Any]:
    problems = list((await session.scalars(select(Problem).order_by(Problem.id))).all())
    return success([{"id": problem.id, "title": problem.title} for problem in problems])


@router.post("/api/problems/")
async def add_problem(request: Request, session: SessionDep, _: CurrentUser) -> dict[str, Any]:
    payload = await parse_body(request, ProblemPayload)
    if await session.get(Problem, payload.id):
        raise ApiError(409, "problem id already exists")
    problem = Problem(**payload.model_dump())
    session.add(problem)
    await session.commit()
    if await count_file_testcases(problem) == 0:
        await remove_file_testcases(problem.id)
    return success({"id": problem.id}, "add success")


@router.get("/api/problems/{problem_id}")
async def get_problem(problem_id: str, session: SessionDep, _: CurrentUser) -> dict[str, Any]:
    problem = await session.get(Problem, problem_id)
    if problem is None:
        raise ApiError(404, "problem not found")
    return success(await _problem_data(problem))


@router.put("/api/problems/{problem_id}")
async def edit_problem(
    problem_id: str, request: Request, session: SessionDep, _: CurrentUser
) -> dict[str, Any]:
    payload = await parse_body(request, ProblemPayload)
    if payload.id != problem_id:
        raise ApiError(400, "path id and body id do not match")
    problem = await session.get(Problem, problem_id)
    if problem is None:
        raise ApiError(404, "problem not found")
    previous_fingerprint = problem_fingerprint(problem)
    for key, value in payload.model_dump().items():
        setattr(problem, key, value)
    await session.commit()
    if problem_fingerprint(problem) != previous_fingerprint:
        await remove_file_testcases(problem_id)
    return success({"id": problem.id}, "update success")


@router.delete("/api/problems/{problem_id}")
async def delete_problem(problem_id: str, session: SessionDep, _: AdminUser) -> dict[str, Any]:
    problem = await session.get(Problem, problem_id)
    if problem is None:
        raise ApiError(404, "problem not found")
    submission_ids = list(
        (
            await session.scalars(select(Submission.id).where(Submission.problem_id == problem_id))
        ).all()
    )
    for submission_id in submission_ids:
        await cancel_submission_task(submission_id)
    await session.execute(delete(AccessLog).where(AccessLog.problem_id == problem_id))
    await session.delete(problem)
    await session.commit()
    await remove_file_testcases(problem_id)
    return success({"id": problem_id}, "delete success")


@router.put("/api/problems/{problem_id}/log_visibility")
async def update_log_visibility(
    problem_id: str, request: Request, session: SessionDep, _: AdminUser
) -> dict[str, Any]:
    payload = await parse_body(request, LogVisibilityPayload)
    problem = await session.get(Problem, problem_id)
    if problem is None:
        raise ApiError(404, "problem not found")
    problem.public_cases = payload.public_cases
    await session.commit()
    return success(
        {"problem_id": problem_id, "public_cases": problem.public_cases},
        "log visibility updated",
    )


@router.get("/api/languages/")
async def list_languages(session: SessionDep, _: CurrentUser) -> dict[str, Any]:
    names = list((await session.scalars(select(Language.name).order_by(Language.name))).all())
    return success({"name": names})


@router.post("/api/languages/")
async def add_language(request: Request, session: SessionDep, _: CurrentUser) -> dict[str, Any]:
    payload = await parse_body(request, LanguagePayload)
    try:
        validate_language_commands(payload)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    if await session.get(Language, payload.name):
        raise ApiError(409, "language already exists")
    session.add(Language(**payload.model_dump()))
    await session.commit()
    return success({"name": payload.name}, "language registered")


@router.post("/api/submissions/")
async def create_submission(
    request: Request, session: SessionDep, current_user: CurrentUser
) -> dict[str, Any]:
    payload = await parse_body(request, SubmissionPayload)
    cutoff = utc_now() - timedelta(minutes=1)
    recent_count = int(
        await session.scalar(
            select(func.count(Submission.id)).where(
                Submission.user_id == current_user.id,
                Submission.created_at >= cutoff,
            )
        )
        or 0
    )
    if recent_count >= 3:
        raise ApiError(429, "submission rate limit exceeded")
    if await session.get(Problem, payload.problem_id) is None:
        raise ApiError(404, "problem not found")
    if await session.get(Language, payload.language) is None:
        raise ApiError(404, "language not found")
    submission = Submission(
        user_id=current_user.id,
        problem_id=payload.problem_id,
        language=payload.language,
        code=payload.code,
        status="pending",
    )
    session.add(submission)
    await session.commit()
    await session.refresh(submission)
    start_judging(submission.id)
    return success({"submission_id": str(submission.id), "status": "pending"})


@router.get("/api/submissions/")
async def list_submissions(
    request: Request, session: SessionDep, current_user: CurrentUser
) -> dict[str, Any]:
    query = request.query_params
    user_value = query.get("user_id")
    problem_id = query.get("problem_id")
    status = query.get("status")
    if not user_value and not problem_id:
        raise ApiError(400, "user_id or problem_id is required")

    user_id: int | None = None
    if user_value:
        user_id = parse_resource_id(user_value, "user_id")
        if current_user.role != "admin" and user_id != current_user.id:
            # Permission has a higher documented priority than secondary
            # filter validation and resource existence checks.
            raise ApiError(403, "permission denied")
    if status is not None and status not in {"pending", "success", "error"}:
        raise ApiError(400, "invalid status")
    page, page_size = parse_pagination(query.get("page"), query.get("page_size"))

    conditions: list[Any] = []
    if user_id is not None:
        if await session.get(User, user_id) is None:
            raise ApiError(404, "user not found")
        conditions.append(Submission.user_id == user_id)
    elif current_user.role != "admin":
        conditions.append(Submission.user_id == current_user.id)

    if problem_id:
        if await session.get(Problem, problem_id) is None:
            raise ApiError(404, "problem not found")
        conditions.append(Submission.problem_id == problem_id)
    if status:
        conditions.append(Submission.status == status)

    total = int(await session.scalar(select(func.count(Submission.id)).where(*conditions)) or 0)
    statement = select(Submission).where(*conditions).order_by(Submission.id.desc())
    if page_size:
        statement = statement.offset((page - 1) * page_size).limit(page_size)
    submissions = list((await session.scalars(statement)).all())
    return success(
        {
            "total": total,
            "submissions": [_submission_detail(item) for item in submissions],
        }
    )


@router.get("/api/submissions/{submission_id}")
async def get_submission(
    submission_id: str, session: SessionDep, current_user: CurrentUser
) -> dict[str, Any]:
    parsed_id = parse_resource_id(submission_id, "submission_id")
    submission = await session.get(Submission, parsed_id)
    if submission is None:
        raise ApiError(404, "submission not found")
    if current_user.role != "admin" and submission.user_id != current_user.id:
        raise ApiError(403, "permission denied")
    return success(_submission_detail(submission, include_code=True))


@router.put("/api/submissions/{submission_id}/rejudge")
async def rejudge_submission(
    submission_id: str, session: SessionDep, _: AdminUser
) -> dict[str, Any]:
    parsed_id = parse_resource_id(submission_id, "submission_id")
    submission = await session.get(Submission, parsed_id)
    if submission is None:
        raise ApiError(404, "submission not found")
    await cancel_submission_task(parsed_id)
    submission.status = "pending"
    submission.score = None
    submission.counts = None
    submission.compile_info = None
    submission.run_info = None
    submission.error_info = None
    submission.details = None
    submission.updated_at = utc_now()
    await session.commit()
    start_judging(parsed_id)
    return success({"submission_id": str(parsed_id), "status": "pending"}, "rejudge started")


@router.get("/api/submissions/{submission_id}/log")
async def get_submission_log(
    submission_id: str, session: SessionDep, current_user: CurrentUser
) -> dict[str, Any]:
    parsed_id = parse_resource_id(submission_id, "submission_id")
    submission = await session.get(Submission, parsed_id)
    if submission is None:
        raise ApiError(404, "submission not found")
    problem = await session.get(Problem, submission.problem_id)
    if problem is None:
        raise ApiError(404, "problem not found")

    allowed = (
        current_user.role == "admin"
        or submission.user_id == current_user.id
        or problem.public_cases
    )
    session.add(
        AccessLog(
            user_id=current_user.id,
            problem_id=submission.problem_id,
            action="view_logs",
            status="200" if allowed else "403",
        )
    )
    await session.commit()
    if not allowed:
        raise ApiError(403, "permission denied")
    return success(
        {
            "details": submission.details or [],
            "score": submission.score,
            "counts": submission.counts,
        }
    )


@router.get("/api/logs/access/")
async def list_access_logs(request: Request, session: SessionDep, _: AdminUser) -> dict[str, Any]:
    query = request.query_params
    page, page_size = parse_pagination(query.get("page"), query.get("page_size"))
    conditions: list[Any] = []
    if query.get("user_id"):
        conditions.append(AccessLog.user_id == parse_resource_id(query["user_id"], "user_id"))
    if query.get("problem_id"):
        conditions.append(AccessLog.problem_id == query["problem_id"])
    statement = select(AccessLog).where(*conditions).order_by(AccessLog.id.desc())
    if page_size:
        statement = statement.offset((page - 1) * page_size).limit(page_size)
    logs = list((await session.scalars(statement)).all())
    return success(
        [
            {
                "user_id": str(log.user_id),
                "problem_id": log.problem_id,
                "action": log.action,
                "time": log.time.isoformat(),
                "status": log.status,
            }
            for log in logs
        ]
    )


@router.put("/api/ai/model-config")
async def configure_ai_model(request: Request, current_user: CurrentUser) -> dict[str, Any]:
    payload = await parse_body(request, AIModelConfigPayload)
    try:
        data = await ai_service.set_model_config(current_user.id, payload)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return success(data, "model config updated")


@router.get("/api/ai/model-config")
async def read_ai_model_config(current_user: CurrentUser) -> dict[str, Any]:
    config = await ai_service.get_model_config(current_user.id)
    if config is None:
        raise ApiError(404, "model config not found")
    return success(config)


@router.get("/api/ai/model-configs/")
async def list_ai_model_configs(current_user: CurrentUser) -> dict[str, Any]:
    return success(await ai_service.list_model_configs(current_user.id))


@router.delete("/api/ai/model-configs/{config_name}")
async def delete_ai_model_config(config_name: str, current_user: CurrentUser) -> dict[str, Any]:
    try:
        await ai_service.delete_model_config(current_user.id, config_name)
    except ValueError as exc:
        raise ApiError(404, str(exc)) from exc
    return success(None, "model config deleted")


@router.post("/api/ai/problem-tasks/")
async def create_ai_problem_task(
    request: Request, session: SessionDep, current_user: CurrentUser
) -> dict[str, Any]:
    payload = await parse_body(request, AIProblemTaskPayload)
    if payload.problem_id and await session.get(Problem, payload.problem_id) is None:
        raise ApiError(404, "problem not found")
    try:
        task = await ai_service.create_problem_task(current_user.id, payload)
    except ValueError as exc:
        raise ApiError(400, str(exc)) from exc
    return success({"task_id": task.task_id, "status": "pending"}, "task created")


@router.get("/api/ai/problem-tasks/{task_id}")
async def get_ai_problem_task(
    task_id: str, session: SessionDep, current_user: CurrentUser
) -> dict[str, Any]:
    task = await session.get(AIProblemTask, task_id)
    if task is None:
        raise ApiError(404, "task not found")
    if current_user.role != "admin" and task.user_id != current_user.id:
        raise ApiError(403, "permission denied")
    return success(_ai_task_data(task))


@router.put("/api/ai/problem-tasks/{task_id}/cancel")
async def cancel_ai_problem_task(
    task_id: str, session: SessionDep, current_user: CurrentUser
) -> dict[str, Any]:
    task = await session.get(AIProblemTask, task_id)
    if task is None:
        raise ApiError(404, "task not found")
    if current_user.role != "admin" and task.user_id != current_user.id:
        raise ApiError(403, "permission denied")
    if task.status in {"completed", "cancelled", "failed"}:
        raise ApiError(409, "task already finished")
    await ai_service.cancel_problem_task(task_id)
    return success({"task_id": task_id, "status": "cancelled"}, "task cancelled")


@router.post("/api/reset/")
async def reset_system(request: Request, session: SessionDep) -> dict[str, Any]:
    if not get_settings().allow_anonymous_reset:
        user_id = request.session.get("user_id")
        if not isinstance(user_id, int):
            raise ApiError(401, "not logged in")
        user = await session.get(User, user_id)
        if user is None:
            raise ApiError(401, "not logged in")
        if user.role != "admin":
            raise ApiError(403, "permission denied")
    await cancel_all_submission_tasks()
    await ai_service.cancel_all_ai_tasks(clear_configs=True, purge_configs=True)
    await session.rollback()
    await session.close()
    await reset_database()
    await clear_testcase_store()
    request.session.clear()
    return success(None, "system reset successfully")
