from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import User, get_session
from app.errors import ApiError

SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_current_user(request: Request, session: SessionDep) -> User:
    user_id = request.session.get("user_id")
    session_token = request.session.get("session_token")
    if not isinstance(user_id, int) or not isinstance(session_token, str):
        raise ApiError(401, "not logged in")
    user = await session.get(User, user_id)
    if user is None:
        request.session.clear()
        raise ApiError(401, "not logged in")
    if not user.session_token or not secrets.compare_digest(user.session_token, session_token):
        request.session.clear()
        raise ApiError(401, "not logged in")
    if user.role == "banned":
        raise ApiError(403, "user is banned")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def require_admin(current_user: CurrentUser) -> User:
    if current_user.role != "admin":
        raise ApiError(403, "permission denied")
    return current_user


AdminUser = Annotated[User, Depends(require_admin)]
