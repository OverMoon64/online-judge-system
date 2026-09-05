from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings
from app.security import hash_password


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="user", index=True)
    join_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class Problem(Base):
    __tablename__ = "problems"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text)
    input_description: Mapped[str] = mapped_column(Text)
    output_description: Mapped[str] = mapped_column(Text)
    samples: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    constraints: Mapped[str] = mapped_column(Text)
    testcases: Mapped[list[dict[str, str]]] = mapped_column(JSON, default=list)
    hint: Mapped[str] = mapped_column(Text, default="")
    source: Mapped[str] = mapped_column(String(200), default="")
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    time_limit: Mapped[float] = mapped_column(Float, default=3.0)
    memory_limit: Mapped[int] = mapped_column(Integer, default=128)
    author: Mapped[str] = mapped_column(String(100), default="")
    difficulty: Mapped[str] = mapped_column(String(40), default="")
    public_cases: Mapped[bool] = mapped_column(Boolean, default=False)


class Language(Base):
    __tablename__ = "languages"

    name: Mapped[str] = mapped_column(String(40), primary_key=True)
    file_ext: Mapped[str] = mapped_column(String(16))
    compile_cmd: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_cmd: Mapped[str] = mapped_column(Text)
    time_limit: Mapped[float | None] = mapped_column(Float, nullable=True)
    memory_limit: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    problem_id: Mapped[str] = mapped_column(
        ForeignKey("problems.id", ondelete="CASCADE"), index=True
    )
    language: Mapped[str] = mapped_column(String(40), index=True)
    code: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    counts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    compile_info: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    run_info: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_info: Mapped[str | None] = mapped_column(Text, nullable=True)
    details: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


class AccessLog(Base):
    __tablename__ = "access_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    problem_id: Mapped[str] = mapped_column(String(80), index=True)
    action: Mapped[str] = mapped_column(String(40), default="view_logs")
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    status: Mapped[str] = mapped_column(String(8))


class AIProblemTask(Base):
    __tablename__ = "ai_problem_tasks"

    task_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    progress: Mapped[str] = mapped_column(String(300), default="等待处理")
    progress_percent: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    usage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )


def _make_engine(url: str) -> AsyncEngine:
    if url.startswith("sqlite"):
        Path("data").mkdir(parents=True, exist_ok=True)
    created = create_async_engine(url, future=True)
    if url.startswith("sqlite"):

        @event.listens_for(created.sync_engine, "connect")
        def _enable_foreign_keys(dbapi_connection: Any, _: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    return created


engine = _make_engine(get_settings().database_url)
session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def configure_database(url: str) -> None:
    global engine, session_factory
    await engine.dispose()
    engine = _make_engine(url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


def builtin_languages() -> list[Language]:
    return [
        Language(
            name="python",
            file_ext=".py",
            compile_cmd=None,
            run_cmd="python3 {src}",
            time_limit=3.0,
            memory_limit=128,
        ),
        Language(
            name="cpp",
            file_ext=".cpp",
            compile_cmd="g++ {src} -std=c++14 -O2 -o {exe}",
            run_cmd="{exe}",
            time_limit=3.0,
            memory_limit=128,
        ),
    ]


async def initialize_database() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with session_factory() as session:
        if await session.get(User, 1) is None:
            session.add(
                User(
                    id=1,
                    username="admin",
                    password_hash=await hash_password("admintestpassword"),
                    role="admin",
                )
            )
        for language in builtin_languages():
            if await session.get(Language, language.name) is None:
                session.add(language)
        await session.commit()


async def reset_database() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)
    async with session_factory() as session:
        session.add(
            User(
                id=1,
                username="admin",
                password_hash=await hash_password("admintestpassword"),
                role="admin",
            )
        )
        session.add_all(builtin_languages())
        await session.commit()
