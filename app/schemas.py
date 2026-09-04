from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class TestCase(StrictModel):
    input: str
    output: str


class ProblemPayload(StrictModel):
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    input_description: str = Field(min_length=1)
    output_description: str = Field(min_length=1)
    samples: list[TestCase]
    constraints: str = Field(min_length=1)
    testcases: list[TestCase]
    hint: str = ""
    source: str = ""
    tags: list[str] = Field(default_factory=list)
    time_limit: float = Field(default=3.0, gt=0, le=60)
    memory_limit: int = Field(default=128, gt=0, le=4096)
    author: str = ""
    difficulty: str = ""

    @field_validator("id", "title", "description", "input_description", "output_description")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class LogVisibilityPayload(StrictModel):
    public_cases: bool = False


class LanguagePayload(StrictModel):
    name: str = Field(min_length=1, max_length=40)
    file_ext: str = Field(min_length=1, max_length=16)
    compile_cmd: str | None = None
    run_cmd: str = Field(min_length=1)
    time_limit: float | None = Field(default=None, gt=0, le=60)
    memory_limit: int | None = Field(default=None, gt=0, le=4096)

    @field_validator("file_ext")
    @classmethod
    def validate_extension(cls, value: str) -> str:
        if not value.startswith(".") or any(char in value for char in "/\\\0"):
            raise ValueError("file_ext must be a safe extension beginning with '.'")
        return value


class SubmissionPayload(StrictModel):
    problem_id: str = Field(min_length=1, max_length=80)
    language: str = Field(min_length=1, max_length=40)
    code: str = Field(min_length=1, max_length=200_000)


class LoginPayload(StrictModel):
    username: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=1, max_length=256)


class RegisterPayload(StrictModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=6, max_length=256)

    @field_validator("username")
    @classmethod
    def reject_blank_username(cls, value: str) -> str:
        if value != value.strip() or not value.strip():
            raise ValueError("username must not have surrounding whitespace")
        return value


class RolePayload(StrictModel):
    role: Literal["admin", "user", "banned"]


class AIModelConfigPayload(StrictModel):
    name: str = Field(default="默认模型", min_length=1, max_length=60)
    provider_url: str = Field(min_length=8, max_length=500)
    model: str = Field(min_length=1, max_length=120)
    api_key: str = Field(min_length=1, max_length=1000)
    input_price: float = Field(default=0.0, ge=0)
    output_price: float = Field(default=0.0, ge=0)
    price_unit: int = Field(default=1_000_000, gt=0)
    currency: str = Field(default="CNY", min_length=1, max_length=12)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value or "/" in value or "\\" in value:
            raise ValueError("name must not be empty or contain path separators")
        return value

    @field_validator("provider_url")
    @classmethod
    def validate_provider_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("provider_url must be an HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("provider_url must not contain credentials")
        return value.rstrip("/")


class AIProblemTaskPayload(StrictModel):
    requirement: str = Field(min_length=10, max_length=10_000)
    problem_id: str | None = Field(default=None, min_length=1, max_length=80)
    model_config_name: str | None = Field(default=None, min_length=1, max_length=60)
    knowledge_points: list[str] = Field(default_factory=list, max_length=20)
    difficulty: str = Field(default="中等", max_length=40)
    testcase_count: int = Field(default=6, ge=2, le=10)


class GeneratedProblem(StrictModel):
    problem: ProblemPayload
    reference_solution: str = Field(min_length=1)
    solution_explanation: str = ""
