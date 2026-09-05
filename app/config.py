from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./data/oj.db"
    session_secret: str = "development-only-change-this-secret"
    cookie_secure: bool = False
    allow_anonymous_reset: bool = False
    allow_local_ai: bool = False
    max_output_bytes: int = 1_048_576
    compile_timeout_seconds: float = 15.0
    allowed_executables: str = "python,python3,g++,gcc,clang,clang++,java,javac,node,ruby,go"
    ai_config_file: str = "./data/ai-model-configs.enc"
    testcase_dir: str = "./data/testcases"

    model_config = SettingsConfigDict(
        env_prefix="OJ_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def executable_allowlist(self) -> set[str]:
        return {item.strip() for item in self.allowed_executables.split(",") if item.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
