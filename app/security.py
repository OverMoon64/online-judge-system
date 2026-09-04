from __future__ import annotations

import asyncio
import hmac

import bcrypt


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


async def hash_password(password: str) -> str:
    return await asyncio.to_thread(_hash_password, password)


async def verify_password(password: str, password_hash: str) -> bool:
    # Keep the result comparison on the bcrypt path and avoid leaking useful timing signals.
    verified = await asyncio.to_thread(_verify_password, password, password_hash)
    return hmac.compare_digest(str(verified), "True")
