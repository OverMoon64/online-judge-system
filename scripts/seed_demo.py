from __future__ import annotations

import argparse
import asyncio

import httpx

PROBLEM = {
    "id": "sum_2",
    "title": "两数之和",
    "description": "输入两个整数 a、b，输出它们的和。",
    "input_description": "一行包含两个整数 a 和 b，以空格分隔。",
    "output_description": "输出一个整数 a+b。",
    "samples": [{"input": "1 2\n", "output": "3\n"}],
    "constraints": "|a|, |b| <= 10^9",
    "testcases": [
        {"input": "1 2\n", "output": "3\n"},
        {"input": "-5 8\n", "output": "3\n"},
        {"input": "0 0\n", "output": "0\n"},
        {"input": "1000000000 -1000000000\n", "output": "0\n"},
    ],
    "hint": "注意负数和整数范围。",
    "source": "课程演示",
    "tags": ["入门", "输入输出"],
    "time_limit": 1.0,
    "memory_limit": 128,
    "author": "Course Demo",
    "difficulty": "入门",
}


async def seed(base_url: str) -> None:
    async with httpx.AsyncClient(base_url=base_url, follow_redirects=True) as client:
        login = await client.post(
            "/api/auth/login",
            json={"username": "admin", "password": "admintestpassword"},
        )
        login.raise_for_status()
        problem = await client.post("/api/problems/", json=PROBLEM)
        if problem.status_code not in {200, 409}:
            problem.raise_for_status()
        user = await client.post(
            "/api/users/", json={"username": "student", "password": "student123"}
        )
        if user.status_code not in {200, 400}:
            user.raise_for_status()
    print("Demo data is ready: admin/admintestpassword, student/student123")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create repeatable OJ demo data")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    asyncio.run(seed(args.base_url))


if __name__ == "__main__":
    main()
