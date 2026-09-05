from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

from app.config import get_settings

_store_override: Path | None = None
_case_name_pattern = re.compile(r"stress-[0-9]{3}\.(?:in|out)")
MAX_FILE_CASES = 3
MAX_INPUT_BYTES = 1_000_000
MAX_OUTPUT_BYTES = 1_048_576


def configure_testcase_store(path: str | Path | None) -> None:
    global _store_override
    _store_override = Path(path) if path is not None else None


def _store_root() -> Path:
    root = (_store_override or Path(get_settings().testcase_dir)).resolve()
    broad_targets = {Path(root.anchor).resolve(), Path.home().resolve(), Path.cwd().resolve()}
    if root in broad_targets:
        raise ValueError("testcase_dir must be a dedicated subdirectory")
    return root


def _bundle_directory(problem_id: str) -> Path:
    key = hashlib.sha256(problem_id.encode("utf-8")).hexdigest()
    return _store_root() / key


def problem_fingerprint(problem: Any) -> str:
    if hasattr(problem, "model_dump"):
        values = problem.model_dump(mode="json")
    elif isinstance(problem, dict):
        values = problem
    else:
        values = {
            name: getattr(problem, name)
            for name in (
                "id",
                "input_description",
                "output_description",
                "constraints",
                "samples",
                "testcases",
            )
        }
    contract = {
        name: values.get(name)
        for name in (
            "id",
            "input_description",
            "output_description",
            "constraints",
            "samples",
            "testcases",
        )
    }
    encoded = json.dumps(
        contract, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


async def write_file_testcases(
    problem: Any,
    cases: list[dict[str, str]],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not 1 <= len(cases) <= MAX_FILE_CASES:
        raise ValueError("file testcase count must be between 1 and 3")

    prepared: list[tuple[bytes, bytes, str]] = []
    for index, case in enumerate(cases, start=1):
        input_bytes = case["input"].encode("utf-8")
        output_bytes = case["output"].encode("utf-8")
        if not input_bytes or len(input_bytes) > MAX_INPUT_BYTES:
            raise ValueError(f"file testcase {index} input size is invalid")
        if len(output_bytes) > MAX_OUTPUT_BYTES:
            raise ValueError(f"file testcase {index} output is too large")
        prepared.append((input_bytes, output_bytes, case.get("label", f"压力点 {index}")))

    problem_id = str(getattr(problem, "id", None) or problem["id"])
    manifest: dict[str, Any] = {
        "version": 1,
        "problem_id": problem_id,
        "problem_fingerprint": problem_fingerprint(problem),
        "metadata": metadata or {},
        "cases": [],
    }

    def write() -> dict[str, Any]:
        root = _store_root()
        root.mkdir(parents=True, exist_ok=True)
        target = _bundle_directory(problem_id)
        temporary = root / f".{target.name}-{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        try:
            for index, (input_bytes, output_bytes, label) in enumerate(prepared, start=1):
                input_name = f"stress-{index:03d}.in"
                output_name = f"stress-{index:03d}.out"
                (temporary / input_name).write_bytes(input_bytes)
                (temporary / output_name).write_bytes(output_bytes)
                manifest["cases"].append(
                    {
                        "label": label,
                        "input_file": input_name,
                        "output_file": output_name,
                        "input_bytes": len(input_bytes),
                        "output_bytes": len(output_bytes),
                        "input_sha256": _sha256(input_bytes),
                        "output_sha256": _sha256(output_bytes),
                    }
                )
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if target.exists():
                shutil.rmtree(target)
            temporary.replace(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return manifest

    return await asyncio.to_thread(write)


def _load_verified(problem: Any, *, include_content: bool) -> list[dict[str, Any]]:
    problem_id = str(getattr(problem, "id", None) or problem["id"])
    directory = _bundle_directory(problem_id)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return []
    if (
        manifest.get("version") != 1
        or manifest.get("problem_id") != problem_id
        or manifest.get("problem_fingerprint") != problem_fingerprint(problem)
    ):
        return []
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_FILE_CASES:
        return []

    verified: list[dict[str, Any]] = []
    for case in cases:
        if not isinstance(case, dict):
            return []
        input_name = case.get("input_file")
        output_name = case.get("output_file")
        if not isinstance(input_name, str) or not isinstance(output_name, str):
            return []
        if not _case_name_pattern.fullmatch(input_name) or not _case_name_pattern.fullmatch(
            output_name
        ):
            return []
        try:
            input_bytes = (directory / input_name).read_bytes()
            output_bytes = (directory / output_name).read_bytes()
        except OSError:
            return []
        if (
            not input_bytes
            or len(input_bytes) > MAX_INPUT_BYTES
            or len(output_bytes) > MAX_OUTPUT_BYTES
            or case.get("input_bytes") != len(input_bytes)
            or case.get("output_bytes") != len(output_bytes)
            or case.get("input_sha256") != _sha256(input_bytes)
            or case.get("output_sha256") != _sha256(output_bytes)
        ):
            return []
        item = {
            "label": str(case.get("label") or "文件压力点"),
            "input_file": input_name,
            "output_file": output_name,
            "input_bytes": len(input_bytes),
            "output_bytes": len(output_bytes),
        }
        if include_content:
            try:
                item["input"] = input_bytes.decode("utf-8")
                item["output"] = output_bytes.decode("utf-8")
            except UnicodeDecodeError:
                return []
        verified.append(item)
    return verified


async def load_file_testcases(problem: Any) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_load_verified, problem, include_content=True)


async def count_file_testcases(problem: Any) -> int:
    cases = await asyncio.to_thread(_load_verified, problem, include_content=False)
    return len(cases)


async def remove_file_testcases(problem_id: str) -> None:
    directory = _bundle_directory(problem_id)
    root = _store_root()
    if directory.parent != root:
        raise ValueError("invalid testcase directory")
    await asyncio.to_thread(shutil.rmtree, directory, True)


async def clear_testcase_store() -> None:
    root = _store_root()
    await asyncio.to_thread(shutil.rmtree, root, True)
