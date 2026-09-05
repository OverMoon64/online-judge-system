from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from app.judge import ReferenceExecution, execute_reference_solution, normalize_output
from app.schemas import AIProblemTaskPayload, GeneratedProblem
from app.testcase_files import write_file_testcases

_request_keywords = (
    "复杂度",
    "大数据",
    "最大规模",
    "压力",
    "暴力",
    "性能",
    "超时",
    "卡掉",
    "complexity",
    "stress",
    "brute force",
    "tle",
)
_integer_pattern = re.compile(r"[+-]?\d+")
_bound_value_pattern = r"(?:\d+\s*(?:\^|\*\*)\s*\d+|\d+\s*\*\s*10\s*(?:\^|\*\*)\s*\d+|\d+)"


@dataclass(slots=True)
class StressCandidate:
    template: str
    label: str
    input: str
    scale: dict[str, int]


def stress_requested(request: AIProblemTaskPayload) -> bool:
    context = " ".join([request.requirement, *request.knowledge_points]).lower()
    return any(keyword in context for keyword in _request_keywords)


def _normalize_constraint_notation(value: str) -> str:
    normalized = value.replace("\\leq", "<=").replace("\\le", "<=")
    normalized = normalized.replace("≤", "<=").replace("\\times", "*").replace("×", "*")
    normalized = normalized.replace("$", "").replace("{", "").replace("}", "")
    return re.sub(r"(?<=\d)[,，](?=\d)", "", normalized)


def _parse_bound_value(value: str) -> int | None:
    cleaned = re.sub(r"\s+", "", _normalize_constraint_notation(value).lower())
    match = re.fullmatch(r"(\d+)\*10(?:\^|\*\*)(\d+)", cleaned)
    if match:
        result = int(match.group(1)) * 10 ** int(match.group(2))
    else:
        match = re.fullmatch(r"(\d+)(?:\^|\*\*)(\d+)", cleaned)
        result = int(match.group(1)) ** int(match.group(2)) if match else int(cleaned)
    return result if 0 < result <= 10**18 else None


def _upper_bound(constraints: str, variable: str) -> int | None:
    constraints = _normalize_constraint_notation(constraints)
    escaped = re.escape(variable)
    patterns = (
        rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])\s*(?:<=|≤)\s*({_bound_value_pattern})",
        rf"(?:最大|上限|max)[^\n]{{0,20}}(?<![a-z0-9_]){escaped}(?![a-z0-9_])[^\d]{{0,8}}({_bound_value_pattern})",
    )
    for pattern in patterns:
        match = re.search(pattern, constraints, flags=re.IGNORECASE)
        if match:
            try:
                return _parse_bound_value(match.group(1))
            except (TypeError, ValueError, OverflowError):
                return None
    return None


def _string_upper_bound(constraints: str) -> int | None:
    constraints = _normalize_constraint_notation(constraints)
    patterns = (
        rf"\|\s*[a-z]\s*\|\s*(?:<=|≤)\s*({_bound_value_pattern})",
        rf"(?:字符串)?长度[^\d]{{0,12}}({_bound_value_pattern})",
    )
    for pattern in patterns:
        match = re.search(pattern, constraints, flags=re.IGNORECASE)
        if match:
            try:
                return _parse_bound_value(match.group(1))
            except (TypeError, ValueError, OverflowError):
                return None
    return _upper_bound(constraints, "n")


def _integers(value: str) -> list[int] | None:
    tokens = value.split()
    if not tokens or any(_integer_pattern.fullmatch(token) is None for token in tokens):
        return None
    try:
        return [int(token) for token in tokens]
    except ValueError:
        return None


def _first_input(generated: GeneratedProblem) -> str | None:
    cases = [*generated.problem.samples, *generated.problem.testcases]
    return cases[0].input if cases else None


def _context(generated: GeneratedProblem, request: AIProblemTaskPayload) -> str:
    return " ".join(
        [
            request.requirement,
            *request.knowledge_points,
            generated.problem.title,
            generated.problem.description,
            generated.problem.input_description,
            generated.problem.constraints,
            generated.solution_explanation,
            *generated.problem.tags,
        ]
    ).lower()


def _graph_candidate(numbers: list[int], constraints: str, context: str) -> StressCandidate | None:
    if not any(word in context for word in ("图", "边", "最短路", "并查集", "拓扑")):
        return None
    if len(numbers) < 4:
        return None
    sample_n, sample_m = numbers[:2]
    fields = 2 if len(numbers) == 2 + sample_m * 2 else 3 if len(numbers) == 2 + sample_m * 3 else 0
    if sample_n <= 0 or sample_m < 0 or fields == 0:
        return None
    upper_n = _upper_bound(constraints, "n")
    if upper_n is None:
        return None
    size = min(upper_n, 30_000)
    upper_m = _upper_bound(constraints, "m")
    if upper_m is not None:
        size = min(size, upper_m + 1)
    if size < max(1_000, sample_n * 2):
        return None
    starts_at_zero = 0 in numbers[2 : 2 + sample_m * fields]
    first_node = 0 if starts_at_zero else 1
    weight = numbers[4] if fields == 3 and len(numbers) > 4 else 1
    lines = [f"{size} {size - 1}"]
    for offset in range(size - 1):
        left = first_node + offset
        edge = f"{left} {left + 1}"
        lines.append(f"{edge} {weight}" if fields == 3 else edge)
    return StressCandidate(
        "graph_chain",
        "大规模链式图",
        "\n".join(lines) + "\n",
        {"n": size, "m": size - 1},
    )


def _grid_candidate(numbers: list[int], constraints: str, context: str) -> StressCandidate | None:
    if not any(word in context for word in ("网格", "矩阵", "迷宫")) or len(numbers) < 3:
        return None
    sample_n, sample_m = numbers[:2]
    if sample_n <= 0 or sample_m <= 0 or len(numbers) != 2 + sample_n * sample_m:
        return None
    upper_n = _upper_bound(constraints, "n")
    upper_m = _upper_bound(constraints, "m") or upper_n
    if upper_n is None or upper_m is None:
        return None
    rows = min(upper_n, 316)
    columns = min(upper_m, max(1, 100_000 // rows))
    if rows * columns < max(4_000, sample_n * sample_m * 2):
        return None
    values = numbers[2:]
    cells = [str(values[index % len(values)]) for index in range(rows * columns)]
    lines = [f"{rows} {columns}"]
    for row in range(rows):
        start = row * columns
        lines.append(" ".join(cells[start : start + columns]))
    return StressCandidate(
        "integer_grid",
        "大规模网格",
        "\n".join(lines) + "\n",
        {"n": rows, "m": columns},
    )


def _array_candidate(numbers: list[int], constraints: str, context: str) -> StressCandidate | None:
    if not any(
        word in context for word in ("数组", "序列", "排序", "逆序", "滑动窗口", "双指针", "子数组")
    ):
        return None
    sample_n = numbers[0] if numbers else 0
    if sample_n <= 0 or len(numbers) != sample_n + 1:
        return None
    upper_n = _upper_bound(constraints, "n")
    if upper_n is None:
        return None
    size = min(upper_n, 75_000)
    if size < max(2_000, sample_n * 2):
        return None
    seed = list(reversed(numbers[1:]))
    if any(word in context for word in ("互不相同", "各不相同", "distinct")):
        values = range(size, 0, -1)
    else:
        values = (seed[index % len(seed)] for index in range(size))
    return StressCandidate(
        "counted_integer_array",
        "大规模数组",
        f"{size}\n" + " ".join(map(str, values)) + "\n",
        {"n": size},
    )


def _scalar_candidate(numbers: list[int], constraints: str, context: str) -> StressCandidate | None:
    if len(numbers) != 1 or not any(
        word in context for word in ("斐波那契", "递推", "阶乘", "素数", "快速幂", "数论", "取模")
    ):
        return None
    upper_n = _upper_bound(constraints, "n")
    if upper_n is None or upper_n <= numbers[0]:
        return None
    return StressCandidate(
        "bounded_integer",
        "最大整数边界",
        f"{upper_n}\n",
        {"n": upper_n},
    )


def _string_candidate(sample_input: str, constraints: str, context: str) -> StressCandidate | None:
    if not any(word in context for word in ("字符串", "子串", "回文", "模式匹配")):
        return None
    tokens = sample_input.split()
    if len(tokens) != 1 or _integer_pattern.fullmatch(tokens[0]):
        return None
    upper = _string_upper_bound(constraints)
    if upper is None:
        return None
    size = min(upper, 200_000)
    if size < max(4_000, len(tokens[0]) * 2):
        return None
    seed = tokens[0]
    value = (seed * (size // len(seed) + 1))[:size]
    return StressCandidate(
        "single_string",
        "大规模字符串",
        value + "\n",
        {"length": size},
    )


def infer_stress_candidate(
    generated: GeneratedProblem, request: AIProblemTaskPayload
) -> StressCandidate | None:
    sample_input = _first_input(generated)
    if sample_input is None:
        return None
    constraints = generated.problem.constraints
    context = _context(generated, request)
    numbers = _integers(sample_input)
    if numbers is not None:
        for builder in (_graph_candidate, _grid_candidate, _array_candidate, _scalar_candidate):
            candidate = builder(numbers, constraints, context)
            if candidate is not None:
                return candidate
    return _string_candidate(sample_input, constraints, context)


def _execution_error(execution: ReferenceExecution) -> str | None:
    if execution.setup_error:
        return execution.setup_error
    if len(execution.results) != 1:
        return "参考程序没有返回唯一压力点结果"
    result = execution.results[0]
    return None if result.status == "OK" else result.status


async def build_optional_file_stress(
    generated: GeneratedProblem,
    request: AIProblemTaskPayload,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested": stress_requested(request),
        "applied": False,
        "count": 0,
    }
    if not metadata["requested"]:
        return metadata
    if request.problem_id:
        metadata["reason"] = "修改已有题目时不预写文件压力点"
        return metadata
    candidate = infer_stress_candidate(generated, request)
    if candidate is None:
        metadata["reason"] = "未能安全识别受支持的输入结构"
        return metadata
    metadata.update(
        {"template": candidate.template, "label": candidate.label, "scale": candidate.scale}
    )
    try:
        python_execution, cpp_execution = await asyncio.gather(
            execute_reference_solution(
                generated.problem,
                generated.reference_solution,
                "python",
                inputs=[candidate.input],
            ),
            execute_reference_solution(
                generated.problem,
                generated.reference_solution_cpp,
                "cpp",
                inputs=[candidate.input],
            ),
        )
        python_error = _execution_error(python_execution)
        cpp_error = _execution_error(cpp_execution)
        metadata["reference_status"] = {
            "python": python_error or "OK",
            "cpp": cpp_error or "OK",
        }
        if python_error or cpp_error:
            metadata["reason"] = "参考程序未在压力点资源限制内完成"
            return metadata
        python_result = python_execution.results[0]
        cpp_result = cpp_execution.results[0]
        if normalize_output(python_result.stdout) != normalize_output(cpp_result.stdout):
            metadata["reason"] = "Python/C++ 压力点输出不一致"
            return metadata
        manifest = await write_file_testcases(
            generated.problem,
            [
                {
                    "label": candidate.label,
                    "input": candidate.input,
                    "output": python_result.stdout,
                }
            ],
            metadata={"template": candidate.template, "scale": candidate.scale},
        )
        file_case = manifest["cases"][0]
        metadata.update(
            {
                "applied": True,
                "count": 1,
                "input_bytes": file_case["input_bytes"],
                "output_bytes": file_case["output_bytes"],
                "files": [{"input": file_case["input_file"], "output": file_case["output_file"]}],
                "reference_time": {
                    "python": round(python_result.elapsed, 4),
                    "cpp": round(cpp_result.elapsed, 4),
                },
            }
        )
        return metadata
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        metadata["reason"] = f"文件压力点生成失败：{str(exc)[:160]}"
        return metadata
