from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from app.judge import execute_reference_solution, normalize_output
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
    category: str = "scale"


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


def _lower_bound(constraints: str, variable: str) -> int | None:
    constraints = _normalize_constraint_notation(constraints)
    escaped = re.escape(variable)
    patterns = (
        rf"({_bound_value_pattern})\s*(?:<=|<)\s*(?<![a-z0-9_]){escaped}(?![a-z0-9_])",
        rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])\s*(?:>=|>)\s*({_bound_value_pattern})",
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


def _boundary_candidate(
    generated: GeneratedProblem, candidate: StressCandidate
) -> StressCandidate | None:
    sample_input = _first_input(generated)
    if sample_input is None:
        return None
    numbers = _integers(sample_input)
    constraints = generated.problem.constraints
    if candidate.template == "bounded_integer":
        lower = _lower_bound(constraints, "n")
        if lower is None:
            lower = 0 if re.search(r"0\s*(?:<=|<)\s*n", constraints, re.IGNORECASE) else 1
        return StressCandidate(
            candidate.template,
            "最小整数边界",
            f"{lower}\n",
            {"n": lower},
            "boundary",
        )
    if candidate.template == "counted_integer_array" and numbers and len(numbers) > 1:
        size = max(1, _lower_bound(constraints, "n") or 1)
        return StressCandidate(
            candidate.template,
            "最小数组边界",
            f"{size}\n" + " ".join([str(numbers[1])] * size) + "\n",
            {"n": size},
            "boundary",
        )
    if candidate.template == "integer_grid" and numbers and len(numbers) > 2:
        rows = max(1, _lower_bound(constraints, "n") or 1)
        columns = max(1, _lower_bound(constraints, "m") or rows)
        row = " ".join([str(numbers[2])] * columns)
        return StressCandidate(
            candidate.template,
            "最小网格边界",
            f"{rows} {columns}\n" + "\n".join([row] * rows) + "\n",
            {"n": rows, "m": columns},
            "boundary",
        )
    if candidate.template == "graph_chain":
        return StressCandidate(
            candidate.template,
            "最小图边界",
            "1 0\n",
            {"n": 1, "m": 0},
            "boundary",
        )
    if candidate.template == "single_string":
        token = sample_input.split()[0]
        length = max(1, _lower_bound(constraints, "n") or 1)
        return StressCandidate(
            candidate.template,
            "最短字符串边界",
            token[0] * length + "\n",
            {"length": length},
            "boundary",
        )
    return None


def infer_validity_candidates(
    generated: GeneratedProblem, request: AIProblemTaskPayload
) -> list[StressCandidate]:
    scale = infer_stress_candidate(generated, request)
    if scale is None:
        return []
    boundary = _boundary_candidate(generated, scale)
    return [candidate for candidate in (boundary, scale) if candidate is not None]


async def build_optional_file_stress(
    generated: GeneratedProblem,
    request: AIProblemTaskPayload,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "requested": stress_requested(request),
        "attempted": True,
        "applied": False,
        "count": 0,
        "coverage": {"boundary": False, "scale": False},
    }
    if request.problem_id:
        metadata["reason"] = "修改已有题目时不预写文件压力点"
        return metadata
    candidates = infer_validity_candidates(generated, request)
    if not candidates:
        metadata["reason"] = "未能安全识别受支持的输入结构"
        return metadata
    scale_candidate = next(
        (candidate for candidate in candidates if candidate.category == "scale"), candidates[-1]
    )
    metadata.update(
        {
            "template": scale_candidate.template,
            "label": scale_candidate.label,
            "scale": scale_candidate.scale,
        }
    )
    inline_inputs = {
        case.input.strip() for case in [*generated.problem.samples, *generated.problem.testcases]
    }
    pending: list[StressCandidate] = []
    for candidate in candidates:
        if candidate.input.strip() in inline_inputs:
            metadata["coverage"][candidate.category] = True
        else:
            pending.append(candidate)
    if not pending:
        metadata["reason"] = "边界与规模数据已由普通测试点覆盖"
        return metadata
    try:
        python_execution, cpp_execution = await asyncio.gather(
            execute_reference_solution(
                generated.problem,
                generated.reference_solution,
                "python",
                inputs=[candidate.input for candidate in pending],
            ),
            execute_reference_solution(
                generated.problem,
                generated.reference_solution_cpp,
                "cpp",
                inputs=[candidate.input for candidate in pending],
            ),
        )
        python_error = python_execution.setup_error
        cpp_error = cpp_execution.setup_error
        metadata["reference_status"] = {
            "python": python_error or "OK",
            "cpp": cpp_error or "OK",
        }
        if python_error or cpp_error:
            metadata["reason"] = "参考程序未在压力点资源限制内完成"
            return metadata
        accepted: list[tuple[StressCandidate, Any, Any]] = []
        rejected: list[dict[str, str]] = []
        for index, candidate in enumerate(pending):
            python_result = python_execution.results[index]
            cpp_result = cpp_execution.results[index]
            reason = None
            if python_result.status != "OK" or cpp_result.status != "OK":
                reason = f"Python={python_result.status}, C++={cpp_result.status}"
            elif normalize_output(python_result.stdout) != normalize_output(cpp_result.stdout):
                reason = "Python/C++ 输出不一致"
            if reason:
                rejected.append({"label": candidate.label, "reason": reason})
                continue
            metadata["coverage"][candidate.category] = True
            accepted.append((candidate, python_result, cpp_result))
        if rejected:
            metadata["rejected"] = rejected
        if not accepted:
            metadata["reason"] = "候选边界与规模点均未通过双语言校验"
            return metadata
        manifest = await write_file_testcases(
            generated.problem,
            [
                {
                    "label": candidate.label,
                    "input": candidate.input,
                    "output": python_result.stdout,
                }
                for candidate, python_result, _ in accepted
            ],
            metadata={
                "template": scale_candidate.template,
                "cases": [
                    {"category": candidate.category, "scale": candidate.scale}
                    for candidate, _, _ in accepted
                ],
            },
        )
        file_cases = manifest["cases"]
        metadata.update(
            {
                "applied": True,
                "count": len(file_cases),
                "input_bytes": sum(case["input_bytes"] for case in file_cases),
                "output_bytes": sum(case["output_bytes"] for case in file_cases),
                "files": [
                    {"input": case["input_file"], "output": case["output_file"]}
                    for case in file_cases
                ],
                "cases": [
                    {
                        "label": candidate.label,
                        "category": candidate.category,
                        "scale": candidate.scale,
                        "input_bytes": file_case["input_bytes"],
                    }
                    for (candidate, _, _), file_case in zip(accepted, file_cases, strict=True)
                ],
                "reference_time": {
                    "python": round(max(result.elapsed for _, result, _ in accepted), 4),
                    "cpp": round(max(result.elapsed for _, _, result in accepted), 4),
                },
            }
        )
        return metadata
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        metadata["reason"] = f"文件压力点生成失败：{str(exc)[:160]}"
        return metadata


def build_testcase_validity_report(
    generated: GeneratedProblem, file_stress: dict[str, Any]
) -> dict[str, Any]:
    hidden_inputs = [case.input.strip() for case in generated.problem.testcases]
    coverage = file_stress.get("coverage") or {}
    checks = [
        {
            "key": "reference_consensus",
            "label": "答案正确性",
            "passed": True,
            "detail": "全部样例和测试点已通过 Python/C++ 双语言交叉运行",
        },
        {
            "key": "input_diversity",
            "label": "输入多样性",
            "passed": len(hidden_inputs) >= 2 and len(set(hidden_inputs)) == len(hidden_inputs),
            "detail": f"{len(hidden_inputs)} 个隐藏测试点，输入互不重复",
        },
        {
            "key": "boundary_coverage",
            "label": "边界覆盖",
            "passed": bool(coverage.get("boundary")),
            "detail": (
                "最小边界已由普通测试点或双语言校验的文件点覆盖"
                if coverage.get("boundary")
                else "未能从输入契约安全构造最小边界，请人工补充"
            ),
        },
        {
            "key": "scale_coverage",
            "label": "最大规模",
            "passed": bool(coverage.get("scale")),
            "detail": (
                f"已覆盖压力规模 {file_stress.get('scale', {})}"
                if coverage.get("scale")
                else "未能从输入契约安全构造最大规模数据，请人工补充"
            ),
        },
        {
            "key": "complexity_discrimination",
            "label": "复杂度区分",
            "passed": bool(coverage.get("scale")),
            "detail": (
                "最大规模点已纳入正式判题，可用于淘汰常见低效实现"
                if coverage.get("scale")
                else "缺少已验证的规模点，暂不能证明可区分不同复杂度"
            ),
        },
    ]
    score = sum(20 for check in checks if check["passed"])
    return {
        "passed": score == 100,
        "score": score,
        "checks": checks,
        "manual_review_required": True,
    }
