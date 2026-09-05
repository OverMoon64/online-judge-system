from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend.app import (
    LANGUAGE_COMPATIBILITY,
    MODEL_COMPATIBILITY,
    REASONING_LABELS,
    compact_preview,
    output_calibration_rows,
    resolve_problem_selection,
    resource_limit_summary,
    status_badge,
    submission_verdict,
)


def test_ai_page_lists_required_model_families_and_reasoning_levels() -> None:
    compatibility = " ".join(str(item) for item in MODEL_COMPATIBILITY)
    assert all(name in compatibility for name in ("Qwen", "DeepSeek", "Kimi"))
    assert "OpenAI-compatible" in compatibility
    assert set(REASONING_LABELS) == {"auto", "low", "medium", "high", "max"}


def test_language_page_lists_supported_command_templates() -> None:
    compatibility = " ".join(str(item) for item in LANGUAGE_COMPATIBILITY)
    assert all(
        name in compatibility
        for name in ("Python", "C++14", "C17", "Java", "JavaScript", "Ruby", "Go")
    )
    assert all(
        "{src}" in item["运行命令"] or "{exe}" in item["运行命令"]
        for item in LANGUAGE_COMPATIBILITY
    )


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"status": "pending", "result": "pending"}, "pending"),
        ({"status": "error", "result": "UNK"}, "error"),
        ({"status": "success", "result": "AC"}, "AC"),
        ({"status": "success", "result": "WA"}, "WA"),
        ({"status": "success", "result": "TLE"}, "TLE"),
        ({"status": "success", "result": "MLE"}, "MLE"),
        ({"status": "success", "result": "RE"}, "RE"),
        ({"status": "success", "result": "CE"}, "CE"),
        ({"status": "success", "result": "UNK"}, "UNK"),
    ],
)
def test_submission_verdict_keeps_specific_outcome(
    payload: dict[str, object], expected: str
) -> None:
    assert submission_verdict(payload) == expected


def test_submission_badges_include_acronym_and_distinct_colors() -> None:
    badges = {status: status_badge(status) for status in ("AC", "WA", "TLE", "MLE", "RE", "CE")}
    assert all(status in badge for status, badge in badges.items())
    assert (
        len({badge.split("background:", 1)[1].split(";", 1)[0] for badge in badges.values()}) == 6
    )


def test_output_calibration_rows_are_readable() -> None:
    rows = output_calibration_rows(
        {
            "output_calibration": {
                "applied": True,
                "count": 2,
                "items": [
                    {
                        "kind": "sample",
                        "index": 1,
                        "input": "100\n",
                        "original_output": "wrong",
                        "calibrated_output": "687995182",
                    },
                    {
                        "kind": "testcase",
                        "index": 3,
                        "input": "100000\n",
                        "original_output": "",
                        "calibrated_output": "911435502",
                    },
                ],
            }
        }
    )
    assert rows[0] == {
        "类型": "样例",
        "序号": 1,
        "输入": "100\n",
        "模型原答案": "wrong",
        "校准答案": "687995182",
    }
    assert rows[1]["类型"] == "隐藏测试点"
    assert rows[1]["模型原答案"] == ""


def test_problem_selection_survives_reruns_and_problem_list_changes() -> None:
    problem_ids = ["first", "chosen", "last"]
    assert resolve_problem_selection(problem_ids, None, "chosen") == "chosen"
    assert resolve_problem_selection(problem_ids, "chosen") == "chosen"
    assert resolve_problem_selection(problem_ids, "missing") == "first"
    assert resolve_problem_selection([], "chosen") is None


def test_resource_limit_summary_explains_automatic_decision() -> None:
    summary = resource_limit_summary(
        {
            "resource_limits": {
                "source": "automatic",
                "final": {"time_limit": 3.0, "memory_limit": 256},
                "reasons": ["难度为困难", "需要维护图、网格或状态结构"],
            }
        }
    )
    assert summary is not None
    assert "自动评估" in summary
    assert "3.0 s" in summary
    assert "256 MB" in summary
    assert resource_limit_summary({}) is None


def test_large_preview_is_clear() -> None:
    assert compact_preview("short", 10) == "short"
    preview = compact_preview("x" * 30, 10)
    assert preview.startswith("x" * 10)
    assert "共 30 字符" in preview


def test_frontend_initial_account_page_renders() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    app = AppTest.from_file(str(frontend), default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "账户中心"
    assert {tab.label for tab in app.tabs} == {"登录账户", "注册账户"}
    assert any("在线评测系统" in item.value for item in app.markdown)
    assert not app.radio
