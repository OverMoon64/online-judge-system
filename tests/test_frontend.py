from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend import app as frontend_app
from frontend.app import (
    LANGUAGE_COMPATIBILITY,
    MODEL_COMPATIBILITY,
    REASONING_LABELS,
    compact_preview,
    output_calibration_rows,
    resolve_problem_selection,
    resource_limit_summary,
    status_badge,
    submission_list_params,
    submission_verdict,
    validity_check_rows,
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


def test_logout_clears_sensitive_model_form_state(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {
        "login": {"user_id": "1", "username": "admin", "role": "admin"},
        "ai_config_api_key": "secret",
        "ai_config_provider_url": "https://provider.example/v1",
        "judge_code_problem_python": "print('user code')",
        "unrelated_preference": "keep",
    }
    monkeypatch.setattr(frontend_app.st, "session_state", state)
    frontend_app.clear_local_session()
    assert "login" not in state
    assert "ai_config_api_key" not in state
    assert "ai_config_provider_url" not in state
    assert "judge_code_problem_python" not in state
    assert state["unrelated_preference"] == "keep"


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


def test_testcase_validity_rows_explain_passed_and_review_items() -> None:
    rows = validity_check_rows(
        {
            "testcase_validity": {
                "checks": [
                    {"label": "边界覆盖", "passed": True, "detail": "已覆盖最小值"},
                    {"label": "最大规模", "passed": False, "detail": "请人工补充"},
                ]
            }
        }
    )
    assert rows == [
        {"检查项": "边界覆盖", "结果": "通过", "依据": "已覆盖最小值"},
        {"检查项": "最大规模", "结果": "需复核", "依据": "请人工补充"},
    ]


def test_problem_selection_survives_reruns_and_problem_list_changes() -> None:
    problem_ids = ["first", "chosen", "last"]
    assert resolve_problem_selection(problem_ids, None, "chosen") == "chosen"
    assert resolve_problem_selection(problem_ids, "chosen") == "chosen"
    assert resolve_problem_selection(problem_ids, "missing") == "first"
    assert resolve_problem_selection([], "chosen") is None


def test_submission_filters_respect_user_and_admin_api_rules() -> None:
    user = {"user_id": "2", "username": "alice", "role": "user"}
    assert submission_list_params(user, problem_id="", page=2, selected_user_id="99") == {
        "user_id": "2",
        "page": 2,
        "page_size": 5,
    }

    admin = {"user_id": "1", "username": "admin", "role": "admin"}
    assert submission_list_params(admin, problem_id="", page=1, selected_user_id="2") == {
        "user_id": "2",
        "page": 1,
        "page_size": 5,
    }
    assert submission_list_params(admin, problem_id="sum_2", page=3, selected_user_id="") == {
        "problem_id": "sum_2",
        "page": 3,
        "page_size": 5,
    }
    assert submission_list_params(admin, problem_id="", page=1, selected_user_id="") is None


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
