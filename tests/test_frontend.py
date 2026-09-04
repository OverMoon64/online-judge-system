from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend.app import (
    output_calibration_rows,
    resolve_problem_selection,
    status_badge,
    submission_verdict,
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


def test_frontend_initial_account_page_renders() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    app = AppTest.from_file(str(frontend), default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "账户中心"
    assert {tab.label for tab in app.tabs} == {"登录账户", "注册账户"}
    assert any("在线评测系统" in item.value for item in app.markdown)
    assert not app.radio
