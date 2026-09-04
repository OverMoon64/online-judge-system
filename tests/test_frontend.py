from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend.app import status_badge, submission_verdict


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


def test_frontend_initial_account_page_renders() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    app = AppTest.from_file(str(frontend), default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "账户中心"
    assert {tab.label for tab in app.tabs} == {"登录账户", "注册账户"}
    assert any("在线评测系统" in item.value for item in app.markdown)
    assert not app.radio
