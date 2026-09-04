from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_frontend_initial_account_page_renders() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    app = AppTest.from_file(str(frontend), default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "账户中心"
    assert {tab.label for tab in app.tabs} == {"登录账户", "注册账户"}
    assert app.radio[0].options == ["登录 / 注册"]
