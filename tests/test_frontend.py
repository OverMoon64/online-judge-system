from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_frontend_initial_account_page_renders() -> None:
    frontend = Path(__file__).resolve().parents[1] / "frontend" / "app.py"
    app = AppTest.from_file(str(frontend), default_timeout=10).run()
    assert not app.exception
    assert app.title[0].value == "⚖️ Async Online Judge"
    assert {tab.label for tab in app.tabs} == {"登录", "注册"}
