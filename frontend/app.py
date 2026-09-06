from __future__ import annotations

import json
import os
from html import escape
from typing import Any

import httpx
import streamlit as st
from code_editor import code_editor
from streamlit_cookies_manager import EncryptedCookieManager

try:
    from frontend.api_client import (
        clear_client_cookies,
        close_persistent_client,
        get_persistent_client,
        normalize_base_url,
        request_json,
    )
    from frontend.browser_session import (
        browser_cookie_password,
        clear_browser_session,
        load_browser_session,
        restore_backend_session,
        save_browser_session,
        serialize_backend_session,
    )
except ModuleNotFoundError:  # Streamlit executes this file with frontend/ on sys.path.
    from api_client import (  # type: ignore[no-redef]
        clear_client_cookies,
        close_persistent_client,
        get_persistent_client,
        normalize_base_url,
        request_json,
    )
    from browser_session import (  # type: ignore[no-redef]
        browser_cookie_password,
        clear_browser_session,
        load_browser_session,
        restore_backend_session,
        save_browser_session,
        serialize_backend_session,
    )

st.set_page_config(page_title="在线评测系统", page_icon="⚖️", layout="wide")

DEFAULT_API_BASE_URL = normalize_base_url(os.getenv("OJ_API_BASE_URL", "http://127.0.0.1:8000"))
ROLE_LABELS = {"admin": "管理员", "user": "普通用户", "banned": "已封禁"}
STATUS_LABELS = {
    "AC": "AC · 通过",
    "WA": "WA · 答案错误",
    "TLE": "TLE · 超出时间限制",
    "MLE": "MLE · 超出内存限制",
    "RE": "RE · 运行时错误",
    "CE": "CE · 编译错误",
    "UNK": "UNK · 未知错误",
    "pending": "Pending · 评测中",
    "error": "Error · 任务异常",
    "200": "允许访问",
    "403": "拒绝访问",
}
STATUS_STYLES = {
    "AC": ("#067647", "#ecfdf3"),
    "WA": ("#b42318", "#fef3f2"),
    "TLE": ("#b54708", "#fffaeb"),
    "MLE": ("#6941c6", "#f4f3ff"),
    "RE": ("#c11574", "#fdf2fa"),
    "CE": ("#3538cd", "#eef4ff"),
    "UNK": ("#475467", "#f2f4f7"),
    "pending": ("#175cd3", "#eff8ff"),
    "error": ("#b42318", "#fef3f2"),
    "200": ("#067647", "#ecfdf3"),
    "403": ("#b42318", "#fef3f2"),
}
REASONING_LABELS = {
    "auto": "自动（提供商默认）",
    "low": "低",
    "medium": "中",
    "high": "高",
    "max": "最高",
}
MODEL_COMPATIBILITY = [
    {
        "提供商": "阿里云百炼 / Qwen",
        "Base URL 示例": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "模型示例": "qwen3.7-plus、qwen3.7-flash",
        "推理控制": "thinking_budget / reasoning_effort",
    },
    {
        "提供商": "DeepSeek",
        "Base URL 示例": "https://api.deepseek.com",
        "模型示例": "deepseek-v4-flash、deepseek-v4-pro",
        "推理控制": "reasoning_effort",
    },
    {
        "提供商": "Kimi / Moonshot",
        "Base URL 示例": "https://api.moonshot.cn/v1",
        "模型示例": "kimi-k2.6、kimi-k2.5",
        "推理控制": "thinking 开关；档位同时写入提示词",
    },
    {
        "提供商": "其他 OpenAI-compatible",
        "Base URL 示例": "服务商给出的 /v1 地址",
        "模型示例": "OpenAI、GLM、OpenRouter、SiliconFlow 等",
        "推理控制": "reasoning_effort；不支持时自动回退",
    },
]
LANGUAGE_COMPATIBILITY = [
    {
        "语言": "Python 3（内置）",
        "扩展名": ".py",
        "编译命令": "—",
        "运行命令": "python3 {src}",
        "所需程序": "python3",
    },
    {
        "语言": "C++14（内置）",
        "扩展名": ".cpp",
        "编译命令": "g++ {src} -std=c++14 -O2 -o {exe}",
        "运行命令": "{exe}",
        "所需程序": "g++",
    },
    {
        "语言": "C11 / C17",
        "扩展名": ".c",
        "编译命令": "gcc {src} -std=c17 -O2 -o {exe}",
        "运行命令": "{exe}",
        "所需程序": "gcc",
    },
    {
        "语言": "C++17",
        "扩展名": ".cpp",
        "编译命令": "g++ {src} -std=c++17 -O2 -o {exe}",
        "运行命令": "{exe}",
        "所需程序": "g++ 或 clang++",
    },
    {
        "语言": "C++20",
        "扩展名": ".cpp",
        "编译命令": "g++ {src} -std=c++20 -O2 -o {exe}",
        "运行命令": "{exe}",
        "所需程序": "g++ 或 clang++",
    },
    {
        "语言": "Java 11+",
        "扩展名": ".java",
        "编译命令": "—",
        "运行命令": "java {src}",
        "所需程序": "java",
    },
    {
        "语言": "JavaScript",
        "扩展名": ".js",
        "编译命令": "—",
        "运行命令": "node {src}",
        "所需程序": "node",
    },
    {
        "语言": "Ruby",
        "扩展名": ".rb",
        "编译命令": "—",
        "运行命令": "ruby {src}",
        "所需程序": "ruby",
    },
    {
        "语言": "Go",
        "扩展名": ".go",
        "编译命令": "go build -o {exe} {src}",
        "运行命令": "{exe}",
        "所需程序": "go",
    },
]


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}
        .oj-brand {font-size: 1.2rem; font-weight: 750; color: #193b68; margin-bottom: .2rem;}
        .oj-muted {color: #667085; font-size: .9rem;}
        .oj-user-card {padding: .8rem .9rem; border: 1px solid #d0d5dd; border-radius: .7rem;
                       background: #f8fafc; margin: .6rem 0;}
        .oj-user-name {font-weight: 700; color: #101828;}
        .oj-badge {display: inline-flex; align-items: center; padding: .18rem .55rem;
                   border-radius: 999px; font-size: .78rem; font-weight: 750; line-height: 1.4;}
        .oj-section-note {padding: .75rem 1rem; border-left: 4px solid #2e6acf;
                          background: #f5f8ff; border-radius: .25rem; color: #344054;}
        div[data-testid="stMetric"] {border: 1px solid #eaecf0; padding: .75rem 1rem;
                                     border-radius: .65rem; background: white;}
        .st-key-top_navigation {padding: .8rem 1rem .35rem; border: 1px solid #e4e7ec;
                                border-radius: .9rem; background: #ffffff;
                                box-shadow: 0 4px 16px rgba(16, 24, 40, .05);}
        .st-key-main_navigation div[role="radiogroup"] {gap: .35rem;}
        .st-key-main_navigation button {border-radius: .6rem !important; font-weight: 650;}
        .oj-top-user {text-align: right; font-size: .88rem; color: #475467; padding-top: .2rem;}
        .oj-top-user strong {display: block; color: #101828; font-size: .95rem;}
        @media (max-width: 720px) {
          .block-container {padding: 1rem .8rem 2rem;}
          h1 {font-size: 1.65rem !important;}
          h2 {font-size: 1.3rem !important;}
          .oj-user-card {padding: .65rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_client() -> httpx.Client:
    return get_persistent_client(st.session_state, DEFAULT_API_BASE_URL)


def initialize_browser_session() -> None:
    try:
        store = EncryptedCookieManager(
            prefix="online-judge/",
            password=browser_cookie_password(),
        )
    except Exception:
        st.session_state["_browser_cookie_store"] = None
        return
    st.session_state["_browser_cookie_store"] = store
    if not store.ready():
        return

    pending = st.session_state.pop("_browser_session_pending", None)
    if isinstance(pending, str):
        save_browser_session(store, pending)
    if st.session_state.get("login") or st.session_state.get("_browser_restore_attempted"):
        return

    st.session_state["_browser_restore_attempted"] = True
    if not restore_backend_session(get_client(), load_browser_session(store)):
        return
    restored = request_json(get_client(), "GET", "/api/auth/session")
    if restored.get("code") == 200:
        st.session_state["login"] = restored["data"]
        return
    clear_client_cookies(st.session_state)
    clear_browser_session(store)


def persist_current_browser_session() -> bool:
    payload = serialize_backend_session(get_client())
    if payload is None:
        return False
    st.session_state["_browser_session_pending"] = payload
    store = st.session_state.get("_browser_cookie_store")
    if store is not None and save_browser_session(store, payload):
        st.session_state.pop("_browser_session_pending", None)
        st.session_state["_browser_restore_attempted"] = True
        return True
    return False


def clear_persisted_browser_session() -> None:
    store = st.session_state.get("_browser_cookie_store")
    clear_browser_session(store)
    st.session_state.pop("_browser_session_pending", None)
    st.session_state.pop("_browser_restore_attempted", None)


def clear_local_session(*, close_client: bool = False) -> None:
    clear_persisted_browser_session()
    for key in (
        "login",
        "last_submission_id",
        "submission_detail",
        "submission_log",
        "ai_task_id",
        "ai_problem_draft",
        "selected_submission_id",
        "problem_section",
        "next_problem_section",
        "judge_problem_id",
        "main_navigation",
        "next_navigation",
    ):
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if str(key).startswith(("judge_code_", "judge_editor_", "submit_code_", "ai_config_")):
            st.session_state.pop(key, None)
    if close_client:
        close_persistent_client(st.session_state)
    else:
        clear_client_cookies(st.session_state)


def set_flash(level: str, message: str) -> None:
    st.session_state["flash_message"] = (level, message)


def navigate_on_next_rerun(page: str) -> None:
    st.session_state["next_navigation"] = page


def show_flash() -> None:
    flash = st.session_state.pop("flash_message", None)
    if not flash:
        return
    level, message = flash
    getattr(st, level, st.info)(message)


def role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role)


def status_badge(status: str) -> str:
    foreground, background = STATUS_STYLES.get(status, ("#344054", "#f2f4f7"))
    label = STATUS_LABELS.get(status, status)
    return (
        f'<span class="oj-badge" style="color:{foreground};background:{background};">'
        f"{escape(label)}</span>"
    )


def submission_verdict(data: dict[str, Any]) -> str:
    if data.get("status") == "pending":
        return "pending"
    if data.get("status") == "error":
        return "error"
    result = str(data.get("result") or "").upper()
    if result in {"AC", "WA", "TLE", "MLE", "RE", "CE", "UNK"}:
        return result
    details = data.get("details") or []
    for case in details:
        case_result = str(case.get("result") or "UNK").upper()
        if case_result != "AC":
            return case_result if case_result in STATUS_STYLES else "UNK"
    if details:
        return "AC"
    if (data.get("compile_info") or {}).get("result") == "failed":
        return "CE"
    if data.get("counts") and data.get("score") == data.get("counts"):
        return "AC"
    if data.get("counts") is not None and data.get("score", 0) < data.get("counts", 0):
        return "WA"
    return "UNK"


def output_calibration_rows(validation: dict[str, Any]) -> list[dict[str, Any]]:
    calibration = validation.get("output_calibration") or {}
    labels = {"sample": "样例", "testcase": "隐藏测试点"}
    return [
        {
            "类型": labels.get(str(item.get("kind")), str(item.get("kind") or "—")),
            "序号": item.get("index", "—"),
            "输入": item.get("input", ""),
            "模型原答案": item.get("original_output", ""),
            "校准答案": item.get("calibrated_output", ""),
        }
        for item in calibration.get("items") or []
    ]


def validity_check_rows(validation: dict[str, Any]) -> list[dict[str, Any]]:
    report = validation.get("testcase_validity") or {}
    return [
        {
            "检查项": check.get("label", check.get("key", "—")),
            "结果": "通过" if check.get("passed") else "需复核",
            "依据": check.get("detail", ""),
        }
        for check in report.get("checks") or []
    ]


def resource_limit_summary(validation: dict[str, Any]) -> str | None:
    policy = validation.get("resource_limits") or {}
    final = policy.get("final") or {}
    if final.get("time_limit") is None or final.get("memory_limit") is None:
        return None
    source_label = {
        "explicit": "按命题需求指定",
        "mixed": "指定值与自动评估结合",
        "automatic": "按难度和算法结构自动评估",
    }.get(policy.get("source"), "自动评估")
    reasons = "、".join(str(item) for item in policy.get("reasons") or [])
    summary = f"资源策略：{source_label} · {final['time_limit']} s · {final['memory_limit']} MB"
    return f"{summary} · {reasons}" if reasons else summary


def compact_preview(value: str, limit: int = 2_000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"\n…（共 {len(value)} 字符，页面仅显示前 {limit} 字符）"


def resolve_problem_selection(
    problem_ids: list[str], current: str | None, requested: str | None = None
) -> str | None:
    if requested in problem_ids:
        return requested
    if current in problem_ids:
        return current
    return problem_ids[0] if problem_ids else None


def submission_list_params(
    login: dict[str, Any],
    *,
    problem_id: str,
    page: int,
    selected_user_id: str | None = None,
) -> dict[str, Any] | None:
    user_id = selected_user_id if login.get("role") == "admin" else str(login["user_id"])
    if not user_id and not problem_id:
        return None
    params: dict[str, Any] = {"page": page, "page_size": 5}
    if user_id:
        params["user_id"] = user_id
    if problem_id:
        params["problem_id"] = problem_id
    return params


def api_call(method: str, path: str, *, quiet: bool = False, **kwargs: Any) -> dict[str, Any]:
    result = request_json(get_client(), method, path, **kwargs)
    if result.get("code") == 401 and st.session_state.get("login"):
        clear_local_session()
        st.warning("登录状态已失效，请重新登录。")
    if result.get("code") != 200 and not quiet:
        st.error(f"{result.get('code', '错误')}：{result.get('msg', '请求失败')}")
    return result


def logout_user() -> None:
    result = request_json(get_client(), "POST", "/api/auth/logout")
    clear_local_session(close_client=True)
    if result.get("code") == 200:
        set_flash("success", "已安全退出登录。")
    else:
        set_flash("warning", "本地登录状态已清理；后端退出请求未成功。")
    st.rerun()


def require_login() -> dict[str, Any] | None:
    login = st.session_state.get("login")
    if not login:
        st.warning("请先登录后再访问此页面。")
        return None
    return login


def pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def parse_json_list(value: str, field: str) -> list[Any] | None:
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError
        return parsed
    except (json.JSONDecodeError, ValueError):
        st.error(f"{field} 必须是合法的 JSON 数组。")
        return None


def page_account() -> None:
    st.title("账户中心")
    login = st.session_state.get("login")
    if not login:
        login_tab, register_tab = st.tabs(["登录账户", "注册账户"])
        with login_tab:
            st.markdown("#### 欢迎回来")
            with st.form("login_form"):
                username = st.text_input("用户名 *", placeholder="请输入用户名")
                password = st.text_input("密码 *", type="password", placeholder="请输入密码")
                submitted = st.form_submit_button("登录", type="primary", width="stretch")
            if submitted:
                if not username.strip() or not password:
                    st.error("用户名和密码均为必填项。")
                else:
                    with st.spinner("正在验证登录信息……"):
                        result = api_call(
                            "POST",
                            "/api/auth/login",
                            json={"username": username.strip(), "password": password},
                        )
                    if result.get("code") == 200:
                        st.session_state.login = result["data"]
                        persist_current_browser_session()
                        navigate_on_next_rerun("账户概览")
                        set_flash("success", f"欢迎回来，{result['data']['username']}。")
                        st.rerun()
        with register_tab:
            st.markdown("#### 创建普通用户")
            with st.form("register_form"):
                username = st.text_input("新用户名 *", help="长度为 3–40 个字符")
                password = st.text_input("新密码 *", type="password", help="至少 6 位，最长 256 位")
                confirmed_password = st.text_input("确认密码 *", type="password")
                submitted = st.form_submit_button("注册", width="stretch")
            if submitted:
                if len(username.strip()) < 3 or len(password) < 6:
                    st.error("用户名至少 3 个字符，密码至少 6 位。")
                elif password != confirmed_password:
                    st.error("两次输入的密码不一致。")
                else:
                    with st.spinner("正在创建账户……"):
                        result = api_call(
                            "POST",
                            "/api/users/",
                            json={"username": username.strip(), "password": password},
                        )
                    if result.get("code") == 200:
                        st.success("注册成功，请切换到“登录账户”。")
        return

    st.markdown(f"### {escape(login['username'])}")
    with st.spinner("正在读取账户资料……"):
        profile = api_call("GET", f"/api/users/{login['user_id']}", quiet=True)
    if profile.get("code") != 200:
        if profile.get("code") not in {401, 503}:
            st.error(profile.get("msg", "资料加载失败"))
        return

    data = profile["data"]
    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("累计提交", data["submit_count"])
    metric2.metric("通过题目", data["resolve_count"])
    ratio = 0 if not data["submit_count"] else data["resolve_count"] / data["submit_count"]
    metric3.metric("通过/提交", f"{ratio:.1%}")
    with st.container(border=True):
        left, right = st.columns(2)
        left.markdown(f"**用户名**  \n{escape(data['username'])}")
        left.markdown(f"**加入日期**  \n{data['join_time']}")
        right.markdown(f"**角色**  \n{role_label(data['role'])}")
        right.markdown(f"**用户 ID**  \n{data['user_id']}")

    with st.expander("修改登录密码"):
        with st.form("change_password_form", clear_on_submit=True):
            current_password = st.text_input("当前密码 *", type="password")
            new_password = st.text_input("新密码 *", type="password", help="至少 6 位，最长 256 位")
            confirmed_password = st.text_input("确认新密码 *", type="password")
            change_submitted = st.form_submit_button("保存新密码", type="primary")
        if change_submitted:
            if not current_password or len(new_password) < 6:
                st.error("请填写当前密码，且新密码不能少于 6 位。")
            elif new_password != confirmed_password:
                st.error("两次输入的新密码不一致。")
            elif current_password == new_password:
                st.error("新密码不能与当前密码相同。")
            else:
                with st.spinner("正在安全更新密码……"):
                    changed = api_call(
                        "PUT",
                        f"/api/users/{login['user_id']}/password",
                        json={
                            "current_password": current_password,
                            "new_password": new_password,
                        },
                    )
                if changed.get("code") == 200:
                    st.success("密码已更新。")


def problem_form(
    key: str, initial: dict[str, Any] | None = None, *, lock_id: bool = False
) -> dict[str, Any] | None:
    initial = initial or {}
    st.caption("标有 * 的字段为必填项；样例与测试点使用 JSON 数组格式。")
    with st.form(key):
        left, right = st.columns(2)
        with left:
            problem_id = st.text_input(
                "题目 ID *", value=str(initial.get("id", "")), disabled=lock_id
            )
            title = st.text_input("标题 *", value=str(initial.get("title", "")))
            difficulty = st.text_input(
                "难度",
                value=str(initial.get("difficulty", "中等")),
                help="如：入门、简单、中等、困难",
            )
            author = st.text_input("作者", value=str(initial.get("author", "")))
            source = st.text_input("来源", value=str(initial.get("source", "")))
        with right:
            time_limit = st.number_input(
                "时间限制（秒）", 0.01, 60.0, float(initial.get("time_limit", 3.0)), 0.1
            )
            memory_limit = st.number_input(
                "内存限制（MB）", 1, 4096, int(initial.get("memory_limit", 128))
            )
            tags_text = st.text_area(
                "标签（JSON 数组）", value=pretty_json(initial.get("tags", [])), height=100
            )
            hint = st.text_area("提示", value=str(initial.get("hint", "")), height=100)
        description = st.text_area(
            "题目描述 *", value=str(initial.get("description", "")), height=140
        )
        input_description = st.text_area(
            "输入格式 *", value=str(initial.get("input_description", "")), height=90
        )
        output_description = st.text_area(
            "输出格式 *", value=str(initial.get("output_description", "")), height=90
        )
        constraints = st.text_area(
            "数据范围 *", value=str(initial.get("constraints", "")), height=90
        )
        samples_text = st.text_area(
            "样例 *（JSON 数组）", value=pretty_json(initial.get("samples", [])), height=160
        )
        testcases_text = st.text_area(
            "测试点 *（JSON 数组）",
            value=pretty_json(initial.get("testcases", [])),
            height=240,
            help='每项格式为 {"input": "...", "output": "..."}',
        )
        submitted = st.form_submit_button("校验并保存", type="primary", width="stretch")
    if not submitted:
        return None
    tags = parse_json_list(tags_text, "标签")
    samples = parse_json_list(samples_text, "样例")
    testcases = parse_json_list(testcases_text, "测试点")
    if tags is None or samples is None or testcases is None:
        return None
    required_text = {
        "题目 ID": problem_id,
        "标题": title,
        "题目描述": description,
        "输入格式": input_description,
        "输出格式": output_description,
        "数据范围": constraints,
    }
    missing = [label for label, value in required_text.items() if not value.strip()]
    if missing:
        st.error(f"请填写必填字段：{'、'.join(missing)}。")
        return None
    if not samples or not testcases:
        st.error("至少需要 1 组样例和 1 个测试点。")
        return None
    return {
        "id": problem_id.strip(),
        "title": title.strip(),
        "description": description,
        "input_description": input_description,
        "output_description": output_description,
        "samples": samples,
        "constraints": constraints,
        "testcases": testcases,
        "hint": hint,
        "source": source,
        "tags": tags,
        "time_limit": time_limit,
        "memory_limit": memory_limit,
        "author": author,
        "difficulty": difficulty,
    }


def load_problem_overviews(
    problems: list[dict[str, Any]], login: dict[str, Any]
) -> list[dict[str, Any]]:
    overviews: list[dict[str, Any]] = []
    for item in problems:
        detail_result = api_call("GET", f"/api/problems/{item['id']}", quiet=True)
        if detail_result.get("code") != 200:
            if not st.session_state.get("login"):
                break
            continue
        attempts_result = api_call(
            "GET",
            "/api/submissions/",
            quiet=True,
            params={
                "user_id": login["user_id"],
                "problem_id": item["id"],
                "page_size": 100,
            },
        )
        submissions = (
            attempts_result.get("data", {}).get("submissions", [])
            if attempts_result.get("code") == 200
            else []
        )
        attempts = (
            attempts_result.get("data", {}).get("total", 0)
            if attempts_result.get("code") == 200
            else 0
        )
        accepted = any(submission_verdict(record) == "AC" for record in submissions)
        latest_verdict = submission_verdict(submissions[0]) if submissions else None
        state = "✅ 已通过" if accepted else ("🟠 已尝试" if attempts else "⚪ 未尝试")
        overviews.append(
            {
                "detail": detail_result["data"],
                "attempts": attempts,
                "accepted": accepted,
                "latest_verdict": latest_verdict,
                "state": state,
            }
        )
    return overviews


def render_problem_statement(
    data: dict[str, Any], *, attempts: int | None = None, state: str | None = None
) -> None:
    with st.container(border=True):
        st.subheader(f"{data['id']} · {data['title']}")
        metadata = [
            f"难度：{data.get('difficulty') or '未标注'}",
            f"来源：{data.get('source') or '课程题库'}",
        ]
        if state:
            metadata.append(state)
        st.caption(" · ".join(metadata))

        limit_columns = st.columns(4 if attempts is not None else 3)
        limit_columns[0].metric("时间限制", f"{data['time_limit']} s")
        limit_columns[1].metric("内存限制", f"{data['memory_limit']} MB")
        testcase_count = len(data.get("testcases") or []) + int(
            data.get("file_testcase_count", 0) or 0
        )
        limit_columns[2].metric("测试点", testcase_count)
        if attempts is not None:
            limit_columns[3].metric("个人提交", attempts)

        st.markdown("#### 题目描述")
        st.markdown(data["description"])
        info_left, info_right = st.columns(2)
        with info_left:
            st.markdown("**输入格式**")
            st.markdown(data["input_description"])
        with info_right:
            st.markdown("**输出格式**")
            st.markdown(data["output_description"])
        st.markdown("**数据范围**")
        st.markdown(data["constraints"])
        if data.get("hint"):
            st.info(f"提示：{data['hint']}")

        samples = data.get("samples") or []
        with st.expander("输入输出样例", expanded=True):
            if not samples:
                st.caption("本题没有公开样例。")
            for index, sample in enumerate(samples, start=1):
                st.markdown(f"**样例 {index}**")
                sample_in, sample_out = st.columns(2)
                sample_in.caption("输入")
                sample_in.code(sample.get("input", ""), language=None)
                sample_out.caption("输出")
                sample_out.code(sample.get("output", ""), language=None)


def page_problems() -> None:
    login = require_login()
    if not login:
        return
    st.title("题库与评测")
    result = api_call("GET", "/api/problems/", quiet=True)
    problems = (result.get("data") or []) if result.get("code") == 200 else []
    sections = ["题库", "题目详情与提交", "新增题目", "编辑题目"]
    pending_section = st.session_state.pop("next_problem_section", None)
    if pending_section:
        st.session_state.problem_section = {
            "题库概览": "题库",
            "题目与提交": "题目详情与提交",
        }.get(pending_section, pending_section)
    if st.session_state.get("problem_section") not in sections:
        st.session_state.problem_section = "题库"
    section = st.segmented_control(
        "题库功能",
        sections,
        key="problem_section",
        selection_mode="single",
        required=True,
        width="stretch",
        label_visibility="collapsed",
    )

    if section == "题库":
        if not problems:
            st.info("暂无题目。")
        else:
            with st.spinner("正在整理题目与个人通过状态……"):
                overviews = load_problem_overviews(problems, login)
            for overview in overviews:
                data = overview["detail"]
                with st.container(border=True):
                    title_col, state_col, action_col = st.columns([5, 2, 1.2])
                    title_col.markdown(f"### {escape(data['id'])} · {escape(data['title'])}")
                    difficulty = data.get("difficulty") or "未标注"
                    metadata = [difficulty]
                    metadata.extend(tag for tag in (data.get("tags") or []) if tag != difficulty)
                    title_col.caption(" · ".join(metadata))
                    state_col.caption(overview["state"])
                    if overview["latest_verdict"]:
                        state_col.markdown(
                            status_badge(overview["latest_verdict"]), unsafe_allow_html=True
                        )
                    else:
                        state_col.write(f"提交 {overview['attempts']} 次")
                    if action_col.button(
                        "进入题目", key=f"open_problem_{data['id']}", width="stretch"
                    ):
                        st.session_state.judge_problem_id = data["id"]
                        st.session_state.next_problem_section = "题目详情与提交"
                        st.rerun()
            if login["role"] == "admin":
                choices = {
                    f"{overview['detail']['id']} · {overview['detail']['title']}": overview
                    for overview in overviews
                }
                selected_label = st.selectbox("选择管理题目", list(choices), key="problem_browser")
                data = choices[selected_label]["detail"]
                problem_id = data["id"]
                with st.expander("管理员题目操作"):
                    visibility = st.toggle(
                        "向所有登录用户公开测试点日志",
                        value=data.get("public_cases", False),
                        key=f"visibility_{problem_id}",
                    )
                    if st.button("保存日志可见性", key=f"save_visibility_{problem_id}"):
                        with st.spinner("正在保存……"):
                            updated = api_call(
                                "PUT",
                                f"/api/problems/{problem_id}/log_visibility",
                                json={"public_cases": visibility},
                            )
                        if updated.get("code") == 200:
                            st.success("日志可见性已更新。")
                    confirm_delete = st.checkbox(
                        f"确认删除题目 {problem_id}", key=f"confirm_delete_{problem_id}"
                    )
                    if st.button(
                        "删除题目",
                        disabled=not confirm_delete,
                        key=f"delete_{problem_id}",
                    ):
                        with st.spinner("正在删除题目……"):
                            deleted = api_call("DELETE", f"/api/problems/{problem_id}")
                        if deleted.get("code") == 200:
                            set_flash("success", "题目已删除。")
                            st.rerun()
    elif section == "题目详情与提交":
        render_problem_workspace()
    elif section == "新增题目":
        draft = st.session_state.get("ai_problem_draft")
        if draft:
            st.info("已载入 AI 命题草稿。请人工审阅所有字段后再保存。")
        payload = problem_form("create_problem", draft)
        if payload:
            with st.spinner("正在保存题目……"):
                created = api_call("POST", "/api/problems/", json=payload)
            if created.get("code") == 200:
                st.session_state.pop("ai_problem_draft", None)
                st.session_state.next_problem_section = "题库"
                set_flash("success", f"题目 {payload['id']} 已创建。")
                st.rerun()
    elif section == "编辑题目":
        if not problems:
            st.info("暂无可编辑题目。")
        else:
            ids = [item["id"] for item in problems]
            selected_id = st.selectbox("选择待编辑题目", ids, key="problem_editor")
            detail = api_call("GET", f"/api/problems/{selected_id}", quiet=True)
            if detail.get("code") != 200:
                return
            payload = problem_form(f"edit_problem_{selected_id}", detail["data"], lock_id=True)
            if payload:
                with st.spinner("正在更新题目……"):
                    updated = api_call("PUT", f"/api/problems/{selected_id}", json=payload)
                if updated.get("code") == 200:
                    set_flash("success", f"题目 {selected_id} 已更新。")
                    st.rerun()


@st.fragment(run_every=1.0)
def live_submission_panel() -> None:
    submission_id = st.session_state.get("last_submission_id")
    if not submission_id:
        return
    result = api_call("GET", f"/api/submissions/{submission_id}", quiet=True)
    if result.get("code") != 200:
        if result.get("code") != 401:
            st.error(result.get("msg", "查询失败"))
        return
    data = result["data"]
    if data["status"] == "pending":
        st.markdown(status_badge("pending"), unsafe_allow_html=True)
        st.caption(f"提交 #{submission_id} 正在编译或运行，请稍候……")
        st.progress(35)
    elif data["status"] == "success":
        verdict = submission_verdict(data)
        st.markdown(status_badge(verdict), unsafe_allow_html=True)
        st.caption(f"提交 #{submission_id} · 得分 {data['score']} / {data['counts']}")
        progress = 0 if not data["counts"] else data["score"] / data["counts"]
        st.progress(progress)
    else:
        st.markdown(status_badge("error"), unsafe_allow_html=True)
        st.error(data.get("error_info") or "评测任务异常")


def render_submission_log(data: dict[str, Any]) -> None:
    details = data.get("details") or []
    passed = sum(case.get("result") == "AC" for case in details)
    details_visible = "details" in data
    metrics = st.columns(3 if details_visible else 2)
    score_col, count_col = metrics[:2]
    score_col.metric("得分", data.get("score", 0))
    count_col.metric("总分", data.get("counts", 0))
    if details_visible:
        metrics[2].metric("通过测试点", f"{passed} / {len(details)}")
    compile_info = data.get("compile_info") or {}
    if compile_info.get("result") == "failed":
        st.error(f"编译失败：{compile_info.get('message') or '请检查语法和编译选项'}")
    if data.get("error_info"):
        st.error(data["error_info"])
    if not details:
        message = "暂无测试点明细。" if details_visible else "该题未公开测试点明细。"
        st.info(message)
        return
    st.markdown("#### 测试点结果")
    for case in details:
        with st.container(border=True):
            case_col, result_col, time_col, memory_col = st.columns([1.2, 1, 1, 1])
            case_col.markdown(f"**测试点 #{case.get('id', '—')}**")
            if case.get("source") == "file":
                case_col.caption("文件压力点")
            result_col.markdown(
                status_badge(str(case.get("result", "UNK"))), unsafe_allow_html=True
            )
            time_col.caption("运行时间")
            time_col.write(f"{case.get('time', 0):.4f} s")
            memory_col.caption("峰值内存")
            memory_col.write(f"{case.get('memory', 0):.3f} MB")


def _starter_code(language: str) -> str:
    if language == "cpp":
        return "#include <iostream>\nusing namespace std;\n\nint main() {\n    // 在这里编写代码\n    return 0;\n}\n"
    if language == "python":
        return "# 在这里编写代码\n"
    return ""


def _editor_language(language: str) -> str:
    return {"cpp": "c_cpp", "python": "python"}.get(language, "text")


def _display_language(language: str) -> str:
    return {"cpp": "cpp", "python": "python"}.get(language, "text")


def render_submission_detail(submission_id: str, login: dict[str, Any]) -> None:
    detail_result = api_call("GET", f"/api/submissions/{submission_id}", quiet=True)
    if detail_result.get("code") != 200:
        st.error(detail_result.get("msg", "提交详情加载失败"))
        return
    data = detail_result["data"]
    verdict = submission_verdict(data)
    with st.container(border=True):
        head, action = st.columns([5, 1])
        with head:
            st.markdown(status_badge(verdict), unsafe_allow_html=True)
            st.caption(
                f"提交 #{submission_id} · 题目 {data.get('problem_id', '—')} · "
                f"{data.get('language', '—')} · {data.get('created_at', '')}"
            )
        if login["role"] == "admin" and action.button(
            "重新评测", key=f"rejudge_{submission_id}", width="stretch"
        ):
            result = api_call("PUT", f"/api/submissions/{submission_id}/rejudge")
            if result.get("code") == 200:
                st.session_state.last_submission_id = submission_id
                st.success("已开始重新评测。")

        code_tab, cases_tab = st.tabs(["源代码", "测试点"])
        with code_tab:
            st.code(
                data.get("code", ""),
                language=_display_language(data.get("language", "")),
                line_numbers=True,
                wrap_lines=False,
                height=420,
            )
        with cases_tab:
            log_result = api_call("GET", f"/api/submissions/{submission_id}/log", quiet=True)
            if log_result.get("code") == 200:
                render_submission_log(log_result["data"])
            else:
                st.error(log_result.get("msg", "测试点日志加载失败"))


def render_problem_workspace() -> None:
    login = require_login()
    if not login:
        return
    problem_result = api_call("GET", "/api/problems/", quiet=True)
    language_result = api_call("GET", "/api/languages/", quiet=True)
    problems = problem_result.get("data") or []
    languages = (language_result.get("data") or {}).get("name", [])

    if not problems:
        st.info("题库为空，暂时无法提交。")
    elif not languages:
        st.warning("尚未配置可用语言，请联系管理员。")
    else:
        titles = {item["id"]: item["title"] for item in problems}
        problem_ids = list(titles)
        st.session_state.judge_problem_id = resolve_problem_selection(
            problem_ids,
            st.session_state.get("judge_problem_id"),
        )
        selected_problem_id = st.selectbox(
            "选择题目 *",
            problem_ids,
            key="judge_problem_id",
            format_func=lambda problem_id: f"{problem_id} · {titles[problem_id]}",
        )
        selected_detail = api_call("GET", f"/api/problems/{selected_problem_id}", quiet=True)
        if selected_detail.get("code") == 200:
            render_problem_statement(selected_detail["data"])
        else:
            st.error(selected_detail.get("msg", "题目信息加载失败"))

        with st.container(border=True):
            editor_title, editor_language = st.columns([3, 1])
            editor_title.markdown("### 提交代码")
            language = editor_language.selectbox("语言 *", languages, key="judge_language")
            editor_response = code_editor(
                _starter_code(language),
                lang=_editor_language(language),
                theme="light",
                shortcuts="vscode",
                height=30,
                focus=True,
                allow_reset=True,
                buttons=[],
                response_mode=["debounce", "blur"],
                options={
                    "showLineNumbers": True,
                    "tabSize": 4,
                    "useSoftTabs": True,
                    "enableBasicAutocompletion": True,
                    "enableLiveAutocompletion": True,
                    "enableSnippets": True,
                    "behavioursEnabled": True,
                    "highlightActiveLine": True,
                    "showPrintMargin": False,
                    "wrap": True,
                },
                key=f"judge_editor_{selected_problem_id}_{language}",
            )
            response_id = editor_response.get("id")
            shortcut_submitted = (
                editor_response.get("type") == "submit"
                and response_id
                and response_id != st.session_state.get("last_editor_submit_id")
            )
            code_state_key = f"judge_code_{selected_problem_id}_{language}"
            if editor_response.get("text") is not None:
                st.session_state[code_state_key] = editor_response.get("text", "")
            button_submitted = st.button(
                "提交评测",
                type="primary",
                key=f"submit_code_{selected_problem_id}_{language}",
            )
            submitted = bool(shortcut_submitted or button_submitted)
            if submitted:
                if shortcut_submitted:
                    st.session_state.last_editor_submit_id = response_id
                code = st.session_state.get(code_state_key, editor_response.get("text", ""))
                if not code.strip():
                    st.error("源代码不能为空。")
                else:
                    with st.spinner("正在创建评测任务……"):
                        result = api_call(
                            "POST",
                            "/api/submissions/",
                            json={
                                "problem_id": selected_problem_id,
                                "language": language,
                                "code": code,
                            },
                        )
                    if result.get("code") == 200:
                        st.session_state.last_submission_id = result["data"]["submission_id"]
                        st.success(f"提交成功：#{result['data']['submission_id']}")
            live_submission_panel()


def page_submission_records() -> None:
    login = require_login()
    if not login:
        return
    st.title("提交记录")
    problem_result = api_call("GET", "/api/problems/", quiet=True)
    problems = problem_result.get("data") or []

    if login["role"] == "admin":
        users_result = api_call("GET", "/api/users/?page_size=200", quiet=True)
        users = (users_result.get("data") or {}).get("users", [])
        user_options = {
            f"本人 · {login['username']}": str(login["user_id"]),
            "全部用户（需选择题目）": "",
        }
        user_options.update(
            {
                f"{user['username']} (#{user['user_id']})": str(user["user_id"])
                for user in users
                if str(user["user_id"]) != str(login["user_id"])
            }
        )
        user_col, problem_col, refresh_col = st.columns([2, 2, 1])
        selected_user = user_col.selectbox(
            "用户筛选", list(user_options), key="submission_user_filter"
        )
        selected_user_id = user_options[selected_user]
    else:
        problem_col, refresh_col = st.columns([4, 1])
        selected_user_id = str(login["user_id"])
    filter_problem = problem_col.selectbox(
        "题目筛选",
        [""] + [item["id"] for item in problems],
        format_func=lambda x: x or "全部题目",
        key="submission_problem_filter",
    )
    refresh_col.write("")
    refresh_col.write("")
    refresh_col.button("刷新记录", width="stretch")
    filter_signature = (selected_user_id, filter_problem)
    if st.session_state.get("submission_filter_signature") != filter_signature:
        st.session_state.submission_filter_signature = filter_signature
        st.session_state.submission_page = 1
    current_page = int(st.session_state.get("submission_page", 1))
    params = submission_list_params(
        login,
        problem_id=filter_problem,
        page=current_page,
        selected_user_id=selected_user_id,
    )
    if params is None:
        st.info("查看全部用户提交时，请先选择一道题目。")
        return
    with st.spinner("正在读取提交记录……"):
        records = api_call("GET", "/api/submissions/", params=params, quiet=True)
    if records.get("code") != 200:
        if records.get("code") != 401:
            st.error(records.get("msg", "提交记录加载失败"))
        return

    total = int(records["data"]["total"])
    total_pages = max(1, (total + 4) // 5)
    if current_page > total_pages:
        st.session_state.submission_page = total_pages
        st.rerun()
    submissions = records["data"]["submissions"]
    if not submissions:
        st.info("暂无提交记录。")
        return

    previous_col, page_col, next_col = st.columns([1, 3, 1], vertical_alignment="center")
    if previous_col.button("上一页", disabled=current_page <= 1, width="stretch"):
        st.session_state.submission_page = current_page - 1
        st.rerun()
    page_col.markdown(
        f"<div style='text-align:center'>第 {current_page} / {total_pages} 页 · 共 {total} 条</div>",
        unsafe_allow_html=True,
    )
    if next_col.button("下一页", disabled=current_page >= total_pages, width="stretch"):
        st.session_state.submission_page = current_page + 1
        st.rerun()

    record_ids = [record["submission_id"] for record in submissions]
    if st.session_state.get("selected_submission_id") not in record_ids:
        st.session_state.selected_submission_id = record_ids[0]
    for record in submissions:
        verdict = submission_verdict(record)
        with st.container(border=True):
            identity, result_column, score_column, time_column, action_column = st.columns(
                [2.2, 2, 1, 2, 0.8]
            )
            identity.markdown(f"**#{record['submission_id']} · {record.get('problem_id', '—')}**")
            identity.caption(f"语言：{record.get('language', '—')}")
            result_column.caption("判题结果")
            result_column.markdown(status_badge(verdict), unsafe_allow_html=True)
            score_column.caption("得分")
            score_column.write(
                "—"
                if record["status"] == "pending"
                else f"{record.get('score', 0)} / {record.get('counts', 0)}"
            )
            time_column.caption("提交时间")
            time_column.write(str(record.get("created_at", "—")).replace("T", " ")[:19])
            action_column.write("")
            if action_column.button(
                "详情",
                key=f"view_submission_{record['submission_id']}",
                width="stretch",
            ):
                st.session_state.selected_submission_id = record["submission_id"]

    st.markdown("### 提交详情")
    render_submission_detail(st.session_state.selected_submission_id, login)


@st.fragment(run_every=1.0)
def live_ai_task_panel() -> None:
    task_id = st.session_state.get("ai_task_id")
    if not task_id:
        return
    response = api_call("GET", f"/api/ai/problem-tasks/{task_id}", quiet=True)
    if response.get("code") != 200:
        st.error(response.get("msg", "任务查询失败"))
        return
    data = response["data"]
    st.progress(data.get("progress_percent", 0), text=data.get("progress", ""))
    usage = data.get("usage") or {}
    effort = str(usage.get("reasoning_effort", "auto"))
    st.caption(
        f"模型配置：{usage.get('model_config_name', '—')} · 模型：{usage.get('model', '—')} · "
        f"推理强度：{REASONING_LABELS.get(effort, effort)}"
    )
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("输入 Token", usage.get("input_tokens", 0))
    col2.metric("输出 Token（含思考）", usage.get("output_tokens", 0))
    col3.metric("模型请求", usage.get("request_count", 0))
    currency = usage.get("currency", "CNY")
    cost = float(usage.get("cost", 0) or 0)
    col4.metric("本次花费", f"¥{cost:.2f}" if currency == "CNY" else f"{cost:.2f} {currency}")
    detail_parts = [f"提供商合计 {usage.get('provider_total_tokens', 0)} Token"]
    if usage.get("reasoning_tokens"):
        detail_parts.append(f"其中思考 {usage['reasoning_tokens']} Token（已包含在输出中）")
    if usage.get("cached_input_tokens"):
        detail_parts.append(f"缓存输入 {usage['cached_input_tokens']} Token")
    st.caption(" · ".join(detail_parts))
    calls = usage.get("calls") or []
    if calls:
        with st.expander("查看分阶段 Token 明细"):
            st.dataframe(
                [
                    {
                        "阶段": call.get("stage", "—"),
                        "请求次数": call.get("request_count", 0),
                        "输入 Token": call.get("input_tokens", 0),
                        "输出 Token": call.get("output_tokens", 0),
                        "其中思考": call.get("reasoning_tokens", 0),
                    }
                    for call in calls
                ],
                hide_index=True,
                width="stretch",
            )
    if usage.get("estimated"):
        st.warning("部分请求未返回完整 usage 或发生过重试，Token 与费用为当前可得数据的估算值。")
    if usage.get("reasoning_fallback"):
        st.info("该提供商不接受所选推理参数，已自动使用模型默认推理设置，命题流程未中断。")
    if data["status"] in {"pending", "running"}:
        stream_result = data.get("result") or {}
        stream_preview = stream_result.get("stream_preview", "")
        if stream_preview:
            stage = stream_result.get("stream_stage", "模型输出")
            with st.expander(f"实时输出 · {stage}", expanded=True):
                st.code(stream_preview, language=None, wrap_lines=True)
        if st.button("中断任务", key=f"cancel_{task_id}"):
            cancelled = api_call("PUT", f"/api/ai/problem-tasks/{task_id}/cancel")
            if cancelled.get("code") == 200:
                st.warning("任务已中断")
    elif data["status"] == "completed":
        st.success("AI 命题已完成")
        result = data["result"]
        problem = result.get("problem", {})
        validation = result.get("validation") or {}
        resource_summary = resource_limit_summary(validation)
        if resource_summary:
            st.caption(resource_summary)
        id_assignment = validation.get("id_assignment") or {}
        if id_assignment.get("source") == "automatic":
            st.caption(f"题目编号：{id_assignment.get('final_id', '—')}")
        calibration = validation.get("output_calibration") or {}
        if calibration.get("applied"):
            calibration_count = int(calibration.get("count", 0) or 0)
            st.info(f"Python/C++ 已校准 {calibration_count} 个答案。")
            with st.expander("查看答案校准明细"):
                st.dataframe(
                    output_calibration_rows(validation),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "输入": st.column_config.TextColumn(width="medium"),
                        "模型原答案": st.column_config.TextColumn(width="medium"),
                        "校准答案": st.column_config.TextColumn(width="medium"),
                    },
                )
        testcase_validity = validation.get("testcase_validity") or {}
        if testcase_validity:
            score = int(testcase_validity.get("score", 0) or 0)
            message = f"测试用例有效性：{score}/100"
            if testcase_validity.get("passed"):
                st.success(message)
            else:
                st.warning(f"{message}，请按检查明细人工补强。")
            with st.expander("查看测试用例有效性检查"):
                st.dataframe(
                    validity_check_rows(validation),
                    hide_index=True,
                    width="stretch",
                )
        file_stress = validation.get("file_stress") or {}
        if file_stress.get("applied"):
            st.info(
                f"已生成 {file_stress.get('count', 0)} 个文件压力点 · "
                f"{file_stress.get('label', '压力测试')} · "
                f"{file_stress.get('input_bytes', 0) / 1024:.1f} KiB"
            )
        elif file_stress.get("attempted"):
            st.caption(f"文件压力点未启用：{file_stress.get('reason', '未通过安全校验')}")
        with st.container(border=True):
            st.subheader(f"{problem.get('id', '—')} · {problem.get('title', '未命名题目')}")
            st.caption(
                f"难度：{problem.get('difficulty') or '未标注'} · "
                f"测试点：{len(problem.get('testcases') or []) + int(file_stress.get('count', 0) or 0)} · "
                f"时间：{problem.get('time_limit', '—')} s · "
                f"内存：{problem.get('memory_limit', '—')} MB"
            )
            st.markdown(problem.get("description", ""))
            with st.expander("查看参考解法与复杂度"):
                python_tab, cpp_tab = st.tabs(["Python 3", "C++14"])
                with python_tab:
                    st.code(
                        result.get("reference_solution", ""),
                        language="python",
                        line_numbers=True,
                    )
                with cpp_tab:
                    st.code(
                        result.get("reference_solution_cpp", ""),
                        language="cpp",
                        line_numbers=True,
                    )
                st.markdown(result.get("solution_explanation", ""))
            with st.expander("查看生成的测试点"):
                for index, case in enumerate(problem.get("testcases") or [], start=1):
                    left, right = st.columns(2)
                    left.caption(f"测试点 {index} · 输入")
                    left.code(compact_preview(case.get("input", "")), language=None)
                    right.caption(f"测试点 {index} · 输出")
                    right.code(compact_preview(case.get("output", "")), language=None)
        if st.button("载入到题目新增表单", key=f"load_{task_id}"):
            draft = dict(result["problem"])
            draft["author"] = usage.get("model") or "AI 模型"
            st.session_state.ai_problem_draft = draft
            st.session_state.next_problem_section = "新增题目"
            navigate_on_next_rerun("题库与评测")
            set_flash("success", "草稿已载入新增题目页，请人工审阅后保存。")
            st.rerun()
    elif data["status"] == "cancelled":
        st.warning("任务已中断，后台不会继续调用模型。")
    else:
        st.error(data.get("error") or "命题任务失败")


def page_ai() -> None:
    if not require_login():
        return
    st.header("AI 智能命题")
    configs_result = api_call("GET", "/api/ai/model-configs/", quiet=True)
    configured_models = (
        configs_result.get("data", {}).get("models", [])
        if configs_result.get("code") == 200
        else []
    )
    task_tab, config_tab = st.tabs(["智能命题", "模型配置"])

    with task_tab:
        if not configured_models:
            st.warning("请先在“模型配置”中添加模型。")
        problem_result = api_call("GET", "/api/problems/", quiet=True)
        existing_ids = [item["id"] for item in problem_result.get("data") or []]
        with st.form("ai_task"):
            model_names = [item["name"] for item in configured_models]
            selected_model = st.selectbox(
                "命题模型 *",
                model_names or ["暂无可用模型"],
                disabled=not model_names,
                format_func=lambda name: next(
                    (
                        f"{item['name']} · {item['model']}"
                        for item in configured_models
                        if item["name"] == name
                    ),
                    name,
                ),
            )
            requirement = st.text_area(
                "命题需求 *",
                height=150,
                placeholder="描述知识点、题型和特殊要求",
            )
            knowledge_text = st.text_input("知识点（逗号分隔）")
            difficulty_col, testcase_col, reasoning_col = st.columns(3)
            difficulty = difficulty_col.selectbox(
                "预期难度", ["自动", "入门", "简单", "中等", "困难"]
            )
            testcase_choice = testcase_col.selectbox(
                "隐藏测试点策略",
                ["自动"] + list(range(2, 11)),
            )
            reasoning_effort = reasoning_col.selectbox(
                "推理强度",
                list(REASONING_LABELS),
                format_func=lambda value: REASONING_LABELS[value],
            )
            problem_id = st.selectbox("参考/修改已有题目（可选）", [""] + existing_ids)
            submitted = st.form_submit_button(
                "开始智能命题", type="primary", width="stretch", disabled=not model_names
            )
        if submitted:
            if len(requirement.strip()) < 10:
                st.error("命题需求至少需要 10 个字符。")
            else:
                result = api_call(
                    "POST",
                    "/api/ai/problem-tasks/",
                    json={
                        "requirement": requirement,
                        "problem_id": problem_id or None,
                        "model_config_name": selected_model,
                        "knowledge_points": [
                            item.strip() for item in knowledge_text.split(",") if item.strip()
                        ],
                        "difficulty": difficulty,
                        "testcase_count": (None if testcase_choice == "自动" else testcase_choice),
                        "reasoning_effort": reasoning_effort,
                    },
                )
                if result.get("code") == 200:
                    st.session_state.ai_task_id = result["data"]["task_id"]
                    st.success(f"任务已创建：{result['data']['task_id']}")
        live_ai_task_panel()

    with config_tab:
        with st.expander("兼容模型与填写示例", expanded=not configured_models):
            st.caption(
                "支持使用 Bearer API Key 的任意 OpenAI-compatible Chat Completions 服务；"
                "下列三类有专项推理参数适配。"
            )
            st.dataframe(MODEL_COMPATIBILITY, hide_index=True, width="stretch")
        if configured_models:
            st.markdown("#### 已配置模型")
            for item in configured_models:
                usage = item.get("usage") or {}
                with st.container(border=True):
                    title, remove = st.columns([5, 1])
                    title.markdown(
                        f"**{escape(item['name'])}** · `{escape(item['model'])}`"
                        + (" · 当前默认" if item.get("active") else "")
                    )
                    if remove.button("删除", key=f"delete_model_{item['name']}"):
                        deleted = api_call("DELETE", f"/api/ai/model-configs/{item['name']}")
                        if deleted.get("code") == 200:
                            set_flash("success", f"模型配置 {item['name']} 已删除。")
                            st.rerun()
                    st.caption(item["provider_url"])
                    input_col, output_col, cost_col = st.columns(3)
                    input_col.metric("累计输入 Token", usage.get("input_tokens", 0))
                    output_col.metric("累计输出 Token", usage.get("output_tokens", 0))
                    amount = float(usage.get("cost", 0) or 0)
                    cost_col.metric(
                        "累计花费",
                        f"¥{amount:.2f}"
                        if item.get("currency") == "CNY"
                        else f"{amount:.2f} {item.get('currency', '')}",
                    )

        st.markdown("#### 添加或更新模型")
        with st.container(border=True):
            config_name = st.text_input(
                "配置名称 *", placeholder="例如：百炼主模型", key="ai_config_name"
            )
            provider_url = st.text_input(
                "提供商 Base URL *",
                placeholder="https://你的业务空间地址/compatible-mode/v1",
                key="ai_config_provider_url",
            )
            model = st.text_input(
                "模型名称 *", placeholder="例如 qwen3.7-plus", key="ai_config_model"
            )
            api_key = st.text_input("API Key *", type="password", key="ai_config_api_key")
            left, right = st.columns(2)
            input_price = left.number_input(
                "输入单价", 0.0, value=0.0, format="%.6f", key="ai_config_input_price"
            )
            output_price = right.number_input(
                "输出单价", 0.0, value=0.0, format="%.6f", key="ai_config_output_price"
            )
            price_unit = st.number_input(
                "计价单位（Token）", 1, value=1_000_000, key="ai_config_price_unit"
            )
            currency = st.text_input("币种", value="CNY", key="ai_config_currency")
            submitted = st.button("保存配置", type="primary")
        if submitted:
            result = api_call(
                "PUT",
                "/api/ai/model-config",
                json={
                    "name": config_name,
                    "provider_url": provider_url,
                    "model": model,
                    "api_key": api_key,
                    "input_price": input_price,
                    "output_price": output_price,
                    "price_unit": price_unit,
                    "currency": currency,
                },
            )
            if result.get("code") == 200:
                set_flash("success", f"模型配置 {config_name} 已加密保存。")
                st.rerun()


def render_user_management(login: dict[str, Any]) -> None:
    st.markdown("#### 用户与角色")
    with st.spinner("正在读取用户列表……"):
        result = api_call("GET", "/api/users/?page=1&page_size=100", quiet=True)
    if result.get("code") != 200:
        if result.get("code") not in {401, 503}:
            st.error(result.get("msg", "用户列表加载失败"))
        return
    users = result["data"]["users"]
    if not users:
        st.info("暂无用户。")
        return
    table = [
        {
            "用户 ID": user["user_id"],
            "用户名": user["username"],
            "角色": role_label(user["role"]),
            "提交": user["submit_count"],
            "通过题目": user["resolve_count"],
            "加入日期": user["join_time"],
        }
        for user in users
    ]
    st.dataframe(table, width="stretch", hide_index=True)
    choices = {f"{user['username']} (#{user['user_id']})": user for user in users}
    with st.form("admin_role_form"):
        selected_label = st.selectbox("选择用户 *", list(choices))
        selected = choices[selected_label]
        roles = ["user", "admin", "banned"]
        role = st.selectbox(
            "目标角色 *",
            roles,
            index=roles.index(selected["role"]),
            format_func=role_label,
        )
        submitted = st.form_submit_button("更新角色", type="primary")
    if submitted:
        with st.spinner("正在更新角色……"):
            changed = api_call("PUT", f"/api/users/{selected['user_id']}/role", json={"role": role})
        if changed.get("code") == 200:
            if selected["user_id"] == login["user_id"]:
                st.session_state.login["role"] = role
            set_flash("success", f"{selected['username']} 的角色已更新。")
            st.rerun()


def render_language_management() -> None:
    st.markdown("#### 判题语言")
    current = api_call("GET", "/api/languages/", quiet=True)
    if current.get("code") == 200:
        names = current["data"]["name"]
        st.info("当前语言：" + ("、".join(names) if names else "暂无"))
    with st.expander("可添加语言与命令示例"):
        st.dataframe(LANGUAGE_COMPATIBILITY, hide_index=True, width="stretch")
        st.caption("添加前请确认对应程序已安装；命令仅支持 {src}、{exe} 占位符。")
    with st.form("language_form"):
        name = st.text_input("语言名称 *", placeholder="例如 go")
        extension = st.text_input("文件扩展名 *", placeholder="例如 .go")
        compile_command = st.text_input(
            "编译命令", placeholder="解释型语言可留空，例如 go build -o {exe} {src}"
        )
        run_command = st.text_input("运行命令 *", placeholder="例如 {exe}")
        col1, col2 = st.columns(2)
        time_limit = col1.number_input("默认时间限制（秒）", 0.01, 60.0, 3.0)
        memory_limit = col2.number_input("默认内存限制（MB）", 1, 4096, 128)
        registered = st.form_submit_button("注册语言", type="primary")
    if registered:
        if not name.strip() or not extension.strip() or not run_command.strip():
            st.error("语言名称、文件扩展名和运行命令为必填项。")
            return
        payload = {
            "name": name.strip(),
            "file_ext": extension.strip(),
            "compile_cmd": compile_command.strip() or None,
            "run_cmd": run_command.strip(),
            "time_limit": time_limit,
            "memory_limit": memory_limit,
        }
        with st.spinner("正在注册判题语言……"):
            result = api_call("POST", "/api/languages/", json=payload)
        if result.get("code") == 200:
            set_flash("success", f"语言 {payload['name']} 注册成功。")
            st.rerun()


def render_audit_logs() -> None:
    st.markdown("#### 测试点日志访问审计")
    users_result = api_call("GET", "/api/users/?page=1&page_size=100", quiet=True)
    problems_result = api_call("GET", "/api/problems/", quiet=True)
    users = (users_result.get("data") or {}).get("users", [])
    problems = problems_result.get("data") or []
    user_names = {str(user["user_id"]): user["username"] for user in users}
    problem_names = {problem["id"]: problem["title"] for problem in problems}
    user_options = {"全部用户": ""} | {
        f"{user['username']} (#{user['user_id']})": str(user["user_id"]) for user in users
    }
    problem_options = {"全部题目": ""} | {
        f"{problem['id']} · {problem['title']}": problem["id"] for problem in problems
    }
    filter1, filter2, filter3, refresh = st.columns([2, 2, 1.5, 1])
    selected_user = filter1.selectbox("访问用户", list(user_options), key="audit_user_filter")
    selected_problem = filter2.selectbox(
        "访问题目", list(problem_options), key="audit_problem_filter"
    )
    selected_status = filter3.selectbox(
        "访问结果", ["全部", "允许", "拒绝"], key="audit_status_filter"
    )
    refresh.write("")
    refresh.write("")
    refresh.button("刷新", width="stretch")
    params: dict[str, Any] = {"page_size": 200}
    if user_options[selected_user]:
        params["user_id"] = user_options[selected_user]
    if problem_options[selected_problem]:
        params["problem_id"] = problem_options[selected_problem]
    with st.spinner("正在读取审计日志……"):
        result = api_call("GET", "/api/logs/access/", params=params, quiet=True)
    if result.get("code") != 200:
        st.error(result.get("msg", "审计日志加载失败"))
        return
    logs = result["data"]
    if selected_status != "全部":
        target_status = "200" if selected_status == "允许" else "403"
        logs = [entry for entry in logs if str(entry.get("status")) == target_status]
    if not logs:
        st.info("当前筛选条件下没有审计记录。")
        return

    allowed = sum(str(entry.get("status")) == "200" for entry in logs)
    denied = len(logs) - allowed
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("访问记录", len(logs))
    metric2.metric("允许", allowed)
    metric3.metric("拒绝", denied)
    metric4.metric("涉及用户", len({str(entry.get("user_id")) for entry in logs}))
    rows = [
        {
            "时间": str(entry.get("time", "—")).replace("T", " ")[:19],
            "用户": f"{user_names.get(str(entry.get('user_id')), '未知用户')} "
            f"(#{entry.get('user_id', '—')})",
            "题目": f"{problem_names.get(str(entry.get('problem_id')), '已删除/未知题目')} "
            f"({entry.get('problem_id', '—')})",
            "操作": "查看测试点日志",
            "结果": "允许" if str(entry.get("status")) == "200" else "拒绝",
        }
        for entry in logs
    ]
    st.dataframe(rows, width="stretch", hide_index=True)
    with st.expander("最近访问详情", expanded=True):
        for entry in logs[:10]:
            with st.container(border=True):
                identity, target, outcome = st.columns([2, 3, 1])
                user_id = str(entry.get("user_id", "—"))
                problem_id = str(entry.get("problem_id", "—"))
                identity.markdown(f"**{escape(user_names.get(user_id, '未知用户'))}**")
                identity.caption(
                    f"用户 #{user_id} · {str(entry.get('time', '')).replace('T', ' ')[:19]}"
                )
                target.markdown(f"**{escape(problem_names.get(problem_id, '已删除/未知题目'))}**")
                target.caption(f"题目 {escape(problem_id)} · 查看测试点日志")
                outcome.markdown(
                    status_badge(str(entry.get("status", "403"))), unsafe_allow_html=True
                )


def render_system_reset() -> None:
    st.markdown("#### 恢复课程初始环境")
    st.error("危险操作：将取消后台任务，并清空用户、题目、提交和审计数据。")
    confirmation = st.text_input("请输入 RESET 以确认", key="reset_confirmation")
    if st.button(
        "重置系统",
        disabled=confirmation != "RESET",
        type="secondary",
        width="stretch",
    ):
        with st.spinner("正在重置系统……"):
            result = api_call("POST", "/api/reset/")
        if result.get("code") == 200:
            clear_local_session(close_client=True)
            set_flash("success", "系统已重置，请使用初始管理员重新登录。")
            st.rerun()


def page_admin() -> None:
    login = require_login()
    if not login:
        return
    st.title("后台管理")
    if login["role"] != "admin":
        st.error("该页面仅向管理员显示；所有接口仍会执行后端权限校验。")
        return
    user_tab, language_tab, audit_tab, reset_tab = st.tabs(
        ["用户管理", "语言管理", "审计日志", "系统重置"]
    )
    with user_tab:
        render_user_management(login)
    with language_tab:
        render_language_management()
    with audit_tab:
        render_audit_logs()
    with reset_tab:
        render_system_reset()


def main() -> None:
    inject_theme()
    initialize_browser_session()
    login = st.session_state.get("login")
    pages: dict[str, Any]
    if login:
        pages = {
            "账户概览": page_account,
            "题库与评测": page_problems,
            "提交记录": page_submission_records,
            "AI 智能命题": page_ai,
        }
        if login["role"] == "admin":
            pages["后台管理"] = page_admin
    else:
        pages = {"登录 / 注册": page_account}

    next_navigation = st.session_state.pop("next_navigation", None)
    if next_navigation in pages:
        st.session_state.main_navigation = next_navigation
    current_navigation = st.session_state.get("main_navigation")
    if current_navigation not in pages:
        st.session_state.main_navigation = next(iter(pages))

    with st.container(key="top_navigation"):
        brand_col, user_col, logout_col = st.columns([5, 2, 1], vertical_alignment="center")
        brand_col.markdown('<div class="oj-brand">⚖️ 在线评测系统</div>', unsafe_allow_html=True)
        if login:
            user_col.markdown(
                (
                    '<div class="oj-top-user">'
                    f"<strong>{escape(login['username'])}</strong>"
                    f"{role_label(login['role'])} · ID {login['user_id']}"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if logout_col.button("退出", width="stretch", type="secondary"):
                logout_user()

        if len(pages) > 1:
            navigation = st.segmented_control(
                "功能导航",
                list(pages),
                key="main_navigation",
                selection_mode="single",
                required=True,
                width="stretch",
                label_visibility="collapsed",
                format_func=lambda page: {
                    "账户概览": "👤 账户概览",
                    "题库与评测": "💻 题库与评测",
                    "提交记录": "📋 提交记录",
                    "AI 智能命题": "✨ AI 智能命题",
                    "后台管理": "⚙️ 后台管理",
                }.get(page, page),
            )
        else:
            navigation = next(iter(pages))

    show_flash()
    pages[navigation]()


if __name__ == "__main__":
    main()
