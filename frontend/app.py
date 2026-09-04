from __future__ import annotations

import json
import os
from html import escape
from typing import Any

import httpx
import streamlit as st

try:
    from frontend.api_client import (
        clear_client_cookies,
        close_persistent_client,
        get_persistent_client,
        normalize_base_url,
        request_json,
    )
except ModuleNotFoundError:  # Streamlit executes this file with frontend/ on sys.path.
    from api_client import (  # type: ignore[no-redef]
        clear_client_cookies,
        close_persistent_client,
        get_persistent_client,
        normalize_base_url,
        request_json,
    )

st.set_page_config(page_title="在线评测系统", page_icon="⚖️", layout="wide")

DEFAULT_API_BASE_URL = normalize_base_url(os.getenv("OJ_API_BASE_URL", "http://127.0.0.1:8000"))
ROLE_LABELS = {"admin": "管理员", "user": "普通用户", "banned": "已封禁"}
STATUS_STYLES = {
    "AC": ("#067647", "#ecfdf3"),
    "WA": ("#b42318", "#fef3f2"),
    "TLE": ("#b54708", "#fffaeb"),
    "MLE": ("#6941c6", "#f4f3ff"),
    "RE": ("#c11574", "#fdf2fa"),
    "CE": ("#344054", "#f2f4f7"),
    "UNK": ("#475467", "#f2f4f7"),
    "pending": ("#175cd3", "#eff8ff"),
    "error": ("#b42318", "#fef3f2"),
    "未通过": ("#b42318", "#fef3f2"),
}


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1180px;}
        [data-testid="stSidebar"] {border-right: 1px solid #e4e7ec;}
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


def clear_local_session(*, close_client: bool = False) -> None:
    for key in (
        "login",
        "last_submission_id",
        "submission_detail",
        "submission_log",
        "ai_task_id",
        "ai_problem_draft",
        "main_navigation",
        "next_navigation",
    ):
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
    return (
        f'<span class="oj-badge" style="color:{foreground};background:{background};">'
        f"{escape(status)}</span>"
    )


def submission_verdict(data: dict[str, Any]) -> str:
    if data.get("status") == "pending":
        return "pending"
    if data.get("status") == "error":
        return "error"
    if data.get("counts") and data.get("score") == data.get("counts"):
        return "AC"
    return "未通过"


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
    st.caption("使用后端签名 Session Cookie 保持登录状态，所有权限由后端接口校验。")
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
    st.caption(f"当前角色：{role_label(login['role'])} · 用户 ID：{login['user_id']}")
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
        state = "✅ 已通过" if accepted else ("🟠 已尝试" if attempts else "⚪ 未尝试")
        overviews.append(
            {
                "detail": detail_result["data"],
                "attempts": attempts,
                "accepted": accepted,
                "state": state,
            }
        )
    return overviews


def page_problems() -> None:
    login = require_login()
    if not login:
        return
    st.title("题库与题目")
    st.caption("浏览题目、查看个人通过状态，并在同一页面完成题目维护。")
    browse_tab, create_tab, edit_tab = st.tabs(["浏览题库", "新增题目", "编辑题目"])
    result = api_call("GET", "/api/problems/", quiet=True)
    problems = result.get("data") or [] if result.get("code") == 200 else []

    with browse_tab:
        if not problems:
            st.info("题库目前为空。可切换到“新增题目”创建第一道题。")
        else:
            with st.spinner("正在整理题目与个人通过状态……"):
                overviews = load_problem_overviews(problems, login)
            rows = [
                {
                    "状态": overview["state"],
                    "题号": overview["detail"]["id"],
                    "题目": overview["detail"]["title"],
                    "难度": overview["detail"].get("difficulty") or "未标注",
                    "标签": "、".join(overview["detail"].get("tags") or []) or "—",
                    "提交次数": overview["attempts"],
                }
                for overview in overviews
            ]
            st.dataframe(
                rows,
                width="stretch",
                hide_index=True,
                column_config={"提交次数": st.column_config.NumberColumn(format="%d 次")},
            )
            choices = {
                f"{overview['state']} · {overview['detail']['id']} · {overview['detail']['title']}": overview
                for overview in overviews
            }
            selected_label = st.selectbox("查看题目详情", list(choices), key="problem_browser")
            overview = choices[selected_label]
            data = overview["detail"]
            problem_id = data["id"]

            with st.container(border=True):
                title_col, action_col = st.columns([4, 1])
                title_col.subheader(f"{problem_id} · {data['title']}")
                title_col.caption(
                    f"难度：{data.get('difficulty') or '未标注'} · "
                    f"来源：{data.get('source') or '课程题库'} · {overview['state']}"
                )
                if action_col.button("前往提交", type="primary", width="stretch"):
                    st.session_state.submit_problem_id = problem_id
                    navigate_on_next_rerun("提交与评测")
                    st.rerun()

                limit1, limit2, limit3, limit4 = st.columns(4)
                limit1.metric("时间限制", f"{data['time_limit']} s")
                limit2.metric("内存限制", f"{data['memory_limit']} MB")
                limit3.metric("测试点", len(data["testcases"]))
                limit4.metric("个人提交", overview["attempts"])
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

                with st.expander("查看样例", expanded=True):
                    for index, sample in enumerate(data.get("samples") or [], start=1):
                        st.markdown(f"**样例 {index}**")
                        sample_in, sample_out = st.columns(2)
                        sample_in.caption("输入")
                        sample_in.code(sample.get("input", ""), language=None)
                        sample_out.caption("输出")
                        sample_out.code(sample.get("output", ""), language=None)

            if login["role"] == "admin":
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

    with create_tab:
        draft = st.session_state.get("ai_problem_draft")
        if draft:
            st.info("已载入 AI 命题草稿。请人工审阅所有字段后再保存。")
        payload = problem_form("create_problem", draft)
        if payload:
            with st.spinner("正在保存题目……"):
                created = api_call("POST", "/api/problems/", json=payload)
            if created.get("code") == 200:
                st.session_state.pop("ai_problem_draft", None)
                set_flash("success", f"题目 {payload['id']} 已创建。")
                st.rerun()

    with edit_tab:
        if not problems:
            st.info("暂无可编辑题目。")
        else:
            ids = [item["id"] for item in problems]
            selected_id = st.selectbox("选择待编辑题目", ids, key="problem_editor")
            detail = api_call("GET", f"/api/problems/{selected_id}", quiet=True)
            if detail.get("code") == 200:
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
    score_col, count_col, compile_col = st.columns(3)
    score_col.metric("得分", data.get("score", 0))
    count_col.metric("总分", data.get("counts", 0))
    compile_result = (data.get("compile_info") or {}).get("result", "无需编译")
    compile_col.metric("编译", compile_result)
    if data.get("error_info"):
        st.error(data["error_info"])
    details = data.get("details") or []
    if not details:
        st.info("暂无测试点明细。")
        return
    st.markdown("#### 测试点结果")
    for case in details:
        with st.container(border=True):
            case_col, result_col, time_col, memory_col = st.columns([1.2, 1, 1, 1])
            case_col.markdown(f"**测试点 #{case.get('id', '—')}**")
            result_col.markdown(
                status_badge(str(case.get("result", "UNK"))), unsafe_allow_html=True
            )
            time_col.caption("运行时间")
            time_col.write(f"{case.get('time', 0):.4f} s")
            memory_col.caption("峰值内存")
            memory_col.write(f"{case.get('memory', 0):.3f} MB")


def page_submissions() -> None:
    login = require_login()
    if not login:
        return
    st.title("提交与评测")
    st.caption("选择题目和语言提交代码；每位用户每分钟最多提交 3 次。")
    submit_tab, records_tab, detail_tab = st.tabs(["提交代码", "我的记录", "详情与日志"])
    problem_result = api_call("GET", "/api/problems/", quiet=True)
    language_result = api_call("GET", "/api/languages/", quiet=True)
    problems = problem_result.get("data") or []
    languages = (language_result.get("data") or {}).get("name", [])

    with submit_tab:
        if not problems:
            st.info("题库为空，暂时无法提交。")
        elif not languages:
            st.warning("尚未配置可用语言，请联系管理员。")
        else:
            problem_map = {f"{item['id']} · {item['title']}": item["id"] for item in problems}
            labels = list(problem_map)
            preferred = st.session_state.pop("submit_problem_id", None)
            selected_index = next(
                (index for index, label in enumerate(labels) if problem_map[label] == preferred), 0
            )
            problem_label = st.selectbox("题目 *", labels, index=selected_index)
            selected_problem_id = problem_map[problem_label]
            selected_detail = api_call("GET", f"/api/problems/{selected_problem_id}", quiet=True)
            if selected_detail.get("code") == 200:
                detail = selected_detail["data"]
                limit1, limit2, limit3 = st.columns(3)
                limit1.metric("时间限制", f"{detail['time_limit']} s")
                limit2.metric("内存限制", f"{detail['memory_limit']} MB")
                limit3.metric("测试点", len(detail["testcases"]))
            with st.form("submit_code"):
                language = st.selectbox("语言 *", languages)
                code = st.text_area(
                    "源代码 *",
                    height=340,
                    placeholder="请粘贴可直接运行的完整程序，不要包含 Markdown 代码块。",
                )
                submitted = st.form_submit_button("提交评测", type="primary", width="stretch")
            if submitted:
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

    with records_tab:
        filter_col, status_col, refresh_col = st.columns([2, 2, 1])
        filter_problem = filter_col.selectbox(
            "题目筛选",
            [""] + [item["id"] for item in problems],
            format_func=lambda x: x or "全部题目",
        )
        filter_status = status_col.selectbox(
            "任务状态",
            ["", "pending", "success", "error"],
            format_func=lambda x: {
                "": "全部状态",
                "pending": "评测中",
                "success": "已完成",
                "error": "异常",
            }[x],
        )
        refresh_col.write("")
        refresh_col.write("")
        refresh_col.button("刷新", width="stretch")
        params = {"user_id": login["user_id"], "page_size": 100}
        if filter_problem:
            params["problem_id"] = filter_problem
        if filter_status:
            params["status"] = filter_status
        with st.spinner("正在读取提交记录……"):
            records = api_call("GET", "/api/submissions/", params=params, quiet=True)
        if records.get("code") == 200:
            submissions = records["data"]["submissions"]
            if not submissions:
                st.info("当前筛选条件下没有提交记录。")
            else:
                rows = [
                    {
                        "提交号": record["submission_id"],
                        "结果": submission_verdict(record),
                        "得分": (
                            "—"
                            if record["status"] == "pending"
                            else f"{record.get('score', 0)} / {record.get('counts', 0)}"
                        ),
                        "任务状态": record["status"],
                    }
                    for record in submissions
                ]
                st.dataframe(rows, width="stretch", hide_index=True)

    with detail_tab:
        submission_id = st.text_input(
            "提交 ID *", value=str(st.session_state.get("last_submission_id", ""))
        )
        col1, col2, col3 = st.columns(3)
        valid_id = submission_id.isdigit()
        if col1.button("查询详情", width="stretch", disabled=not valid_id):
            with st.spinner("正在查询提交详情……"):
                detail_result = api_call("GET", f"/api/submissions/{submission_id}")
            if detail_result.get("code") == 200:
                st.session_state.submission_detail = detail_result["data"]
        if col2.button("查询测试点日志", width="stretch", disabled=not valid_id):
            with st.spinner("正在读取测试点日志……"):
                log_result = api_call("GET", f"/api/submissions/{submission_id}/log")
            if log_result.get("code") == 200:
                st.session_state.submission_log = log_result["data"]
        if login["role"] == "admin" and col3.button(
            "重新评测", width="stretch", disabled=not valid_id
        ):
            with st.spinner("正在重新创建评测任务……"):
                result = api_call("PUT", f"/api/submissions/{submission_id}/rejudge")
            if result.get("code") == 200:
                st.session_state.last_submission_id = submission_id
                st.success("已开始重新评测。")

        detail_data = st.session_state.get("submission_detail")
        if detail_data:
            verdict = submission_verdict(detail_data)
            st.markdown("#### 提交详情")
            st.markdown(status_badge(verdict), unsafe_allow_html=True)
            st.json(detail_data)
        log_data = st.session_state.get("submission_log")
        if log_data:
            render_submission_log(log_data)


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
    col1, col2, col3 = st.columns(3)
    col1.metric("输入 Token", usage.get("input_tokens", 0))
    col2.metric("输出 Token", usage.get("output_tokens", 0))
    col3.metric("费用", f"{usage.get('cost', 0):.6f} {usage.get('currency', '')}")
    if usage.get("estimated"):
        st.caption("模型未返回完整 usage，当前 Token 与费用无法精确统计。")
    if data["status"] in {"pending", "running"}:
        if st.button("中断任务", key=f"cancel_{task_id}"):
            cancelled = api_call("PUT", f"/api/ai/problem-tasks/{task_id}/cancel")
            if cancelled.get("code") == 200:
                st.warning("任务已中断")
    elif data["status"] == "completed":
        st.success("AI 命题已完成并通过自动校验")
        result = data["result"]
        st.json(result)
        if st.button("载入到题目新增表单", key=f"load_{task_id}"):
            st.session_state.ai_problem_draft = result["problem"]
            st.success("已载入。请切换到“题目 → 新增题目”审阅并保存。")
    elif data["status"] == "cancelled":
        st.warning("任务已中断，后台不会继续调用模型。")
    else:
        st.error(data.get("error") or "命题任务失败")


def page_ai() -> None:
    if not require_login():
        return
    st.header("AI 智能命题")
    st.caption("密钥仅保存在后端进程内存中，不写入数据库、日志或页面响应。")
    config_tab, task_tab = st.tabs(["模型配置", "智能命题"])
    with config_tab:
        current = api_call("GET", "/api/ai/model-config", quiet=True)
        if current.get("code") == 200:
            st.success("已配置模型（API Key 已隐藏）")
            st.json(current["data"])
        with st.form("ai_config"):
            provider_url = st.text_input(
                "提供商 Base URL",
                placeholder="https://你的业务空间地址/compatible-mode/v1",
            )
            model = st.text_input("模型名称", placeholder="例如 qwen3.7-plus")
            api_key = st.text_input("API Key", type="password")
            left, right = st.columns(2)
            input_price = left.number_input("输入单价", 0.0, value=0.0, format="%.6f")
            output_price = right.number_input("输出单价", 0.0, value=0.0, format="%.6f")
            price_unit = st.number_input("计价单位（Token）", 1, value=1_000_000)
            currency = st.text_input("币种", value="CNY")
            submitted = st.form_submit_button("保存配置", type="primary")
        if submitted:
            result = api_call(
                "PUT",
                "/api/ai/model-config",
                json={
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
                st.success("模型配置已应用")

    with task_tab:
        problem_result = api_call("GET", "/api/problems/", quiet=True)
        existing_ids = [item["id"] for item in problem_result.get("data") or []]
        with st.form("ai_task"):
            requirement = st.text_area(
                "命题需求",
                height=150,
                placeholder="例如：设计一道考查滑动窗口的中等难度题，避免套用经典题面……",
            )
            knowledge_text = st.text_input("知识点（逗号分隔）")
            difficulty = st.selectbox("预期难度", ["入门", "简单", "中等", "困难"])
            testcase_count = st.slider("测试点数量", 3, 30, 12)
            problem_id = st.selectbox("参考/修改已有题目（可选）", [""] + existing_ids)
            submitted = st.form_submit_button("开始命题", type="primary")
        if submitted:
            result = api_call(
                "POST",
                "/api/ai/problem-tasks/",
                json={
                    "requirement": requirement,
                    "problem_id": problem_id or None,
                    "knowledge_points": [
                        item.strip() for item in knowledge_text.split(",") if item.strip()
                    ],
                    "difficulty": difficulty,
                    "testcase_count": testcase_count,
                },
            )
            if result.get("code") == 200:
                st.session_state.ai_task_id = result["data"]["task_id"]
                st.success(f"任务已创建：{result['data']['task_id']}")
        live_ai_task_panel()


def render_user_management(login: dict[str, Any]) -> None:
    st.markdown("#### 用户与角色")
    st.caption("角色入口仅用于改善界面体验，最终权限仍由后端校验。")
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
    st.caption("后端使用参数数组执行命令，并校验可执行程序、占位符及危险字符。")
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
    filter1, filter2 = st.columns(2)
    user_id = filter1.text_input("用户 ID（可选）", key="audit_user_id")
    problem_id = filter2.text_input("题目 ID（可选）", key="audit_problem_id")
    if st.button("查询审计日志", type="primary"):
        if user_id and not user_id.isdigit():
            st.error("用户 ID 必须为正整数。")
            return
        params: dict[str, Any] = {"page_size": 200}
        if user_id:
            params["user_id"] = user_id
        if problem_id.strip():
            params["problem_id"] = problem_id.strip()
        with st.spinner("正在读取审计日志……"):
            result = api_call("GET", "/api/logs/access/", params=params)
        if result.get("code") == 200:
            if result["data"]:
                st.dataframe(result["data"], width="stretch", hide_index=True)
            else:
                st.info("当前筛选条件下没有审计记录。")


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
    st.caption("集中管理用户、判题语言、访问审计和课程环境重置。")
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
    login = st.session_state.get("login")
    pages: dict[str, Any]
    if login:
        pages = {
            "账户概览": page_account,
            "题库与题目": page_problems,
            "提交与评测": page_submissions,
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

    with st.sidebar:
        st.markdown('<div class="oj-brand">⚖️ 在线评测系统</div>', unsafe_allow_html=True)
        st.markdown('<div class="oj-muted">课程大作业 2 · Async OJ</div>', unsafe_allow_html=True)
        st.divider()
        if login:
            st.markdown(
                (
                    '<div class="oj-user-card">'
                    f'<div class="oj-user-name">{escape(login["username"])}</div>'
                    f'<div class="oj-muted">{role_label(login["role"])} · ID {login["user_id"]}</div>'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if st.button("退出登录", width="stretch"):
                logout_user()
            st.divider()
        navigation = st.radio("功能导航", list(pages), key="main_navigation")
        st.divider()
        st.caption(f"API：{DEFAULT_API_BASE_URL}")
        st.caption("所有页面仅通过 REST API 访问后端。")

    show_flash()
    pages[navigation]()


if __name__ == "__main__":
    main()
