from __future__ import annotations

import json
import os
from typing import Any

import httpx
import streamlit as st

st.set_page_config(page_title="Async OJ", page_icon="⚖️", layout="wide")


class APIClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = httpx.Client(timeout=20.0, follow_redirects=True)

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            response = self.session.request(method, f"{self.base_url}{path}", **kwargs)
            body = response.json()
            if not isinstance(body, dict):
                raise ValueError
            return body
        except httpx.HTTPError as exc:
            return {"code": 503, "msg": f"无法连接后端：{exc}", "data": None}
        except (ValueError, json.JSONDecodeError):
            return {"code": 502, "msg": "后端返回了无法解析的响应", "data": None}


def get_client() -> APIClient:
    base_url = st.session_state.get(
        "api_base_url", os.getenv("OJ_API_BASE_URL", "http://127.0.0.1:8000")
    )
    client = st.session_state.get("api_client")
    if not isinstance(client, APIClient) or client.base_url != base_url.rstrip("/"):
        if isinstance(client, APIClient):
            client.session.close()
        client = APIClient(base_url)
        st.session_state.api_client = client
    return client


def api_call(method: str, path: str, *, quiet: bool = False, **kwargs: Any) -> dict[str, Any]:
    result = get_client().request(method, path, **kwargs)
    if result.get("code") != 200 and not quiet:
        st.error(f"{result.get('code', '错误')}：{result.get('msg', '请求失败')}")
        if result.get("code") == 401:
            st.session_state.pop("login", None)
    return result


def require_login() -> dict[str, Any] | None:
    login = st.session_state.get("login")
    if not login:
        st.warning("请先在“账户”页面登录。")
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
    st.header("账户与用户管理")
    login = st.session_state.get("login")
    if not login:
        login_tab, register_tab = st.tabs(["登录", "注册"])
        with login_tab:
            with st.form("login_form"):
                username = st.text_input("用户名")
                password = st.text_input("密码", type="password")
                submitted = st.form_submit_button("登录", type="primary")
            if submitted:
                result = api_call(
                    "POST", "/api/auth/login", json={"username": username, "password": password}
                )
                if result.get("code") == 200:
                    st.session_state.login = result["data"]
                    st.success("登录成功")
                    st.rerun()
        with register_tab:
            with st.form("register_form"):
                username = st.text_input("新用户名（3–40 字符）")
                password = st.text_input("新密码（至少 6 位）", type="password")
                submitted = st.form_submit_button("注册")
            if submitted:
                result = api_call(
                    "POST", "/api/users/", json={"username": username, "password": password}
                )
                if result.get("code") == 200:
                    st.success("注册成功，请切换到登录页。")
        return

    st.success(f"当前用户：{login['username']}（{login['role']}）")
    col1, col2 = st.columns([4, 1])
    with col1:
        profile = api_call("GET", f"/api/users/{login['user_id']}", quiet=True)
        if profile.get("code") == 200:
            data = profile["data"]
            st.metric("提交次数", data["submit_count"])
            st.metric("通过题数", data["resolve_count"])
            st.json(data)
    with col2:
        if st.button("退出登录", use_container_width=True):
            result = api_call("POST", "/api/auth/logout")
            if result.get("code") == 200:
                st.session_state.pop("login", None)
                st.rerun()

    if login["role"] == "admin":
        st.subheader("用户管理")
        result = api_call("GET", "/api/users/?page=1&page_size=100")
        if result.get("code") == 200:
            users = result["data"]["users"]
            st.dataframe(users, use_container_width=True, hide_index=True)
            choices = {f"{user['username']} (#{user['user_id']})": user for user in users}
            selected_label = st.selectbox("选择用户", list(choices))
            selected = choices[selected_label]
            role = st.selectbox(
                "设置角色",
                ["user", "admin", "banned"],
                index=["user", "admin", "banned"].index(selected["role"]),
            )
            if st.button("更新角色"):
                changed = api_call(
                    "PUT", f"/api/users/{selected['user_id']}/role", json={"role": role}
                )
                if changed.get("code") == 200:
                    st.success("角色已更新")
                    st.rerun()


def problem_form(key: str, initial: dict[str, Any] | None = None) -> dict[str, Any] | None:
    initial = initial or {}
    with st.form(key):
        left, right = st.columns(2)
        with left:
            problem_id = st.text_input("题目 ID", value=str(initial.get("id", "")))
            title = st.text_input("标题", value=str(initial.get("title", "")))
            difficulty = st.text_input("难度", value=str(initial.get("difficulty", "")))
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
            "题目描述", value=str(initial.get("description", "")), height=140
        )
        input_description = st.text_area(
            "输入格式", value=str(initial.get("input_description", "")), height=90
        )
        output_description = st.text_area(
            "输出格式", value=str(initial.get("output_description", "")), height=90
        )
        constraints = st.text_area("数据范围", value=str(initial.get("constraints", "")), height=90)
        samples_text = st.text_area(
            "样例（JSON 数组）", value=pretty_json(initial.get("samples", [])), height=160
        )
        testcases_text = st.text_area(
            "测试点（JSON 数组）",
            value=pretty_json(initial.get("testcases", [])),
            height=240,
            help='每项格式为 {"input": "...", "output": "..."}',
        )
        submitted = st.form_submit_button("校验并提交", type="primary")
    if not submitted:
        return None
    tags = parse_json_list(tags_text, "标签")
    samples = parse_json_list(samples_text, "样例")
    testcases = parse_json_list(testcases_text, "测试点")
    if tags is None or samples is None or testcases is None:
        return None
    return {
        "id": problem_id,
        "title": title,
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


def page_problems() -> None:
    login = require_login()
    if not login:
        return
    st.header("题目管理")
    browse_tab, create_tab, edit_tab = st.tabs(["题库与详情", "新增题目", "编辑题目"])
    result = api_call("GET", "/api/problems/", quiet=True)
    problems = result.get("data") or [] if result.get("code") == 200 else []

    with browse_tab:
        if not problems:
            st.info("题库为空，请先新增题目。")
        else:
            label_map = {f"{item['id']} · {item['title']}": item["id"] for item in problems}
            label = st.selectbox("选择题目", list(label_map), key="problem_browser")
            problem_id = label_map[label]
            detail = api_call("GET", f"/api/problems/{problem_id}")
            if detail.get("code") == 200:
                data = detail["data"]
                st.subheader(data["title"])
                st.markdown(data["description"])
                col1, col2, col3 = st.columns(3)
                col1.metric("时间限制", f"{data['time_limit']} s")
                col2.metric("内存限制", f"{data['memory_limit']} MB")
                col3.metric("测试点", len(data["testcases"]))
                st.markdown(f"**输入格式**\n\n{data['input_description']}")
                st.markdown(f"**输出格式**\n\n{data['output_description']}")
                st.markdown(f"**数据范围**\n\n{data['constraints']}")
                st.json(data["samples"])
                if login["role"] == "admin":
                    visibility = st.toggle(
                        "向所有登录用户公开测试点日志", value=data.get("public_cases", False)
                    )
                    if st.button("保存日志可见性"):
                        updated = api_call(
                            "PUT",
                            f"/api/problems/{problem_id}/log_visibility",
                            json={"public_cases": visibility},
                        )
                        if updated.get("code") == 200:
                            st.success("可见性已更新")
                    if st.button("删除题目", type="secondary"):
                        deleted = api_call("DELETE", f"/api/problems/{problem_id}")
                        if deleted.get("code") == 200:
                            st.success("题目已删除")
                            st.rerun()

    with create_tab:
        draft = st.session_state.get("ai_problem_draft")
        if draft:
            st.info("已载入 AI 命题草稿，请审阅所有字段后提交。")
        payload = problem_form("create_problem", draft)
        if payload:
            created = api_call("POST", "/api/problems/", json=payload)
            if created.get("code") == 200:
                st.session_state.pop("ai_problem_draft", None)
                st.success("题目新增成功")

    with edit_tab:
        if not problems:
            st.info("暂无可编辑题目。")
        else:
            ids = [item["id"] for item in problems]
            selected_id = st.selectbox("选择待编辑题目", ids, key="problem_editor")
            detail = api_call("GET", f"/api/problems/{selected_id}", quiet=True)
            if detail.get("code") == 200:
                payload = problem_form(f"edit_problem_{selected_id}", detail["data"])
                if payload:
                    updated = api_call("PUT", f"/api/problems/{selected_id}", json=payload)
                    if updated.get("code") == 200:
                        st.success("题目更新成功")


@st.fragment(run_every=1.0)
def live_submission_panel() -> None:
    submission_id = st.session_state.get("last_submission_id")
    if not submission_id:
        return
    result = api_call("GET", f"/api/submissions/{submission_id}", quiet=True)
    if result.get("code") != 200:
        st.error(result.get("msg", "查询失败"))
        return
    data = result["data"]
    if data["status"] == "pending":
        st.info(f"评测 #{submission_id} 正在运行……")
        st.progress(40)
    elif data["status"] == "success":
        st.success(f"评测完成：{data['score']} / {data['counts']}")
        st.json(data)
    else:
        st.error(data.get("error_info") or "评测任务异常")
        st.json(data)


def page_submissions() -> None:
    login = require_login()
    if not login:
        return
    st.header("代码提交与评测")
    submit_tab, records_tab, detail_tab = st.tabs(["提交代码", "提交记录", "详情与日志"])
    problem_result = api_call("GET", "/api/problems/", quiet=True)
    language_result = api_call("GET", "/api/languages/", quiet=True)
    problems = problem_result.get("data") or []
    languages = (language_result.get("data") or {}).get("name", [])

    with submit_tab:
        if not problems:
            st.info("请先创建至少一道题目。")
        else:
            problem_map = {f"{item['id']} · {item['title']}": item["id"] for item in problems}
            with st.form("submit_code"):
                problem_label = st.selectbox("题目", list(problem_map))
                language = st.selectbox("语言", languages)
                code = st.text_area("源代码", height=360, placeholder="在此粘贴完整程序")
                submitted = st.form_submit_button("提交评测", type="primary")
            if submitted:
                result = api_call(
                    "POST",
                    "/api/submissions/",
                    json={
                        "problem_id": problem_map[problem_label],
                        "language": language,
                        "code": code,
                    },
                )
                if result.get("code") == 200:
                    st.session_state.last_submission_id = result["data"]["submission_id"]
                    st.success(f"提交成功：#{result['data']['submission_id']}")
        live_submission_panel()

    with records_tab:
        filter_problem = st.selectbox(
            "按题目筛选（可选）", [""] + [item["id"] for item in problems]
        )
        filter_status = st.selectbox("状态", ["", "pending", "success", "error"])
        params = {"user_id": login["user_id"], "page_size": 100}
        if filter_problem:
            params["problem_id"] = filter_problem
        if filter_status:
            params["status"] = filter_status
        if st.button("刷新记录"):
            records = api_call("GET", "/api/submissions/", params=params)
            if records.get("code") == 200:
                st.dataframe(
                    records["data"]["submissions"], use_container_width=True, hide_index=True
                )

    with detail_tab:
        submission_id = st.text_input(
            "Submission ID", value=str(st.session_state.get("last_submission_id", ""))
        )
        col1, col2 = st.columns(2)
        if col1.button("查询详情") and submission_id:
            detail = api_call("GET", f"/api/submissions/{submission_id}")
            if detail.get("code") == 200:
                st.json(detail["data"])
        if col2.button("查询测试点日志") and submission_id:
            log = api_call("GET", f"/api/submissions/{submission_id}/log")
            if log.get("code") == 200:
                st.json(log["data"])
        if login["role"] == "admin" and st.button("重新评测") and submission_id:
            result = api_call("PUT", f"/api/submissions/{submission_id}/rejudge")
            if result.get("code") == 200:
                st.session_state.last_submission_id = submission_id
                st.success("已开始重新评测")


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


def page_system() -> None:
    login = require_login()
    if not login:
        return
    st.header("语言与系统管理")
    language_tab, audit_tab, reset_tab = st.tabs(["语言配置", "访问审计", "系统重置"])

    with language_tab:
        current = api_call("GET", "/api/languages/", quiet=True)
        if current.get("code") == 200:
            st.write("已注册语言：", "、".join(current["data"]["name"]))
        st.caption("命令以参数数组运行；后端会校验可执行程序、占位符和危险字符。")
        with st.form("language_form"):
            name = st.text_input("语言名称", placeholder="例如 go")
            extension = st.text_input("文件扩展名", placeholder="例如 .go")
            compile_command = st.text_input(
                "编译命令（解释型语言可留空）", placeholder="go build -o {exe} {src}"
            )
            run_command = st.text_input("运行命令", placeholder="{exe}")
            col1, col2 = st.columns(2)
            time_limit = col1.number_input("默认时间限制（秒）", 0.01, 60.0, 3.0)
            memory_limit = col2.number_input("默认内存限制（MB）", 1, 4096, 128)
            registered = st.form_submit_button("注册语言")
        if registered:
            payload = {
                "name": name,
                "file_ext": extension,
                "compile_cmd": compile_command or None,
                "run_cmd": run_command,
                "time_limit": time_limit,
                "memory_limit": memory_limit,
            }
            result = api_call("POST", "/api/languages/", json=payload)
            if result.get("code") == 200:
                st.success("语言注册成功")
                st.rerun()

    with audit_tab:
        if login["role"] != "admin":
            st.info("仅管理员可查看访问审计。")
        else:
            user_id = st.text_input("按用户 ID 筛选（可选）")
            problem_id = st.text_input("按题目 ID 筛选（可选）")
            if st.button("查询审计日志"):
                params: dict[str, Any] = {"page_size": 200}
                if user_id:
                    params["user_id"] = user_id
                if problem_id:
                    params["problem_id"] = problem_id
                result = api_call("GET", "/api/logs/access/", params=params)
                if result.get("code") == 200:
                    st.dataframe(result["data"], use_container_width=True, hide_index=True)

    with reset_tab:
        if login["role"] != "admin":
            st.info("仅管理员可重置测试数据。")
        else:
            st.warning("此操作会取消后台任务并清空用户、题目、提交和审计数据。")
            confirmed = st.checkbox("我确认要恢复课程初始环境")
            if st.button("重置系统", disabled=not confirmed, type="secondary"):
                result = api_call("POST", "/api/reset/")
                if result.get("code") == 200:
                    client = st.session_state.get("api_client")
                    if isinstance(client, APIClient):
                        client.session.close()
                    st.session_state.clear()
                    st.success("系统已重置，请重新登录。")
                    st.rerun()


def main() -> None:
    st.title("⚖️ Async Online Judge")
    st.caption("FastAPI 异步后端 · Python/C++ 判题 · 细粒度权限 · AI 智能命题")
    with st.sidebar:
        st.text_input(
            "后端地址",
            key="api_base_url",
            value=os.getenv("OJ_API_BASE_URL", "http://127.0.0.1:8000"),
        )
        navigation = st.radio("导航", ["账户", "题目", "评测", "AI 命题", "系统管理"])
        if st.session_state.get("login"):
            st.divider()
            login = st.session_state.login
            st.write(f"{login['username']} · {login['role']}")

    pages = {
        "账户": page_account,
        "题目": page_problems,
        "评测": page_submissions,
        "AI 命题": page_ai,
        "系统管理": page_system,
    }
    pages[navigation]()


if __name__ == "__main__":
    main()
