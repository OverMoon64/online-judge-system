# Async Online Judge

一个为“程序设计训练（Python）”大作业实现的小型在线评测系统。项目严格使用 FastAPI
异步接口，支持 Python/C++ 判题、Session 权限、测试点日志与审计、Streamlit 前端，
以及可取消、可计费的 AI 智能命题工作流。

> 安全边界：这是课程本地验收项目。判题器具备时间、内存、输出和进程组限制，但没有
> 容器/虚拟机级隔离，**不得直接作为面向不可信公网用户的生产 OJ 部署**。

## 功能与评分点

| 模块 | 实现 |
|---|---|
| Step 1 题目管理 | 列表、详情、新增、编辑、删除、严格字段校验 |
| Step 2 评测控制 | Python、C++14、动态语言、异步评测、TLE/MLE/输出比对 |
| Step 3 评测管理 | 状态、筛选、分页、限流、详情、同 ID 重新评测 |
| Step 4 用户管理 | bcrypt、Session、注册/登录/退出、角色、封禁、统计 |
| Step 5 评测日志 | 测试点结果、公开策略、访问权限和审计 |
| Step 6 前端交互 | Streamlit 用户、题目、提交、结果和管理页面 |
| Advance | 模型配置、分阶段 AI 命题、实时进度、实际取消、Token/费用 |

## 开发环境

推荐使用 Windows 11 + WSL2 Ubuntu 22.04 + VS Code Remote WSL。课程最终在 Linux
环境评测，判题器也按 Linux 进程模型设计。

```bash
cd /mnt/d/vscode/BIGHomework2
python3 -m venv ~/.venvs/online-judge-system
~/.venvs/online-judge-system/bin/pip install -r requirements-dev.txt
cp .env.example .env
```

请至少修改 `.env` 中的 `OJ_SESSION_SECRET`。`.env`、数据库、日志、临时代码和虚拟环境
均已被 `.gitignore` 排除。

VS Code 已提供推荐扩展、Python 解释器、调试和任务配置。以 Remote WSL 打开目录后，可从
“Terminal → Run Task”分别启动后端和前端。

## 启动

终端一：

```bash
~/.venvs/online-judge-system/bin/uvicorn app.main:app --reload --port 8000
```

终端二：

```bash
~/.venvs/online-judge-system/bin/streamlit run frontend/app.py --server.port 8501
```

- 前端：<http://127.0.0.1:8501>
- API 文档：<http://127.0.0.1:8000/docs>
- 初始管理员：`admin / admintestpassword`

## 前端会话与界面

Streamlit 把同一个 `httpx.Client` 保存在 `st.session_state` 中，登录响应写入的签名
Session Cookie 会在 rerun 后继续用于资料、用户、题目、提交、日志、AI 和管理接口。
退出时前端先调用后端 logout，再清理本地 Cookie 与登录状态；若后端返回 401，页面会提示
登录已失效并回到账户入口，不会依赖前端角色状态绕过认证。

前后端通信统一使用 `OJ_API_BASE_URL`（默认 `http://127.0.0.1:8000`），不要在同一次
运行中混用 `localhost` 与 `127.0.0.1`。登录后，页面顶部显示当前用户名、角色、退出按钮
和按角色精简的功能导航；题目通过状态、判题结果颜色和管理员功能分组均用于改善现场验收
体验，最终权限仍由后端 REST API 判断。

账户中心支持校验当前密码后修改密码。普通用户与管理员都可新增、编辑题目；删除题目、修改
日志公开策略、用户/语言/审计和系统重置仍由管理员负责。

生成演示数据：

```bash
~/.venvs/online-judge-system/bin/python scripts/seed_demo.py
```

## 判题行为

- 提交接口立即返回 `pending`，后台异步编译并逐测试点运行。
- Submission 状态只有 `pending/success/error`；测试点结果为
  `AC/WA/TLE/MLE/RE/CE/UNK`。
- 每个 AC 测试点 10 分。评测本身正常完成时，即使答案 WA/RE/CE，Submission 状态仍为
  `success`；只有判题引擎内部失败才是 `error`。
- 输出比较忽略每行末尾空格和最终多余空行，但不忽略前导空格。
- 每个用户 60 秒内最多创建 3 次提交；管理员重新评测不增加提交次数并复用原 ID。
- 动态语言命令使用 `shlex` 解析和参数数组执行，禁止 shell 元字符及未授权可执行程序，
  从不使用 `shell=True`。

## AI 智能命题

模型配置完全来自页面：配置名、提供商 Base URL、模型名、API Key、输入/输出单价、计价单位
和币种。每位用户可保存并选择多个模型；模型配置使用由 `OJ_SESSION_SECRET` 派生的密钥加密
保存在本机文件中，重启后无需重新输入。API、数据库和日志都不返回或记录 Key；系统重置会
删除加密配置文件。

验收演示建议使用阿里云百炼 OpenAI-compatible 接口和 `qwen3.7-plus`。请从百炼控制台复制
业务空间 Base URL 和 API Key，并按服务商账单规则手动填写输入/输出单价。OpenAI-compatible
协议没有统一价格接口，项目不自动抓取价格，避免地区、阶梯或调价差异造成误导。

进入 AI 模块后默认打开“智能命题”，可从已配置模型中选择一个执行任务。隐藏测试点默认由
模型结合算法复杂度、边界规模与预计运行时长自动生成 2–10 个，也可以手动指定数量。命题任务依次执行
需求分析、题目/参考解法生成、Pydantic 结构校验、测试点去重、受限运行参考解法和最多两次自动
修正；每道题同时生成 Python 3 与 C++14 参考程序，并分别试运行全部测试点。模型请求优先
使用 SSE 流式输出，页面实时展示当前阶段内容；不支持流式参数的兼容服务自动回退普通响应。
模型服务临时失败时自动重试两次。“中断任务”会取消后台协程和正在进行的 HTTP 请求，不是
只停止动画。页面按阶段显示请求次数、输入/输出、思考和缓存 Token；思考 Token 已包含在输出
Token 中，不会重复计费。提供商未返回完整 usage 或重试失败请求没有 usage 时会明确标记估算。
DashScope Qwen 默认关闭思考模式以减少无谓输出，也可在模型配置中切换。

题库与评测整合在同一顶部入口，默认先显示题库；点击“进入题目”后在同一工作台连续展示完整
题面、限制、样例、在线编辑器和独立“提交评测”按钮。个人历史位于顶部独立“提交记录”页，
每页显示 5 条，记录以不同颜色区分 AC、WA、TLE、MLE、RE、CE、UNK 与 Pending；点击
“详情”可查看源代码和逐测试点结果。编辑器提供行号、Tab 缩进、括号补全、语法高亮、查找
替换和 VS Code 快捷键。

## 测试与质量检查

```bash
~/.venvs/online-judge-system/bin/ruff check .
~/.venvs/online-judge-system/bin/ruff format --check .
~/.venvs/online-judge-system/bin/pytest -q --cov=app --cov-report=term-missing
```

测试包含真实 Python/C++ 子进程以及 AC、WA、RE、CE、TLE、MLE、权限矩阵、错误优先级、
审计、重置、AI 用量、实际取消和前端 Cookie 跨 rerun/退出回归。GitHub Actions 会在
Ubuntu/Python 3.10 上执行相同检查，并要求后端覆盖率不低于 85%。

## Git 提交规范

提交遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/)。启用仓库内
提交信息检查：

```bash
git config core.hooksPath .githooks
```

合法示例：`feat(judge): support C++14 submissions`、`fix(auth): reject banned sessions`。

验收演示顺序见 [`docs/demo-checklist.md`](docs/demo-checklist.md)，实验报告源文件见
[`docs/report.md`](docs/report.md)。
