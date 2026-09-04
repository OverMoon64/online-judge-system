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

模型配置完全来自页面：提供商 Base URL、模型名、API Key、输入/输出单价、计价单位和币种。
API Key 仅保存在后端进程内存中，重启或系统重置后清除，查询接口只返回
`api_key_configured: true`。

验收演示建议使用阿里云百炼 OpenAI-compatible 接口和 `qwen3.7-plus`。请从百炼控制台复制
业务空间 Base URL 和 API Key；价格可能变化，应把控制台当日价格填写到页面，程序不会硬编码。

命题任务依次执行需求分析、题目/参考解法生成、Pydantic 结构校验、测试点去重、受限运行参考
解法和一次自动修正。页面每秒轮询进度；“中断任务”会取消后台协程和正在进行的 HTTP 请求，
不是只停止动画。模型未返回 usage 时会明确标记 Token/费用不可精确统计。

## 测试与质量检查

```bash
~/.venvs/online-judge-system/bin/ruff check .
~/.venvs/online-judge-system/bin/ruff format --check .
~/.venvs/online-judge-system/bin/pytest -q --cov=app --cov-report=term-missing
```

测试包含真实 Python/C++ 子进程以及 AC、WA、RE、CE、TLE、MLE、权限矩阵、错误优先级、
审计、重置、AI 用量和实际取消。GitHub Actions 会在 Ubuntu/Python 3.10 上执行相同检查，
并要求后端覆盖率不低于 85%。

## Git 提交规范

提交遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/)。启用仓库内
提交信息检查：

```bash
git config core.hooksPath .githooks
```

合法示例：`feat(judge): support C++14 submissions`、`fix(auth): reject banned sessions`。

验收演示顺序见 [`docs/demo-checklist.md`](docs/demo-checklist.md)，实验报告源文件见
[`docs/report.md`](docs/report.md)。

