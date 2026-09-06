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
~/.venvs/online-judge-system/bin/streamlit run frontend/app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
```

- 前端：<http://127.0.0.1:8501>
- API 文档：<http://127.0.0.1:8000/docs>
- 初始管理员：`admin / admintestpassword`

## 前端会话与界面

Streamlit 把同一个 `httpx.Client` 保存在 `st.session_state` 中，并把后端签名 Session 作为
加密载荷写入浏览器 Cookie。普通 rerun 直接复用客户端；浏览器完整刷新后，前端恢复载荷并调用
`GET /api/auth/session` 让后端重新验证用户、角色和服务端 Session Token。载荷绑定当前 Streamlit
服务进程，服务重启后旧载荷会被删除，因此不会自动恢复上一次的管理员账号。退出会同时撤销数据库中的
随机 Token、调用后端 logout 并清理本地及浏览器 Cookie，因此旧签名 Cookie 也不能重放登录。
若后端返回 401，页面会提示登录已失效并回到账户入口，不会依赖前端角色状态绕过认证。

Streamlit 固定以 headless 模式启动，不会尝试从 WSL 调用 Windows 浏览器。即使 WSL Interoperability
被禁用，也可以在 Windows 浏览器中手动打开 <http://127.0.0.1:8501>。

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
- 后台语言管理页列出 Python 3、C/C++、Java、JavaScript、Ruby 和 Go 的安全命令示例；
  Python 3 与 C++14 为内置语言，其他语言需先在 WSL 安装对应程序再注册。

## AI 智能命题

模型配置完全来自页面：配置名、提供商 Base URL、模型名、API Key、输入/输出单价、计价单位
和币种。每位用户可保存并选择多个模型；模型配置使用由 `OJ_SESSION_SECRET` 派生的密钥加密
保存在本机文件中，重启后无需重新输入。API、数据库和日志都不返回或记录 Key；系统重置会
删除加密配置文件。

客户端支持使用 Bearer API Key 的任意 OpenAI-compatible Chat Completions 地址，模型名不设
白名单；配置页明确列出 Qwen、DeepSeek、Kimi 的示例 URL/模型，并标注 OpenAI、GLM、
OpenRouter、SiliconFlow 等通用兼容服务。原生 Anthropic、Gemini 或 Azure 特殊鉴权接口不属于
这个协议范围。请求不再向所有模型强制发送采样温度，避免 Kimi K2.5/K2.6 因不接受 `0.35`
而返回 400。

验收演示建议使用阿里云百炼 OpenAI-compatible 接口和 `qwen3.7-plus`。请从百炼控制台复制
业务空间 Base URL 和 API Key，并按服务商账单规则手动填写输入/输出单价。OpenAI-compatible
协议没有统一价格接口，项目不自动抓取价格，避免地区、阶梯或调价差异造成误导。

进入 AI 模块后默认打开“智能命题”，可选择自动或指定难度。系统按最终难度要求至少生成
2/2/3/5 个入门/简单/中等/困难隐藏测试点，也可以手动指定 2–10 个。命题任务依次执行
需求分析、题面与参考解法生成、结构与测试数据校验、双语言交叉试运行和最多两次自动修正。
每道题同时生成 Python 3 与 C++14 参考程序。两份程序会运行全部样例和隐藏测试点；输出一致且
存在正确安全锚点时，系统用共识结果校准错误或空白答案。校准完全在本地完成，不增加 Token 或费用。
模型服务临时失败时自动重试两次；中断任务会取消后台协程和正在进行的 HTTP 请求。

每次命题可选自动、低、中、高、最高五档推理强度。自动档保留提供商默认行为；百炼 Qwen
使用 `enable_thinking/thinking_budget`（Qwen3.8 使用 `reasoning_effort`），DeepSeek 使用
`reasoning_effort`，Kimi 官方接口使用 `thinking` 开关并把档位写入命题提示。其他兼容服务先发送
标准 `reasoning_effort`，若接口以 400/422 拒绝该可选字段，系统会自动移除字段并使用模型默认
设置继续，不把参数差异计为一次命题修复。

为优先保证不同兼容模型的生成成功率，系统不再要求模型编写和执行压力数据生成器或低效探针。
模型只需返回完整、紧凑的测试点；系统忽略模型套用的固定资源值，按最终难度和算法结构统一设置
时间、内存限制，需求中明确填写的限制仍优先。新 AI 题目由服务端依次编号为 AI0001、AI0002……，
修改已有题目时保持原 ID。

每道新 AI 题在普通命题和双语言校验成功后，都会尝试确定性构造“最小边界 + 最大规模”文件点，
不再依赖需求中是否出现压力测试关键词。目前支持单整数边界、计数整数数组、单字符串、整数网格
和图输入模板；Python/C++ 参考程序必须都在限制内完成且规范化输出一致，系统才会把输入与共识答案
保存为 `.in/.out` 并纳入判题。文件默认位于忽略提交的
`OJ_TESTCASE_DIR=./data/testcases`，以题目输入契约指纹和 SHA-256 校验绑定；修改输入契约、删题或
系统重置会清理对应文件。无法识别输入结构、参考程序超时或双解不一致时只跳过该增强，题目仍
正常生成，不触发模型修复，也不增加模型调用、Token 或费用。预览页按答案正确性、输入多样性、
边界覆盖、数据规模和复杂度区分证据显示 0–100 有效性结果；缺少证据时明确要求人工补强，不虚报
通过。该机制用于提高暴力解被识别的概率，不等同于形式化复杂度证明，导入前仍需人工审阅。

约束解析兼容 `100,000`、`1 <= N <= 10^5`、LaTeX `1 \\le n \\le 10^5` 和
`5 \\times 10^5` 等模型常用写法。要自动得到满分有效性证据，题面需给出受支持的输入结构及
明确数值上下界；无法安全推断时系统保留已校验的普通测试点并提示人工复核。

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

测试包含真实 Python/C++ 子进程、动态注册 C17，以及 AC、WA、RE、CE、TLE、MLE、权限矩阵、错误优先级、
审计、重置、AI 用量、实际取消、双语言答案校准、边界/规模文件点、浏览器完整刷新及退出重放回归。
文件压力点集成测试会让普通小点可通过的 O(n²) 程序在大规模 `.in` 上触发 TLE，并验证增强失败
不会降低 AI 命题成功率；判题计时测试保证 AC 的显示耗时不超过限制。GitHub Actions 会在
Ubuntu/Python 3.10 上执行相同检查，并要求后端
覆盖率不低于 85%。

## Git 提交规范

提交遵循 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/v1.0.0/)。启用仓库内
提交信息检查：

```bash
git config core.hooksPath .githooks
```

合法示例：`feat(judge): support C++14 submissions`、`fix(auth): reject banned sessions`。

验收演示顺序见 [`docs/demo-checklist.md`](docs/demo-checklist.md)，实验报告源文件见
[`docs/report.md`](docs/report.md)。
