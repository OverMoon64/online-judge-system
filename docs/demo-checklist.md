# 线下验收演示清单

## 演示前

- 在 VS Code Remote WSL 中打开仓库，确认 Python 3.10 与 G++ 可用。
- 从任务面板启动后端和前端，打开 `/health`、`/docs` 和 Streamlit 首页。
- 执行 `python scripts/seed_demo.py`，准备管理员、普通用户和 A+B 题目。
- 在另一个浏览器会话保留普通用户登录，便于演示权限矩阵。
- 在 AI 页面临时填写百炼 URL、模型、Key 和当日价格；不要展示或录制 Key。

## 10 分钟主流程

1. 展示 GitHub Conventional Commit 历史和绿色 CI，说明所有 API 使用 `async def`。
2. 普通用户登录，浏览题目并提交 Python AC；观察 `pending → success` 和逐点日志。
3. 再提交 WA、RE 或 TLE，说明 Submission 状态与测试点结果的区别。
4. 管理员提交/重新评测 C++，再提交一份语法错误代码展示 CE。
5. 普通用户尝试查看他人私有日志得到 403；管理员公开题目日志后再次查看成功。
6. 管理员打开访问审计，展示同一用户的 403 和 200 记录。
7. 管理员更改用户角色为 banned，展示已有会话和再次登录均被拒绝。
8. AI 页面输入知识点、难度和边界要求，展示分阶段进度、Token 和费用。
9. 先创建一项 AI 任务并中断，证明后台实际停止；再完成一项并载入题目新增表单。
10. 展示测试报告、Linux CI、架构图和实验报告中的边界测试结果。

## 备用命令

```bash
curl http://127.0.0.1:8000/health
pytest -q --cov=app --cov-report=term-missing
python scripts/seed_demo.py
```

