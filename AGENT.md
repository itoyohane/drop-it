# 项目约定
- 你是一个精通dj软件相关技术的开发者和dj艺术家

## 技术栈

- 后端：Python、FastAPI、LangChain 单 Agent、SQLite。
- 音乐分析：librosa；DeepSeek-V4.1-Flash 生成歌曲文本描述，DashScope `qwen3.7-text-embedding` + Chroma 用于音乐 RAG。
- 工具业务逻辑集中在 `backend/src/agent/tools.py`，不设 `services/` 层；LangChain `@tool`
  兼容适配器位于 `backend/src/agent/tool_adapters.py`，受控 Graph 节点直接调用 Registry。
- Agent 工具仅有 `search_library`、`find_similar_tracks`、`generate_dj_set`。
- 前端：React + Vite；当前工作区仍是旧原型，尚未适配新后端。
- 架构与开发说明见 [docs](docs/README.md)；界面设计参考 [DESIGN.md](DESIGN.md)。

## 常用命令

- 全部轻量测试：`npm.cmd test`
- 后端离线测试：`python -m pytest backend/src/tests -q --basetemp .pytest-run -p no:cacheprovider`
- 完整音频依赖：`python -m pip install -r requirements-audio.txt`
- 检查 API 模型配置：`python -m backend.music.download_models`
- 启动后端：`python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765`

## 代码规范
- 改动前先阅读相关模块与现有模式。
- 保持改动小而聚焦；不要顺带重构无关代码。
- 修改行为后，运行相关测试或检查命令。
- 每次改动完成后，都必须创建一个对应的 Git commit，以便后续追踪和回滚。
- 每次改动后，都必须编写或更新相关测试，并在交付给用户前，确保所有测试和验证全部通过。
## 安全与边界
- 不要提交 `.env`、密钥或真实用户数据。
- 不要执行破坏性 Git 操作（如 `reset --hard`）。
- 数据库 schema 变更必须同时提供 migration。

## 提交规范
- 使用 Conventional Commits，例如：`feat: 添加导出功能`
