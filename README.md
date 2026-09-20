# DropIt DJ Agent

DropIt 的后端以 LangGraph StateGraph 为对话入口，使用 librosa 分析音乐属性，DeepSeek-V4.1-Flash 生成歌曲文本描述，
DashScope `qwen3.7-text-embedding` 生成向量，并用 Chroma 做音乐 RAG。
Agent 只调用三个工具：`search_library`、`find_similar_tracks`、`generate_dj_set`。

音乐 RAG 保存歌曲描述的文本向量；查询文本通过同一个 DashScope embedding API 编码。
曲目信息、描述、会话、后台任务和 Set 保存在 SQLite，向量保存在 `DROPIT_CHROMA_DIR` 指向的 Chroma 目录。

文档入口：[docs](docs/README.md)。

- [运行与验证](docs/tutorial-run-and-observe.md)：安装、模型下载、导入、对话和故障排查。
- [架构与边界](docs/explanation-agent-architecture.md)：模块职责、音乐 RAG、数据范围和取舍。
- [接口与配置](docs/reference-api-tools-and-data.md)：三个 Tool、HTTP/SSE、数据与环境变量。
- [项目讲解](docs/howto-ai-application-interview.md)：演示顺序、设计理由和测试边界。

```bash
docker compose -f compose.backend.yaml build
docker compose -f compose.backend.yaml run --rm api python -m backend.music.download_models
docker compose -f compose.backend.yaml up
```

启动后访问 `http://127.0.0.1:8765/api/docs`。配置 `DEEPSEEK_API_KEY` 后可使用 Agent 对话，
配置 `DASHSCOPE_API_KEY` 后才能建立/查询歌曲描述向量；缺少任一密钥时，曲库管理和 librosa 数值分析仍可用。

`backend/src/agent/tools.py` 直接实现三个兼容 Tool 及其确定性业务操作；`services/` 已删除。`intent.py` 负责规则优先、
Ollama 兜底的意图识别与 `overstep` 拒答路由；`graph.py` 用显式分支完成检索、规划、持久化和终止；`memory.py` 提供带 TTL 的短期记忆和 80% 阈值上下文压缩。

```bash
python -m pip install -r requirements.txt
python -m pytest backend/src/tests -q --basetemp .pytest-tmp -p no:cacheprovider
```

当前工作区正在从 `backend/` 迁移到 `backend/src/`；根目录的 `backend/__init__.py` 保留
`backend.*` 导入和 `python -m backend...` 入口。显式指定工作区临时目录并关闭 pytest cache，
可避免受限 Windows 环境把临时文件写到工作区外。
