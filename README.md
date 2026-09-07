# DropIt DJ Agent

DropIt 的后端以 LangChain Agent 为对话入口，使用 librosa 分析音乐属性，并用小模型生成歌曲文本描述进行本地 RAG。
Agent 只调用三个工具：`search_library`、`find_similar_tracks`、`generate_dj_set`。

音乐 RAG 保存歌曲描述的文本向量；查询文本通过同一个 MiniLM 文本模型临时编码。曲目信息、描述、向量、
会话、后台任务和 Set 均保存在 SQLite，不需要普通文本向量库、Chroma 或外部音乐 API。

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

启动后访问 `http://127.0.0.1:8765/api/docs`。配置服务器的 `DEEPSEEK_API_KEY` 后可使用 Agent 对话；
缺少聊天密钥不影响曲库管理和本地音乐分析。

`backend/agent/tools.py` 直接实现三个工具；`services/` 已删除。`intend.py` 负责意图提示，
`memory.py` 提供有界、带 TTL 的短期对话记忆。

```bash
python -m pip install -r requirements.txt
python -m pytest backend/tests -q
```
