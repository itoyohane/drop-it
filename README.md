# DropIt DJ Agent

DropIt 的后端以 LangChain Agent 为对话入口，使用 Essentia 分析音乐属性，CLAP 检索本地歌曲。
Agent 只调用三个工具：`search_library`、`find_similar_tracks`、`generate_dj_set`。

音乐 RAG 仅保存歌曲的音频向量；查询文本通过同一个 CLAP 模型临时编码。曲目信息、音频向量、
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

本次重构范围是后端 API、Agent、检索、任务和模型编码接口；前端没有随本次后端分层一起重构。

```bash
python -m pip install -r requirements.txt
python -m pytest backend/tests -q
```
