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

## 本地运行前后端

在仓库根目录安装 Python 与 Node 依赖，然后用一个命令同时启动 FastAPI 和 Vite：

```powershell
python -m pip install -r requirements.txt
npm.cmd install
npm.cmd run dev
```

- Web：`http://127.0.0.1:5173/`
- API 健康检查（经 Vite 代理）：`http://127.0.0.1:5173/api/health`
- FastAPI 文档：`http://127.0.0.1:8765/api/docs`

Vite 默认使用 `5173`，FastAPI 固定使用 `8765`；端口被占用时会直接失败。可先释放端口，或为 Web 选择另一个固定端口：

```powershell
$env:DROPIT_WEB_PORT=5174
npm.cmd run dev
```

此时 Web 和代理健康检查分别位于 `http://127.0.0.1:5174/` 与 `http://127.0.0.1:5174/api/health`。
无需 API Key 即可启动、浏览曲库并检查健康状态。聊天需要 `DEEPSEEK_API_KEY`；未配置时界面会显示后端返回的
`503` 提示，不会一直等待。歌曲描述与语义检索另需 `DASHSCOPE_API_KEY`（描述 Key 可单独配置）。

构建后的前端写入 `backend/dist`，与 FastAPI 的静态目录一致。验证生产静态服务：

```powershell
npm.cmd run build
npm.cmd run start
```

然后打开 `http://127.0.0.1:8765/`。运行全部轻量检查使用 `npm.cmd test`；也可分别运行
`npm.cmd run test:frontend` 和 `npm.cmd run test:backend`。

## Docker API

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
npm.cmd test
```

当前工作区正在从 `backend/` 迁移到 `backend/src/`；根目录的 `backend/__init__.py` 保留
`backend.*` 导入和 `python -m backend...` 入口。显式指定工作区临时目录并关闭 pytest cache，
可避免受限 Windows 环境把临时文件写到工作区外。
