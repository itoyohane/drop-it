# API、Tool 与数据参考

后端入口是 `backend.main:create_app`。开发环境的 OpenAPI 页面位于 `/api/docs`，生产模式关闭。
除非另行说明，请求和响应均为 JSON；错误使用 FastAPI 的 `{ "detail": "..." }` 格式。

代码入口分别位于 `backend/agent/tools.py`、`backend/agent/intent.py`、
`backend/agent/memory.py`、`backend/repositories/` 和 `backend/workers/analyze_track.py`。

## Agent 工具

三个工具都由服务端绑定当前项目，模型不能传入 `project_id`。结果统一序列化为：

```json
{"ok": true, "summary": "找到 3 首曲目。", "data": {"tracks": []}}
```

### `search_library`

| 参数 | 类型与默认值 | 说明 |
| --- | --- | --- |
| `query` | string，`""` | 声音/风格描述；非空时检索歌曲描述文本向量 |
| `filters` | `MusicFilters | null` | 标题、艺人、BPM、能量、Camelot 条件 |
| `limit` | integer，20，1–100 | 最多返回数 |

返回曲目的事实字段、生成描述和可选描述余弦分数。空 query 不要求文本索引。

### `find_similar_tracks`

| 参数 | 类型与默认值 | 说明 |
| --- | --- | --- |
| `track_id` | string，必填 | 先由 `search_library` 解析的当前曲库 ID |
| `limit` | integer，3，1–100 | 最多返回数，不包含参考歌曲 |
| `filters` | `MusicFilters | null` | 对候选歌曲的精确条件 |

参考歌没有当前描述向量会失败。输出含 `description_similarity` 与综合 `score`，二者均不是概率。

### `generate_dj_set`

| 参数 | 类型与默认值 | 说明 |
| --- | --- | --- |
| `request` | string，必填 | 用户的原始 Set 要求，也作为标题和备注来源 |
| `duration_min` | integer，45，10–240 | 目标分钟数 |
| `bpm_min` / `bpm_max` | integer，110/140，60–220 | 候选范围；下限不得高于上限 |
| `energy_curve` | `steady/build/peak/wave`，`build` | 能量曲线 |
| `style_query` | string，`""` | 可选歌曲描述语义检索条件 |
| `track_ids` | string list/null，最多 500 | 可选候选白名单，必须全部属于当前项目 |

成功时保存 Playlist，返回 `playlist_id`、规则报告和曲目事实。全局曲库对话不能保存 Set。

### `MusicFilters`

| 字段 | 默认值与限制 |
| --- | --- |
| `title` / `artist` | 空字符串；不区分大小写的包含匹配 |
| `bpm_min` / `bpm_max` | 0 / 300，范围 0–300 |
| `energy_min` / `energy_max` | 0 / 1，范围 0–1 |
| `camelot_key` | 空字符串或 `1A`–`12A`、`1B`–`12B` |

## HTTP API

### 系统、项目与曲库

| 方法与路径 | 请求 | 结果 |
| --- | --- | --- |
| `GET /api/health` | 无 | 版本、三个工具、模型状态、`librosa-deepseek-dashscope-chroma-rag` provider |
| `GET/POST /api/projects` | `ProjectCreate` 用于 POST | 列表或创建项目，POST 为 201 |
| `GET/PATCH/DELETE /api/projects/{id}` | `ProjectUpdate` 用于 PATCH | 项目读取、更新或删除 |
| `GET /api/library` | 无 | 单用户全局曲库 |
| `GET /api/projects/{id}/library` | 无 | 当前项目曲库 |
| `GET /api/projects/{id}/sources` | 无 | 项目绑定的文件夹来源 |
| `POST /api/projects/{id}/sources/resolve` | `folder_key`、`name` | 创建/复用并绑定来源 |
| `POST /api/projects/{id}/library/import-files` | multipart：`source_id`、`files` | 保存支持的文件并创建分析任务，202 |
| `POST /api/projects/{id}/library/analyze` | `{"track_ids":[]}` | 空数组补缺失阶段；明确 ID 强制重分析，202 |
| `PATCH /api/projects/{id}/library/{track_id}` | 完整 `TrackUpdate` | 修改标题、艺人、BPM、调性、Camelot、能量 |
| `DELETE /api/projects/{id}/library/{track_id}` | 无 | 从当前项目解绑，204；不会删除总曲库中的原曲目。对 `global-chat` 调用会返回 400 |
| `GET /api/projects/{id}/jobs` | 无 | 最近任务 |
| `GET /api/jobs/{job_id}` | 无 | 单个任务 |

导入支持 `.mp3`、`.wav`、`.flac`、`.aiff`、`.aif`、`.m4a`。文件分块落盘；
默认单文件 1024 MB、单批 500 首。调用方必须先绑定 source，导入成功会自动提交分析任务。

### 对话

项目对话：

- `GET/POST /api/projects/{id}/conversations`
- `DELETE /api/projects/{id}/conversations/{conversation_id}`
- `GET /api/projects/{id}/conversations/{conversation_id}/messages`
- `POST /api/projects/{id}/conversations/{conversation_id}/chat`
- `POST /api/projects/{id}/conversations/{conversation_id}/chat/stream`

全局曲库对话使用 `/api/chat/conversations`、`/api/chat/conversations/{id}/messages` 和
`/api/chat/conversations/{id}/chat/stream`。对话请求为 `{ "message": "..." }`，1–4000 字符。
聊天未配置 `DEEPSEEK_API_KEY` 时在写消息前返回 503；曲库管理和分析仍可用。

SSE 每帧为 `data: <JSON>\n\n`：

- `user_saved`：用户消息已保存；
- `status`：当前状态标签；
- `token`：模型增量文本；
- `tool`：`name`、`running/done/failed`、`summary`；
- `error`：可展示的错误；
- `complete`：最终消息和可选 Playlist。

### Playlist

| 方法与路径 | 请求 | 结果 |
| --- | --- | --- |
| `GET/POST /api/projects/{id}/playlists` | `Brief` 用于 POST | 历史或不经 Agent 的规则编排 |
| `PUT /api/playlists/{id}/order` | 完整、无重复的 `track_ids` | 保存顺序，revision 加一 |
| `POST /api/playlists/{id}/approve` | 无 | 状态改为 approved |
| `DELETE /api/playlists/{id}` | 无 | 删除 Set，204 |
| `POST /api/playlists/{id}/export` | format 为 `json/csv/m3u` | 文本格式导出 |

## 主要数据状态

- `Track.analysis_status`：`pending/analyzing/analyzed/failed`。
- `Track.embedding_status`：`pending/ready/failed`。与分析状态独立。
- `Job.status`：`queued/running/completed/failed/cancelled`；kind 保留 `analyze/reindex` 枚举。
- `Playlist.status`：`draft/approved`；时长是整曲秒数之和。

向量存在 Chroma，按 model key 分集合；SQLite 仅保存向量状态和模型 key。项目成员关系仍由 SQLite 校验后再读取 Chroma。
本地音频路径只用于服务端处理，不进入 Tool 给模型的上下文。

## 环境变量

| 名称 | 默认值 | 用途 |
| --- | --- | --- |
| `DROPIT_ENV` | `development` | `development/test/production` |
| `DROPIT_DATA_DIR` | `data` | SQLite、导入音频与默认 Chroma 目录根目录 |
| `DEEPSEEK_API_KEY` | 空 | 服务端聊天凭证，也可作为歌曲描述 API key 的回退 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容端点 |
| `DROPIT_MODEL` | `deepseek-v4-pro` | 聊天模型名 |
| `DEEPSEEK_DESCRIPTION_API_KEY` | 空 | 歌曲描述专用 DeepSeek key；为空时回退到 `DEEPSEEK_API_KEY` |
| `DEEPSEEK_DESCRIPTION_BASE_URL` | `https://api.deepseek.com` | 歌曲描述 Chat Completions 端点根地址 |
| `DEEPSEEK_DESCRIPTION_TIMEOUT_SECONDS` | `60` | 歌曲描述请求超时 |
| `DROPIT_DESCRIPTION_MODEL` | `deepseek-v4.1-flash` | 歌曲描述模型名 |
| `DASHSCOPE_API_KEY` | 空 | DashScope embedding 凭证 |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | DashScope OpenAI 兼容端点根地址 |
| `DASHSCOPE_TIMEOUT_SECONDS` | `60` | embedding 请求超时 |
| `DROPIT_TEXT_EMBEDDING_MODEL` | `qwen3.7-text-embedding` | 歌曲描述和查询向量模型 |
| `DROPIT_TEXT_EMBEDDING_DIMENSIONS` | `1024` | DashScope 输出维度；需与向量索引一致 |
| `DROPIT_CHROMA_DIR` | `data/chroma` | Chroma 持久化目录 |
| `DROPIT_INTENT_FALLBACK_ENABLED` | `true` | 规则未命中时是否调用 Ollama 意图分类 |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434/v1` | Ollama OpenAI 兼容端点 |
| `OLLAMA_MODEL` | `hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M` | 意图分类模型 |
| `OLLAMA_TIMEOUT_SECONDS` | `8` | 意图分类请求超时 |
| `DROPIT_AGENT_MEMORY_MAX_MESSAGES` | `100` | 压缩前的内存消息数安全上限 |
| `DROPIT_AGENT_MEMORY_TTL_SECONDS` | `1800` | 活跃会话内存 TTL |
| `DROPIT_AGENT_CONTEXT_WINDOW_TOKENS` | `32768` | 当前聊天模型的上下文窗口配置 |
| `DROPIT_AGENT_CONTEXT_COMPACTION_RATIO` | `0.8` | 自动压缩触发比例 |
| `DROPIT_AGENT_CONTEXT_KEEP_MESSAGES` | `6` | 压缩时原样保留的最近消息数 |
| `DROPIT_AGENT_CONTEXT_RESERVED_TOKENS` | `4096` | 为工具结果和回答预留的 token |
| `DROPIT_AGENT_CONTEXT_SUMMARY_TOKENS` | `512` | 历史摘要目标长度 |
| `DROPIT_DESCRIPTION_MODEL_REVISION` | `api` | 描述模型 key 版本标签 |
| `DROPIT_TEXT_EMBEDDING_MODEL_REVISION` | `api` | 向量模型 key 版本标签 |
| `DROPIT_CORS_ORIGINS` | 两个本地 5173 地址 | 逗号分隔允许源 |
| `DROPIT_MAX_UPLOAD_MB` | `1024` | 单文件限制，允许 10–4096 |
| `DROPIT_MAX_UPLOAD_FILES` | `500` | 单批限制，允许 1–5000 |
| `LANGSMITH_TRACING` | `false` | 是否启用 LangSmith tracing |
| `LANGSMITH_API_KEY/PROJECT/ENDPOINT` | 空/`drop-it-dev`/官方端点 | tracing 配置 |
| `DROPIT_LOG_LEVEL` | `INFO` | 日志级别设置项 |

`.env` 与 `.env.local` 会被读取且已忽略，不要提交真实密钥。当前没有 `DROPIT_ACCESS_TOKEN`，
也没有用户认证或租户边界。公网部署前必须增加身份认证，并让所有项目、曲目、对话、任务和 Set
按用户/租户过滤，而不只是依赖项目 ID。

参见[架构说明](explanation-agent-architecture.md)和[运行教程](tutorial-run-and-observe.md)。
