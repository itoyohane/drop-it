# API、Tool 与数据参考

后端入口是 `backend.main:create_app`。开发环境的 OpenAPI 页面位于 `/api/docs`，生产模式关闭。
除非另行说明，请求和响应均为 JSON；错误使用 FastAPI 的 `{ "detail": "..." }` 格式。

代码入口分别位于 `backend/agent/tools.py`、`backend/agent/retriever.py`、
`backend/services/`、`backend/repositories/` 和 `backend/workers/analyze_track.py`。

## Agent 工具

三个工具都由服务端绑定当前项目，模型不能传入 `project_id`。结果统一序列化为：

```json
{"ok": true, "summary": "找到 3 首曲目。", "data": {"tracks": []}}
```

### `search_library`

| 参数 | 类型与默认值 | 说明 |
| --- | --- | --- |
| `query` | string，`""` | 英文声音/风格描述；空字符串只查属性 |
| `filters` | `MusicFilters | null` | 标题、艺人、BPM、能量、Camelot 条件 |
| `limit` | integer，20，1–100 | 最多返回数 |

返回曲目的事实字段和可选 CLAP 余弦分数。空 query 不要求 CLAP 索引。

### `find_similar_tracks`

| 参数 | 类型与默认值 | 说明 |
| --- | --- | --- |
| `track_id` | string，必填 | 先由 `search_library` 解析的当前曲库 ID |
| `limit` | integer，3，1–100 | 最多返回数，不包含参考歌曲 |
| `filters` | `MusicFilters | null` | 对候选歌曲的精确条件 |

参考歌没有当前 CLAP 向量会失败。输出含 `audio_similarity` 与综合 `score`，二者均不是概率。

### `generate_dj_set`

| 参数 | 类型与默认值 | 说明 |
| --- | --- | --- |
| `request` | string，必填 | 用户的原始 Set 要求，也作为标题和备注来源 |
| `duration_min` | integer，45，10–240 | 目标分钟数 |
| `bpm_min` / `bpm_max` | integer，110/140，60–220 | 候选范围；下限不得高于上限 |
| `energy_curve` | `steady/build/peak/wave`，`build` | 能量曲线 |
| `style_query` | string，`""` | 可选英文 CLAP 检索描述 |
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
| `GET /api/health` | 无 | 版本、三个工具、模型状态、`clap` provider |
| `GET/POST /api/projects` | `ProjectCreate` 用于 POST | 列表或创建项目，POST 为 201 |
| `GET/PATCH/DELETE /api/projects/{id}` | `ProjectUpdate` 用于 PATCH | 项目读取、更新或删除 |
| `GET /api/library` | 无 | 单用户全局曲库 |
| `GET /api/projects/{id}/library` | 无 | 当前项目曲库 |
| `GET /api/projects/{id}/sources` | 无 | 项目绑定的文件夹来源 |
| `POST /api/projects/{id}/sources/resolve` | `folder_key`、`name` | 创建/复用并绑定来源 |
| `POST /api/projects/{id}/library/import-files` | multipart：`source_id`、`files` | 保存支持的文件并创建分析任务，202 |
| `POST /api/projects/{id}/library/analyze` | `{"track_ids":[]}` | 空数组补缺失阶段；明确 ID 强制重分析，202 |
| `PATCH /api/projects/{id}/library/{track_id}` | 完整 `TrackUpdate` | 修改标题、艺人、BPM、调性、Camelot、能量 |
| `DELETE /api/projects/{id}/library/{track_id}` | 无 | 从项目解绑，204 |
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

向量存在 SQLite `music_embeddings`，键是 `track_id + model`，内容是归一化 little-endian float32。
本地音频路径只用于服务端处理，不进入 Tool 给模型的上下文。

## 环境变量

| 名称 | 默认值 | 用途 |
| --- | --- | --- |
| `DROPIT_ENV` | `development` | `development/test/production` |
| `DROPIT_DATA_DIR` | `data` | SQLite、导入音频与模型缓存根目录 |
| `DEEPSEEK_API_KEY` | 空 | 仅服务端聊天凭证 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | OpenAI 兼容端点 |
| `DROPIT_MODEL` | `deepseek-v4-pro` | 聊天模型名 |
| `DROPIT_CLAP_MODEL` | `laion/larger_clap_music` | Hugging Face 模型 |
| `DROPIT_CLAP_REVISION` | `a0b4534` | 固定权重版本 |
| `DROPIT_CLAP_DEVICE` | `cpu` | 推理设备 |
| `DROPIT_CLAP_LOCAL_FILES_ONLY` | `true` | 启动时禁止自动下载 |
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
