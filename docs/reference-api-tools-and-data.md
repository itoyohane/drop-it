# DropIt 接口、命令与数据参考

本页用于查参数，不承担入门教学。先运行项目看[动手教程](tutorial-run-and-observe.md)，理解执行原因看[架构解释](explanation-agent-architecture.md)，准备讲解看[面试指南](howto-ai-application-interview.md)。返回[学习首页](README.md)。

依据是 [main.py](../backend/src/main.py)、[models.py](../backend/src/models.py)、[graph.py](../backend/src/agent/graph.py) 和 [config.py](../backend/src/config.py)，不是旧评测适配器。源码位于 `backend/src/`；[包路径兼容入口](../backend/__init__.py) 保留 `backend.*` 导入和 `python -m backend...` 命令。

## 1. HTTP 接口

开发环境 API 根地址是 `http://127.0.0.1:8765`，交互文档是 `/api/docs`；生产环境关闭该交互文档。Vite 将浏览器同源的 `/api` 转发给后端。

花括号表示资源 ID。默认请求体为 JSON，上传为 multipart。`201` 表示创建成功，`202` 只表示后台任务已受理，`204` 无响应体；其他成功响应通常为 `200`。

| 方法与路径 | 输入 / 返回与行为 |
| --- | --- |
| `GET /api/health` | 状态、版本、三个业务操作名、模型配置标志、模型名和索引 provider；不验证远端模型可用性 |
| `GET /api/projects` | `{projects: [...]}` |
| `POST /api/projects` | `{name, description?}`；`201`，返回 Project |
| `GET /api/projects/{project_id}` | Project |
| `PATCH /api/projects/{project_id}` | `{name, description?}` |
| `DELETE /api/projects/{project_id}` | 删除项目；`204`，仅在明确需要时调用 |
| `GET /api/projects/{project_id}/conversations` | `{conversations: [...]}` |
| `POST /api/projects/{project_id}/conversations` | `{title?}`；`201`，返回 Conversation |
| `DELETE /api/projects/{project_id}/conversations/{conversation_id}` | 删除会话；`204` |
| `GET /api/projects/{project_id}/conversations/{conversation_id}/messages` | `{messages: [...]}` |
| `POST /api/projects/{project_id}/conversations/{conversation_id}/chat` | ChatRequest → ChatResponse，等待完整结果 |
| `POST /api/projects/{project_id}/conversations/{conversation_id}/chat/stream` | ChatRequest → SSE |
| `GET /api/chat/conversations` | 全局会话列表 |
| `POST /api/chat/conversations` | `{title?}`；`201`，创建全局会话 |
| `DELETE /api/chat/conversations/{conversation_id}` | 删除全局会话；`204` |
| `GET /api/chat/conversations/{conversation_id}/messages` | 全局会话消息 |
| `POST /api/chat/conversations/{conversation_id}/chat/stream` | 全局 SSE；可查总曲库，生成 Set 必须进入具体项目 |
| `GET /api/library` | `{tracks: [...], scope: "global"}` |
| `GET /api/projects/{project_id}/library` | `{tracks: [...], scope: "project"}` |
| `GET /api/projects/{project_id}/sources` | `{sources: [...]}` |
| `POST /api/projects/{project_id}/sources/resolve` | `{folder_key, name}`；解析并绑定来源，返回 `source`、`reused`、`tracks` |
| `POST /api/projects/{project_id}/library/import-files` | multipart：`source_id`、多个 `files`、`reanalyze=false`（true 强制检测重新上传的音频）；`202`，返回 job/tracks/count/skipped |
| `POST /api/projects/{project_id}/library/analyze` | `{track_ids: []}`；`202`；空列表处理未 ready 曲目，显式 IDs 强制重分析，音频缓存已释放/缺失返回 `409` 并保留原分析；无待处理曲目时 `job=null` |
| `PATCH /api/projects/{project_id}/library/{track_id}` | TrackUpdate；更新事实并提交描述、向量重建任务 |
| `DELETE /api/projects/{project_id}/library/{track_id}` | `204`，解除当前项目关联，不等于删除其他项目的同一歌曲 |
| `GET /api/projects/{project_id}/jobs` | `{jobs: [...]}` |
| `GET /api/jobs/{job_id}` | Job；没有对外的取消任务接口 |
| `GET /api/projects/{project_id}/playlists` | `{playlists: [...]}` |
| `POST /api/projects/{project_id}/playlists` | Brief → Playlist；直接确定性规划，不经过聊天参数提取；约束冲突 `409` |
| `PUT /api/playlists/{playlist_id}/order` | `{track_ids: [...]}`；恰好是原曲目集合且无重复，revision 加一 |
| `POST /api/playlists/{playlist_id}/approve` | 状态改为 approved |
| `DELETE /api/playlists/{playlist_id}` | 删除 Set；`204` |
| `POST /api/playlists/{playlist_id}/export` | `{format: "json" | "csv" | "m3u"}`；纯文本，可能含本地文件路径 |

常用请求边界：项目 name 为 1–80 字符、description 最多 300；会话 title 为 1–80，默认“新对话”；来源 folder_key 为 16–128、name 为 1–200。TrackUpdate 必须同时提供 title（1–200）、artist（1–200）、bpm（40–260）、key（1–32）、camelot_key（1–4）和 energy（0–1）；它对 Camelot 字符串只检查长度，不代表调性语义已经有效。

聊天入口检查模型配置，缺少 `DEEPSEEK_API_KEY` 返回 `503`，包括本来能静态拒答的请求。资源不存在通常为 `404`，请求 schema 校验失败为 `422`。流建立后的业务错误可能通过 SSE `error` 返回，不能只看 HTTP 状态。`X-Request-ID` 用于关联日志，不是身份凭证。

## 2. 聊天与 SSE

ChatRequest：`message` 为 1–4000 字符，`run_id` 可省略；显式 ID 为 1–128 字符。ChatResponse 包含已保存的助手 `message`、`playlist`、`model_configured`、`run_id`、`resumed`、`error_code`。

每个帧为 `data: <JSON>` 加两个换行，不依赖命名的 `event:` 字段。

| type | 主要字段 | 消费方式 |
| --- | --- | --- |
| `user_saved` | message、run_id | 用已保存 ID 校正乐观用户消息 |
| `status` | label、intent；通常有 run_id，部分有 resumed | 显示思考/上下文压缩状态 |
| `token` | content | 追加回答；网络 chunk 不等于完整 SSE 帧 |
| `tool` | name、status、summary | schema 允许 running/done/failed，当前图主要发 done/failed |
| `error` | detail、error_code；部分有 run_id | 保留失败信息 |
| `complete` | message、playlist、run_id、resumed、error_code | 最终权威结果；仍需检查 error_code |

Set 冲突：聊天 `error_code="constraint_conflict"`、`playlist=null`；直接创建 Playlist 接口为 `409` 的结构化 detail。外层流异常可能只有 error，没有 complete。

恢复同一逻辑请求须重传相同 run_id、项目、会话和原始 message；修改 message 应使用新 ID。服务端检查文本和范围一致。当前[前端 API](../frontend/src/api.js)只发送 `{message}`，尚未自动保存、重传 run_id；服务端支持恢复不等于 UI 已实现断线续跑。

## 3. 三个业务命令

这是 Graph 的输入契约，不是模型自由选择的 LangChain `@tool` 集合。先确定路由，再由 `extract_command` 提取对应的 Pydantic 命令；额外字段禁止，project/conversation/store 不在 schema 中。

| 路由 → 命令 | 字段、默认值与边界 |
| --- | --- |
| `search_library` → SearchCommand | query=""，最多 500 字符；filters={}；limit=20，1–100 |
| `find_similar_tracks` → SimilarCommand | reference=""，最多 200 字符；filters={}；limit=3，1–100；空 reference 是业务错误，不猜歌曲 |
| `generate_dj_set` → GenerateSetCommand | request=""，最多 4000 字符，提取后为空则补用户原话；duration_min=45，10–240；bpm_min=110、bpm_max=140，各 60–220；energy_curve="build"，可选 steady/build/peak/wave；style_query=""，最多 500 字符；track_ids、required_tracks 默认为 null，各最多 500 项 |

`track_ids` 限定候选集，`required_tracks` 是必须包含的歌曲 IDs；不能把任意歌名当 required ID。命令只检查各 BPM 字段范围，交叉关系由后续筛选、业务校验检查。

MusicFilters 默认 title/artist 为空（大小写无关的包含匹配），BPM 为 0–300，energy 为 0–1，camelot_key 为空或合法的 1A–12B。上下限必须有序；Graph 的 filters 不接受额外字段或 null。

空 query 只查元数据，score 可为 null；非空 query 才生成查询向量。参考歌按当前范围内 ID、`track_id:<ID>` 或完整标题精确匹配；同名多首要求补 ID，不做模糊猜测。

HTTP 创建 Set 使用 Brief，默认 BPM 为 `118–132`，energy="build"、style="house, warm-up"，与 Graph 默认值不同。该入口直接取项目曲目规划，style 作为 Brief 信息，不像聊天分支用 style_query 先语义召回；两个入口的能力不完全一致。

## 4. 数据、状态与约束

| 数据 / 状态 | 含义 |
| --- | --- |
| Track.analysis_status | pending / analyzing / analyzed / failed：数值分析 |
| Track.embedding_status | pending / ready / failed：描述向量；ready 仍需核对模型版本与真实向量 |
| Job.status | queued / running / completed / failed / cancelled；启动时恢复中断任务 |
| LibrarySource.status | pending / analyzing / ready / failed |
| Playlist.status | draft / approved；包含 Brief、tracks、duration_sec、report、trace、revision |
| Agent run | running / completed / failed；completed 表示执行结束，也可能带业务 error_code |
| AgentState | 命令、历史、候选、结果、歌单、校验、修复次数、错误、回答 |
| AgentRuntimeContext | 服务端范围与依赖：project_id、conversation_id、store、registry、model_factory、run_id、claim_owner/claim_token |

默认 SQLite 是 `data/dropit.db`，Chroma 是 `data/chroma/`，上传暂存于数据目录的 imports/sources 下，默认在 librosa 特征持久化后释放，分析失败保留。重启清理已分析的旧缓存，保护待恢复的强制检测任务。重复内容只保留一个上传副本，描述/向量重建不依赖音频。Track.path 保留历史位置，释放后 M3U/CSV 导出路径需映射到用户原文件，不能直接播放。旧重复副本可先停止后端，再运行 `python -m backend.music.cleanup_audio` 预览，带 `--apply` 删除；按内容哈希核对，未匹配、失败或待强制检测文件保留。SQLite 保存关系、音乐事实、描述、索引状态、消息、任务、歌单和 Agent 运行；Chroma 按 embedding model_key 隔离向量。当前按范围内歌曲 IDs 读取向量，再由 NumPy 做余弦全量排序，没有使用 Chroma ANN query。

[checkpoints.py](../backend/src/agent/checkpoints.py) 定义持久化步骤：

```text
command_parsed → candidates_retrieved → set_planned
→ set_validated ↔ set_repaired → playlist_persisted → completed
```

这是步骤词汇和 Set 主路径，不是所有分支的固定序列。唯一键为 `(run_id, step, version)`，校验/修复按轮次版本化；不是 LangGraph 内置 checkpointer。SQLite 唯一约束限制同一 run/role 的消息、同一 run 的 Agent 歌单；歌单 ID 由项目 ID 与 run_id 的规范编码派生。租约默认 15 秒，递增 fencing token 拒绝过期执行者写入。

SetValidator 默认时长相对误差 `10%`、相邻 BPM 差 `8`、能量容差 `0.20`；还检查项目/候选归属、重复、BPM 区间、曲长汇总、Camelot 相容、必选歌曲。最多修复两轮，失败不保存。时长为原曲长度求和，没有混音重叠、cue 点或自动 beat matching。手动调序和 approve 不做同一套完整硬约束复核，不能承诺后续编辑仍保持生成约束。

## 5. 配置速查

Settings 从根目录 `.env`、`.env.local` 和环境变量加载，忽略额外变量。Key 仅在服务端配置，不使用 `VITE_` 变量。以下模型名是源码默认值，不保证服务商或账号当前支持；填写账号实际可用、接口兼容的模型。

| 环境变量 | 默认 / 用途 |
| --- | --- |
| `DROPIT_ENV` | development；可选 test / production |
| `DROPIT_DATA_DIR` | data |
| `DROPIT_CHROMA_DIR` | 未指定时为数据目录下 chroma |
| `DEEPSEEK_API_KEY` | 无；聊天 Key，也可回退用于描述 |
| `DEEPSEEK_BASE_URL` | https://api.deepseek.com |
| `DROPIT_MODEL` | deepseek-v4-pro；聊天、参数提取、摘要 |
| `DEEPSEEK_DESCRIPTION_API_KEY` | 无；未配置时回退聊天 Key |
| `DEEPSEEK_DESCRIPTION_BASE_URL` | https://api.deepseek.com |
| `DEEPSEEK_DESCRIPTION_TIMEOUT_SECONDS` | 60；大于 0，最多 300 |
| `DROPIT_DESCRIPTION_MODEL` | deepseek-v4.1-flash |
| `DROPIT_DESCRIPTION_MODEL_REVISION` | api；描述版本 |
| `DASHSCOPE_API_KEY` | 无；文本嵌入 Key |
| `DASHSCOPE_BASE_URL` | https://dashscope.aliyuncs.com/compatible-mode/v1 |
| `DASHSCOPE_TIMEOUT_SECONDS` | 60；大于 0，最多 300 |
| `DROPIT_TEXT_EMBEDDING_MODEL` | qwen3.7-text-embedding |
| `DROPIT_TEXT_EMBEDDING_MODEL_REVISION` | api；索引版本 |
| `DROPIT_TEXT_EMBEDDING_DIMENSIONS` | 1024；1–4096，须与实际输出一致 |
| `DROPIT_INTENT_FALLBACK_ENABLED` | true；规则未命中时调用 Ollama |
| `OLLAMA_BASE_URL` | http://127.0.0.1:11434/v1 |
| `OLLAMA_MODEL` | hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M |
| `OLLAMA_TIMEOUT_SECONDS` | 8；大于 0，最多 120 |
| `DROPIT_AGENT_MEMORY_MAX_MESSAGES` | 100；12–1000 |
| `DROPIT_AGENT_MEMORY_TTL_SECONDS` | 1800；大于 0，最多 86400 |
| `DROPIT_AGENT_CONTEXT_WINDOW_TOKENS` | 32768；1024–2000000，应用预算而非服务商窗口保证 |
| `DROPIT_AGENT_CONTEXT_COMPACTION_RATIO` | 0.8；0.5–0.95 |
| `DROPIT_AGENT_CONTEXT_KEEP_MESSAGES` | 6；1–50 |
| `DROPIT_AGENT_CONTEXT_RESERVED_TOKENS` | 4096；0–131072 |
| `DROPIT_AGENT_CONTEXT_SUMMARY_TOKENS` | 512；64–4096 |
| `DROPIT_AGENT_RUN_LEASE_SECONDS` | 15；大于 0.05，最多 300 |
| `LANGSMITH_TRACING` | false；开启后发送外部 trace |
| `LANGSMITH_API_KEY` | 无 |
| `LANGSMITH_PROJECT` | drop-it-dev |
| `LANGSMITH_ENDPOINT` | https://api.smith.langchain.com |
| `DROPIT_CORS_ORIGINS` | http://127.0.0.1:5173,http://localhost:5173 |
| `DROPIT_MAX_UPLOAD_MB` | 1024；10–4096，单文件限制 |
| `DROPIT_MAX_UPLOAD_FILES` | 500；1–5000，单批文件数 |
| `DROPIT_RETAIN_AUDIO_FILES` | false；特征持久化后释放上传副本；true 保留音频供再次检测/路径导出 |
| `DROPIT_MUSIC_MODEL_DEVICE` | cpu；遗留本地适配器选项，不控制默认云端模型 |
| `DROPIT_MUSIC_MODELS_LOCAL_FILES_ONLY` | true；遗留选项，不代表默认模型离线 |
| `DROPIT_LOG_LEVEL` | INFO；Settings 字段，不代表所有日志处理器已自动配置 |

Vite 另读 `DROPIT_WEB_PORT`（5173）和 `DROPIT_API_TARGET`（http://127.0.0.1:8765），不属于后端 Settings。`dev:api` 固定监听 8765；改代理地址不会改变 API 监听端口。

`python -m backend.music.download_models` 仅打印模型名和目录，不下载权重、不做 API 连通性检查或音频验收。

## 6. 验证入口

| 文件 | 关注的问题 |
| --- | --- |
| [test_agent_graph.py](../backend/src/tests/test_agent_graph.py) | 分支、命令 schema、参考歌解析、无自由工具循环、回答流式输出 |
| [test_services.py](../backend/src/tests/test_services.py) | 范围、检索、索引失效、任务和 API；历史文件名不代表有 services 层 |
| [test_p0_set_and_checkpoints.py](../backend/src/tests/test_p0_set_and_checkpoints.py) | 校验/修复、版本 checkpoint、fencing、并发重试、恢复、迁移 |
| [test_intent_memory.py](../backend/src/tests/test_intent_memory.py) | 规则、Ollama 解析、TTL、压缩预算 |
| [test_audio_models.py](../backend/src/tests/test_audio_models.py) | 音频与适配器契约 |
| [test_quality_eval.py](../backend/src/tests/test_quality_eval.py) | 汇总/事实比较逻辑，不是当前模型质量验收 |
| [chat-stream.check.mjs](../frontend/tests/chat-stream.check.mjs) | SSE 分片、消息合并、失败保留 |

轻量测试用假模型/分析器做隔离，不需要远端 Key。`DROPIT_TEST_AUDIO_MODELS=1` 打开的专项仍调用遗留本地描述/嵌入适配器，不是默认云端链路 smoke test。

历史数据见[七轮报告](quality-eval-7-rounds.md)，当前复测方法见[教程](tutorial-run-and-observe.md)。任何“100%”都必须说明样本、指标定义、时间与代码版本。
