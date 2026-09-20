# Agent 架构、边界与取舍

DropIt 把自然语言音乐需求转换为当前曲库内可核实的歌曲、相似结果和 DJ Set。聊天模型负责理解与工具选择；音频分析、检索、排序和保存由本地代码执行。

## 两条执行链路

导入链路不经过聊天模型：上传音频 → Mutagen 标签与 SHA-256 去重 → SQLite Job → librosa 提取 BPM、节拍、调性、Camelot、能量和频谱特征 → DeepSeek-V4.1-Flash 把测量特征整理成歌曲文本描述 → DashScope `qwen3.7-text-embedding` 编码描述 → Chroma 持久化向量。

对话链路：`intent.py` 先执行规则识别、规则未命中时调用 Ollama → `overstep` 直接拒答，其他意图进入带 30 分钟 TTL 的短期记忆 → 上下文估算达到窗口 80% 时压缩旧消息 → 每个 `DropItAgent` 实例预先编译一个 LangGraph `StateGraph` → 条件边选择显式分支 → 模型只负责路由对应的 typed command 提取和最终自然语言回答 → 本地节点完成检索、参考歌解析、Set 规划与一次持久化 → SSE 输出并持久化完整消息。

P0 item 1 的图分支是：

```text
search_library → respond → END
resolve_reference → find_similar_tracks → respond → END
retrieve_candidates → plan_set → persist_set → respond → END
respond_chat → END
reject_response → END
```

入口通过条件边直接选择分支，没有空的 `route` 节点，也没有模型自由调用工具的 ReAct 循环。`overstep` 在图外由硬路由直接进入拒答；`music_chat` 的回答模型不绑定业务工具。P0 item 2 的 validation/repair 和 P0 item 3 的 checkpoint 不在本次范围内。

## 目录与依赖方向

```text
backend/
  __init__.py              # backend.* compatibility path for backend/src
  src/
    main.py, config.py, models.py, audio.py
    agent/
      agent.py, intent.py, memory.py, prompts.py, state.py, commands.py, graph.py, tools.py
    music/
      librosa_analyzer.py, text_models.py, indexer.py, download_models.py
    repositories/
      tracks.py, analysis.py, embeddings.py, sqlite.py
    workers/
      analyze_track.py
    migrations/
    tests/
```

项目不再设置 `services/` 层。`agent/tools.py` 直接包含曲库过滤、描述向量检索、相似度重排、Set 规划和导出，避免三项工具在多层门面间跳转。Worker 直接编排分析、描述和索引；Repository 仍集中约束 SQL 与项目成员关系。

| 模块 | 职责 |
| --- | --- |
| [main.py](../backend/src/main.py) | FastAPI、依赖装配、HTTP/SSE、导入和管理接口 |
| [agent/agent.py](../backend/src/agent/agent.py) | Agent API facade、意图与记忆接入、图执行、工具事件转换 |
| [agent/state.py](../backend/src/agent/state.py) | 原始结构化图状态与 server-only runtime context |
| [agent/commands.py](../backend/src/agent/commands.py) | 路由专属 Pydantic command 提取与校验 |
| [agent/graph.py](../backend/src/agent/graph.py) | 一次编译的 StateGraph、条件边和确定性节点 |
| [agent/intent.py](../backend/src/agent/intent.py) | 规则优先、Ollama 兜底的五路意图识别与超纲拒答 |
| [agent/memory.py](../backend/src/agent/memory.py) | 会话隔离、自动过期、token 估算与 80% 阈值上下文压缩 |
| [agent/tools.py](../backend/src/agent/tools.py) | 三个兼容 Tool 及检索、规划、持久化直接操作 |
| [music/librosa_analyzer.py](../backend/music/librosa_analyzer.py) | DJ 数值特征和描述模型输入特征 |
| [music/text_models.py](../backend/music/text_models.py) | DeepSeek 歌曲描述与 DashScope 文本 embedding API 客户端 |
| [workers/analyze_track.py](../backend/workers/analyze_track.py) | 持久化任务、阶段重试和错误隔离 |
| [repositories/sqlite.py](../backend/repositories/sqlite.py) | SQLite 曲库/任务元数据与项目成员范围校验 |
| [repositories/chroma.py](../backend/repositories/chroma.py) | Chroma 向量持久化与按模型版本读取 |

## 文本音乐 RAG

每首歌先得到一段可审计的英文描述。描述始终包含 BPM、调性、Camelot、能量、频谱中心和 onset strength 等测量事实；DeepSeek 只负责把这些特征组织成简短检索文本，不应杜撰流派、演唱者或乐器。描述和生成模型版本保存在 Track 上。

DashScope 把歌曲描述与用户查询放入同一文本向量空间。`search_library(query="")` 只做 SQL 元数据过滤，不调用 embedding API；非空 query 从 Chroma 读取当前项目已授权的向量并做精确余弦排序。`find_similar_tracks` 使用参考歌的描述向量，并按描述余弦 0.8、BPM 0.1、Camelot 0.05、能量 0.05 重排。所有分数只是当前候选集中的排序信号，不是概率。

该方案比本地 CLAP 更易部署且描述可观察，但它只能检索 librosa 特征和描述模型能够表达的属性。真实流派、乐器、歌词或情绪若未被可靠模型测量，不应当作事实。大曲库后续可使用 Chroma 的近似检索；当前仍在候选向量上做显式重排以保留 DJ 约束。

## 状态、升级与隔离

分析、描述和 embedding 分阶段处理。文本模型失败仍保留 librosa 数值结果；重试时只补缺失阶段。模型名、revision 和描述策略进入 model key，旧模型向量不会混入新检索。迁移 007 会清空旧 CLAP 向量并把 Essentia 结果标为待重分析。

工具的 `project_id` 由服务端闭包绑定，图的 `AgentRuntimeContext` 同时保存不可变的项目/会话范围和 Registry/Store 依赖；command schema 不包含 `project_id`，模型不能自行选择范围。项目对话只访问项目曲目；全局曲库可以查歌和找相似，但不能保存 Set。这是单用户范围约束，不是多租户认证。

短期记忆是进程内缓存，按项目与会话隔离，删除会话时立即清除；过期或进程重启后会从 SQLite 最近消息恢复。SQLite 仍是完整对话记录的事实来源。

librosa 在本地运行，DeepSeek 只收到测量特征（以及对话/Tool 上下文），DashScope 只收到歌曲描述或查询文本，不上传完整音频；启用 LangSmith 后 tracing 数据也可能离开本机。API 启动不下载模型权重。
