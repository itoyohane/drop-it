# Agent 架构、边界与取舍

DropIt 把自然语言音乐需求转换为当前曲库内可核实的歌曲、相似结果和 DJ Set。聊天模型负责理解与工具选择；音频分析、检索、排序和保存由本地代码执行。

## 两条执行链路

导入链路不经过聊天模型：上传音频 → Mutagen 标签与 SHA-256 去重 → SQLite Job → librosa 提取 BPM、节拍、调性、Camelot、能量和频谱特征 → FLAN-T5-small 把测量特征整理成歌曲文本描述 → MiniLM 编码描述并保存向量。

对话链路：`intend.py` 给出轻量意图提示 → `memory.py` 提供最近 12 条、30 分钟 TTL 的短期记忆 → LangChain Agent 按需调用三个工具 → 工具读取数据库证据或保存 Set → SSE 输出并持久化完整消息。

## 目录与依赖方向

```text
backend/
  main.py, config.py, models.py, audio.py
  agent/
    agent.py, intend.py, memory.py, prompts.py, tools.py
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
| [main.py](../backend/main.py) | FastAPI、依赖装配、HTTP/SSE、导入和管理接口 |
| [agent/agent.py](../backend/agent/agent.py) | `create_agent`、意图与记忆接入、工具事件转换 |
| [agent/intend.py](../backend/agent/intend.py) | 搜歌、找相似、排 Set、普通音乐问答的轻量预识别 |
| [agent/memory.py](../backend/agent/memory.py) | 会话隔离、容量有界、自动过期的进程内短期记忆 |
| [agent/tools.py](../backend/agent/tools.py) | 三个 Tool 及其完整业务实现 |
| [music/librosa_analyzer.py](../backend/music/librosa_analyzer.py) | DJ 数值特征和描述模型输入特征 |
| [music/text_models.py](../backend/music/text_models.py) | 小模型歌曲描述与文本 embedding |
| [workers/analyze_track.py](../backend/workers/analyze_track.py) | 持久化任务、阶段重试和错误隔离 |
| [repositories/sqlite.py](../backend/repositories/sqlite.py) | SQLite 数据与版本化描述向量 |

## 文本音乐 RAG

每首歌先得到一段可审计的英文描述。描述始终包含 BPM、调性、Camelot、能量、频谱中心和 onset strength 等测量事实；小模型只负责把这些特征组织成简短检索文本，不应杜撰流派、演唱者或乐器。描述和生成模型版本保存在 Track 上。

MiniLM 把歌曲描述与用户查询放入同一文本向量空间。`search_library(query="")` 只做 SQL 元数据过滤，不加载模型；非空 query 对描述向量做 NumPy 精确余弦排序。`find_similar_tracks` 使用参考歌的描述向量，并按描述余弦 0.8、BPM 0.1、Camelot 0.05、能量 0.05 重排。所有分数只是当前候选集中的排序信号，不是概率。

该方案比 CLAP 更轻且描述可观察，但它只能检索 librosa 特征和小模型文本能够表达的属性。真实流派、乐器、歌词或情绪若未被可靠模型测量，不应当作事实。大曲库后续可换近似向量索引，当前线性搜索优先简单可测。

## 状态、升级与隔离

分析、描述和 embedding 分阶段处理。文本模型失败仍保留 librosa 数值结果；重试时只补缺失阶段。模型名、revision 和描述策略进入 model key，旧模型向量不会混入新检索。迁移 007 会清空旧 CLAP 向量并把 Essentia 结果标为待重分析。

工具的 `project_id` 由服务端闭包绑定，模型不能自行选择范围。项目对话只访问项目曲目；全局曲库可以查歌和找相似，但不能保存 Set。这是单用户范围约束，不是多租户认证。

短期记忆是进程内缓存，按项目与会话隔离，删除会话时立即清除；过期或进程重启后会从 SQLite 最近消息恢复。SQLite 仍是完整对话记录的事实来源。

librosa、FLAN-T5 和 MiniLM 在本地运行，不上传完整音频。DeepSeek 会收到对话和 Tool 返回的曲目上下文；启用 LangSmith 后 tracing 数据也可能离开本机。模型权重仅由显式下载命令获取，API 启动默认只读本地缓存。
