# Agent 架构、边界与取舍

DropIt 把用户对音乐的模糊描述转为当前曲库内可核实的歌曲、相似结果和 Set 草案。
模型负责理解与调用工具；音频分析、过滤、排序和保存由本地代码执行。

## 两条执行链路

导入链路不经过聊天模型：上传音频 → Mutagen 标签与 SHA-256 去重 → SQLite 曲目及任务 →
Essentia 分析并保存属性 → CLAP 编码并保存音频向量。

对话链路统一经过 `LangChain create_agent`：读取历史 → 模型按需调用三个工具 →
工具检索数据库证据或保存 Set → 结果回到模型 → SSE 输出与消息持久化。
没有按关键词分流到“直接模型调用”的旁路，也没有每轮预先塞入整库文本的步骤。

## 目录与依赖方向

```text
backend/
  main.py, config.py, models.py, audio.py
  agent/
    agent.py, tools.py, prompts.py, retriever.py
  music/
    essentia_analyzer.py, clap_embedder.py, indexer.py, download_models.py
  services/
    catalog_service.py, similarity_service.py, set_planner.py, analysis_service.py
  repositories/
    tracks.py, analysis.py, embeddings.py, sqlite.py
  workers/
    analyze_track.py
  migrations/
  tests/
```

依赖从入口向内流动：`main → agent/worker → service → repository/music`。
Agent 不写 SQL，Worker 不实现声学算法，音乐适配器不读取项目权限。

| 模块 | 职责 |
| --- | --- |
| [main.py](../backend/main.py) | FastAPI 入口、依赖装配、HTTP/SSE、导入与管理接口 |
| [agent/agent.py](../backend/agent/agent.py) | `create_agent`、最多 40 条历史、工具事件转换，递归上限 16 |
| [agent/prompts.py](../backend/agent/prompts.py) | 唯一系统提示词，集中声明工具用法和事实边界 |
| [agent/tools.py](../backend/agent/tools.py) | 三个工具的 LangChain schema、作用域绑定和结果包装 |
| [agent/retriever.py](../backend/agent/retriever.py) | Agent 侧 RAG 门面，组合 catalog 与 similarity service |
| [music/essentia_analyzer.py](../backend/music/essentia_analyzer.py) | BPM、节拍、调性、Camelot、RMS 能量代理值；当前不生成学习型 tags |
| [music/clap_embedder.py](../backend/music/clap_embedder.py) | 懒加载 CLAP，生成归一化 audio/text embedding |
| [music/indexer.py](../backend/music/indexer.py) | 校验一首歌的向量维度并通过 Repository 持久化 |
| [services/catalog_service.py](../backend/services/catalog_service.py) | 项目/全局曲库范围和结构化过滤 |
| [services/similarity_service.py](../backend/services/similarity_service.py) | CLAP 精确近邻与 Essentia 属性重排 |
| [services/set_planner.py](../backend/services/set_planner.py) | 确定性 Set 规划、规则报告和导出 |
| [services/analysis_service.py](../backend/services/analysis_service.py) | 单曲 Essentia → CLAP 分阶段编排与错误隔离 |
| [repositories/tracks.py](../backend/repositories/tracks.py) | 曲目、项目成员关系的 repository contract |
| [repositories/analysis.py](../backend/repositories/analysis.py) | 分析状态与持久化 Job 的 repository contract |
| [repositories/embeddings.py](../backend/repositories/embeddings.py) | 版本化音乐向量的 repository contract |
| [repositories/sqlite.py](../backend/repositories/sqlite.py) | 三个 contract 的线程安全 SQLite 实现，以及项目、对话、来源和 Playlist 存取 |
| [workers/analyze_track.py](../backend/workers/analyze_track.py) | 单线程消费持久化任务、更新进度、恢复中断任务 |

Repository contract 使用 Python `Protocol`，服务层只依赖所需能力；`SqliteRepository` 通过结构类型
实现这些 contract。这样可以在测试中注入替代实现，也能以后拆分数据库，而不需要让每层都知道 SQL。

## 为什么 Essentia 与 CLAP 都保留

Essentia 提供可用于数值过滤的音乐属性。当前使用节奏、调性和 RMS 算法，没有加载流派、
情绪或乐器分类器。能量是 `clip((RMS dBFS + 38) / 30, 0, 1)`，不是情绪或舞曲强度标签。
BPM 和调性置信度保存原始输出，不把它们解释为校准过的概率。

CLAP 把声音与文字放进同一个向量空间，用于“黑暗、工业感的电子音乐”等描述检索，
也用于参考歌曲的相似检索。它不替代 BPM/调性分析，更不负责生成歌名或事实标签。
中文声音描述由 Agent 转成简短英文；标题、艺人和数值要求走独立过滤参数。

当前音频策略取歌曲 20%、50%、80% 附近最多三个十秒片段，短曲取单段。
每段向量归一化，平均后再归一化，最终保存 512 维向量。抽样节省推理量，但可能漏掉
歌曲局部变化；检索质量需要真实曲库与人工偏好评测，不能从向量维度推断。

## 音乐 RAG 如何工作

数据库只持久化音频向量。文本查询向量在内存生成，不建立文档切块、文本向量表或 Chroma 集合。
检索先确定当前项目歌曲，再应用属性过滤，与当前模型的音频向量做 NumPy 精确余弦排序。
返回的 track_id、标题、艺人、BPM、调性、能量和分数构成模型回答的证据。

`search_library(query="")` 直接读取属性，不加载 CLAP。非空 query 进行文本到音频检索。
部分候选未建立索引时只返回已就绪的部分；候选存在但全部未就绪时明确报错，不生成假向量。

`find_similar_tracks` 排除参考歌曲本身，再按以下权重重排：音频余弦 0.8、BPM 接近度 0.1、
Camelot 兼容性 0.05、能量接近度 0.05。分数只是排序信号，不是“相似概率”。
精确检索的开销随曲库线性增加；当前以简单单机部署为目标，没有近似索引服务。

## Set 工具为什么不让模型直接写歌曲列表

`generate_dj_set` 可通过英文 `style_query` 召回最多 200 首候选，再交给规则规划器。
也可用此前检索返回的 `track_ids` 限定候选。无风格描述时直接使用当前项目曲库，
只选择已分析且满足 BPM 范围的曲目。缺少风格索引时不会悄悄绕过风格需求。

规划器按能量曲线、节奏与调性规则排序，保存 Playlist 并生成报告。它是确定性启发式算法，
不是全局最优求解，也不是独立的 Critic Agent。报告中的 Library、Set Planner、Transition Rules
是规则阶段名称。时长按整曲求和，不计算实际混音重叠；候选不足可能达不到目标时长。

## 数据生命周期与隔离

音频属性与向量状态分离。CLAP 失败仍保留 Essentia 结果，允许无风格语义需求的规则排 Set。
普通补分析只补缺失阶段，显式指定曲目则强制重分析。模型名称、revision、片段策略一起组成
model key，旧版本向量不会混入新模型检索；升级模型后需要重新分析。

元数据修改后检索读取最新数据库值，不重算音频向量。移除项目曲目会撤销项目关联，
不是删除全局曲目、原始音频或其他项目的关联。相同歌曲可能由多个项目共享，编辑属性会影响共享记录。
迁移文件 001–006 必须保留：启动时按顺序升级旧库；不能只留下最新迁移。

工具的 `project_id` 由服务端闭包绑定，不允许模型指定。项目对话只检索项目歌曲；
`global-chat` 是当前单用户的全局曲库视图，可以搜索与找相似，但保存 Set 需进入具体项目。
这不是多租户认证，HTTP API 当前没有登录或访问令牌校验；CORS 也不是鉴权。

## 本地运行不等于所有数据都离线

Essentia 和 CLAP 本地解码、推理，不向音频云服务上传歌曲。聊天需要服务端 DeepSeek 配置，
模型会收到对话及检索到的曲目信息。可选 LangSmith tracing 也可能发送对话和工具数据，默认关闭。
本地 API 上传仍会在服务端保存音频副本；不要把“本地分析”理解为“不复制任何文件”。

后台只有一个分析线程和一个 API 进程。SQLite 持久化任务允许重启恢复，但不是分布式队列。
模型权重显式下载，默认仅使用本地缓存。缺少依赖或权重会失败，不使用随机标签或哈希向量兜底。

## 发布与许可证

Essentia 依赖声明为 AGPL-3.0-only。分发或提供基于该程序的网络服务前，应核对 AGPL 对当前
部署方式、修改和完整对应源码提供方式的要求。CLAP 代码、Hugging Face 模型仓库及具体权重也要
分别核对许可证和商用限制。当前仓库没有许可证文件，本次重构也没有替项目选择许可证；
在许可证决定完成前，不要把 Docker 镜像或模型权重当作可直接商用分发的产物。

下一步操作见[运行教程](tutorial-run-and-observe.md)，参数见[接口参考](reference-api-tools-and-data.md)。
