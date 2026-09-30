# 从一次请求理解 DropIt 的 AI 应用架构

目标不是记住文件名，而是能回答四件事：自然语言如何变成可靠操作、检索证据从哪里来、失败如何收敛、为什么这样划分代码。动手操作见[教程](tutorial-run-and-observe.md)，精确字段见[参考](reference-api-tools-and-data.md)，讲解练习见[面试指南](howto-ai-application-interview.md)。返回[学习首页](README.md)。

## 1. 先建立最小心智模型

DropIt 是一个面向本地音乐曲库的受控 AI 工作流：搜索曲库、推荐相似歌曲、编排 DJ Set。它不是任意任务助手，也不是多个自治 Agent 协商的系统。

模型负责理解语言、提取参数和表达回答；代码负责范围、检索、编排、校验、保存与终止。**当前业务工具选择主要是规则硬路由，未命中才可选地交给 Ollama 分类**，并非聊天大模型自由决定下一步工具。

```text
离线准备：音频 → librosa 特征 → 描述 API → embedding API → SQLite + Chroma
在线请求：浏览器 → FastAPI → Agent 入口 → 单个 LangGraph → 业务函数
                                                   ↓
                                    结构化证据 → 回答模型 → SSE → 浏览器
```

“离线准备”表示不在聊天请求内做曲库分析，不表示不联网。默认描述和 embedding 是云端 API。

## 2. 按一次请求读源码，不按目录从头背

以“搜索 120–128 BPM、适合暖场的歌”为例：

1. [api.js](../frontend/src/api.js) POST 消息，浏览器通过 fetch 读取响应流；[chatStream.mjs](../frontend/src/chatStream.mjs)把可能被拆开的 SSE 帧拼好，再更新 UI。
2. [main.py](../backend/src/main.py) 的 `chat_stream` 检查会话属于路径中的项目、模型 Key 已配置，将 `stream_chat` 的事件编码为 SSE。
3. [agent.py](../backend/src/agent/agent.py) 的 `stream_chat` 建立/恢复 run，取得租约，保存用户消息并恢复历史；`_load_run` 在图执行前做 IntentRecognizer 分类。
4. [graph.py](../backend/src/agent/graph.py) 的 START 条件边选择 search_library。该节点调用 `extract_command` 得到 SearchCommand；精确 BPM 放 filters，暖场的语义要求放 query。
5. [retrieval.py](../backend/src/agent/retrieval.py) 先取当前范围的曲目并按 filters 筛选，再用 query 的描述向量排序。
6. Graph 的 respond 把音乐事实和结果组成证据，交给模型生成解释，产生回答片段；入口保存助手消息，发送 complete。

关键观察：自然语言不直接执行 SQL，回答模型不负责计算向量距离，项目范围也不由模型输出。可控来自这些边界，而不只是 prompt 写了“不要出错”。

## 3. 为什么是这些文件，而不是更多层

| 阅读入口 | 放在一起的职责 | 没有放进去的职责 |
| --- | --- | --- |
| [main.py](../backend/src/main.py) | 依赖装配、HTTP 路由、输入检查、SSE 包装、资源生命周期 | 曲目排序算法、图节点业务 |
| [agent.py](../backend/src/agent/agent.py) | 一轮对话的生命周期、消息/历史、租约心跳、图事件转接、摘要调用 | 检索和 Set 算法 |
| [graph.py](../backend/src/agent/graph.py) | 意图、typed command、State/Context、节点、条件边、证据响应 | 具体向量排序、完整 Set 算法 |
| [retrieval.py](../backend/src/agent/retrieval.py) | 范围内目录、参考歌解析、搜索、相似推荐、Set 候选获取 | Set 校验/修复/保存 |
| [set_planning.py](../backend/src/agent/set_planning.py) | 编排、约束、修复、保存边界、导出；共用音乐规则放这里 | 聊天流与模型参数提取 |
| [memory.py](../backend/src/agent/memory.py) | TTL 活动缓存、预算估算、摘要输入组织与截断策略 | 真正调用摘要模型 |
| [checkpoints.py](../backend/src/agent/checkpoints.py) | 状态序列化、类型恢复、步骤与下一节点映射 | 图拓扑、数据库租约 SQL |
| [workers/analyze_track.py](../backend/src/workers/analyze_track.py) | 后台任务提交、阶段执行、失败保留、启动恢复 | 音频特征算法和模型 HTTP 实现 |
| [librosa_analyzer.py](../backend/src/music/librosa_analyzer.py)、[text_models.py](../backend/src/music/text_models.py)、[indexer.py](../backend/src/music/indexer.py) | 特征分析、模型适配、描述索引分别有明确输入输出 | Agent 调度 |
| [sqlite.py](../backend/src/repositories/sqlite.py)、[chroma.py](../backend/src/repositories/chroma.py) | 关系数据/幂等事务、向量读写 | prompt 和用户语言理解 |

这是按“共同变化、独立验证”划分，不是每个函数一层。读一次搜索只需跟主链路，不必先读完租约和所有数据库迁移。graph.py 仍较长，是当前集中可见性的代价；以后只有出现稳定的独立职责才拆，不先造 Router/Service/Manager 多级转发。

旧的 tools.py、intent.py、services 层已不作为当前实现。`DropItToolRegistry` 名字保留，但现在是检索业务入口，不是 LangChain Tool 注册器；三个操作名也不代表它同时负责所有 Set 生命周期。

## 4. LangGraph：三个业务能力为什么有十一种节点

业务能力是用户能做什么，节点是可观察/可恢复的步骤。生成 Set 要经历召回、规划、校验、修复、保存，因此不能把“工具数量”和“节点数量”混为一谈。当前只有一个直接编译的 StateGraph，没有必要为三个工具再建三个子图。

新请求的主路径如下；恢复请求可直接进入 checkpoint 对应的下一节点。

```text
START
 ├─ search_library ──────────────────────────→ respond → END
 ├─ resolve_reference → find_similar_tracks ─→ respond → END
 ├─ retrieve_candidates → plan_set → validate_set
 │                                    ├─ 不通过且可修复 → repair_set ─┐
 │                                    └─ 通过/结束修复 → persist_set │
 │                                                        ↓         │
 │                                                     respond      │
 │                                                        ↓         │
 │                                                       END        │
 │                                validate_set ←─────────────────────┘
 ├─ respond_chat → END
 └─ reject_response → END
```

源码 `_after_validation` 在修复耗尽后仍路由到 persist_set；但此时已有 constraint_conflict，persist_set 立即跳过写入。这是实际控制边，不是“失败仍保存”。入口也会把错误结果的 playlist 清空。

### State、Context、Command 各装什么

- Command 是模型提取的请求参数，如 query、BPM、时长；Pydantic 禁止模型加 project_id 等字段。
- State 是图中传递的业务进度：候选、结果、校验、修复次数、错误、回答，可保存为 checkpoint。
- Context 是服务端为本轮注入的范围和依赖：项目、会话、Store、Registry、模型工厂、租约 owner/token。frozen dataclass 防止意外字段赋值，但安全还依赖查询范围、业务校验和数据库写入检查。

项目/会话来自服务端路径，不说明已经有用户权限系统：当前没有登录和租户鉴权，知道资源 ID 的客户端没有用户级授权边界。

### 为什么节点不加 @tool，为什么会有相近函数名

LangGraph 使用 `add_node(name, callable)` 注册接收 State/Runtime、返回状态增量的函数。LangChain 的 `@tool` 是供模型工具调用的 schema 包装，当前没有自由工具循环，不需要这层包装。

图中的 `resolve_reference` 是节点名，对应函数 `extract_and_resolve_reference`，负责提取命令、错误转换和 checkpoint；Registry 的 `resolve_reference` 才执行范围内歌曲精确匹配。这不是两份同样的匹配算法。类似地，图节点 plan_set/persist_set 编排执行，set_planning.py 中同名业务函数计算或保存，导入时用别名区分。判断重复要看职责和函数体，不只看名字。

## 5. 音乐 RAG：先准备证据，再回答

普通文档 RAG 常按文档切块；这里一个歌曲描述就是一个检索单元，没有先加文本知识库和 chunking 层。

### 入库链路

`JobRunner._analyze` 对本地音频做 librosa 分析，得到 BPM、调性/Camelot、RMS 能量及谱特征等；描述模型依据这些数值生成文本；MusicIndexer 将描述送入文本 embedding；向量写 Chroma，事实、描述与 ready/model_key 写 SQLite。

当前不是把音频波形送入多模态 embedding，也不是模型真正“听懂整首歌”。BPM/调性估计有误差，energy 是数值代理；描述 prompt 禁止凭空补流派、乐器、歌手与情绪，但 prompt 不能保证绝对无幻觉。上游证据弱，检索相关性也会受限。

失败分阶段保留：特征失败不伪造数值；描述失败可保留已分析特征；embedding 失败可保留描述，再重试索引。模型版本参与 ready 判定，避免新查询向量与旧模型向量混排。显式指定 track_ids 强制分析与普通“重试未 ready”含义不同。

### 查询链路

1. SQLite 先确定项目范围和元数据过滤；全局查询才允许汇总范围。
2. 非空语义 query 用同一 embedding model_key 编码；只读取当前模型、状态正确的候选向量。
3. 单位化向量做点积，即余弦相似度；缺索引、非法向量或维度不一致会显式报错，不返回随机占位歌曲。
4. 把匹配曲目的事实给回答模型；`MusicMatch.context()` 不含本地 path，但包含标题、歌手、描述等。

相似歌曲直接用参考歌向量，不必再次调用 query embedding。当前重排公式是：

```text
score = 0.80 × 描述余弦相似度
      + 0.10 × BPM 接近度
      + 0.05 × Camelot 相容性
      + 0.05 × energy 接近度
```

这是手工规则融合，不是训练出来的推荐器，score 也不是概率。Chroma 负责持久化，当前 NumPy 对范围内向量全量排序，适合小曲库但不是大规模 ANN 检索实现。SQLite 与 Chroma 没有跨库原子事务，写向量后才标 ready；状态检查、幂等重写和重建是当前一致性策略，不能称为分布式强一致。

## 6. Set：确定性规划和校验，而不是让模型自我夸奖

Graph 获取候选后，set_planning.py 按能量目标、BPM 和 Camelot 规则贪心排序，依据原曲时长选择曲目。随后 SetValidator 输出结构化问题，SetRepairer 确定性增删/重排，最多两轮再校验；无可行结果就返回冲突，不保存草案冒充成功。

模型决定不了“校验通过”。这可以称为校验驱动的有限修复循环，不应该说成多 Agent Critic、LLM Reflection 或全局最优约束求解。两轮只限制工作量，不保证所有可行集合都会被贪心算法找到。没有混音重叠时长、音频拼接或现场听感验证。

独立 Set HTTP 入口复用规划/校验/保存函数，方便对比不使用聊天模型时业务是否成立；其候选获取与聊天分支并不完全一致。手动调序/approve 是另一路 API，当前不做全部约束复核，这也是演示时要承认的边界。

## 7. 恢复、幂等、并发是三个问题

run_id 标识“一次逻辑请求”；checkpoint 保存已完成步骤的 State；租约 owner/fencing token 限制“现在谁有权写”；唯一键和稳定歌单 ID 限制“不能重复生成持久结果”。四者配合，少一个都可能产生重复或过期写入。

设想服务在歌单写入后、回答结束前退出：同一 run_id 重试时读取持久步骤和歌单，继续回答，而不是从头重复保存。如果旧执行者在租约失效后才返回，数据库检查 token 和有效租约，拒绝它写消息、步骤或歌单。阻塞业务通过线程执行，避免占住事件循环使心跳无法续租。

但当前不能保证端到端 exactly-once：外部模型调用在 checkpoint 写入前崩溃可能重复计费，部分未持久化计算会重跑；SSE token 也不是可逐条回放的持久日志。已持久化结果的幂等，不等于每个计算和每个网络副作用只发生一次。浏览器还没接 run_id 重试，恢复主要通过显式 API 调用和测试验证。

Agent 恢复由客户端再提交同一请求触发；后台分析 Job 的启动恢复则由 lifespan 中 `jobs.start()` 处理。二者不是同一套自动后台续跑机制。SQLite/线程执行器适合单机学习与演示，不能直接推出已支持多机任务调度。

## 8. 三种“记忆”不要混称

SQLite 消息是持久历史；ShortTermMemory 是按项目/会话隔离、有容量和 TTL 的活动缓存；checkpoint 是一次 run 的执行状态，不是用户长期偏好。

ContextCompressor 用 UTF-8 字节量估算 token，默认到应用预算的 80% 触发。memory.py 负责拆分和预算，agent.py 的 `_compact_context` 调模型摘要旧消息并保留最近六条；失败就裁掉旧消息。摘要放在标注为事实背景的 assistant 消息里，不提升成 system 指令。

这里没有精确 tokenizer 计量、摘要保真证明或覆盖所有检索证据的严格上下文预算；也没有长期偏好学习。SQLite 原消息不会因活动上下文压缩而变成“只有摘要”。

## 9. 安全和质量：能说什么，不能说什么

音频特征在本地计算，默认不把波形上传给模型；但特征会传描述服务，文本会传 embedding 服务，历史和音乐证据会传聊天服务；启用 LangSmith 还会产生外部 trace。不能把“本地曲库”说成“所有数据离线”。

Pydantic schema、范围过滤、约束校验、路径剔除和拒答 prompt 是不同层的防护，不是完整 prompt injection 防线。没有用户鉴权、资源配额隔离、模型审计和公网部署保障。CORS、安全响应头不是授权系统。

测试证明受控契约和失败行为，不自动证明音乐推荐好听。[七轮历史评测](quality-eval-7-rounds.md)的自检索 Hit@3 和事实逐字段一致验证了索引闭环，不等价于新用户自然语言相关性或最终生成回答完全真实。质量提升需要独立查询标注、听感/可接性评价、延迟和成本统计。

## 10. 理解检查

不看正文，尝试解释：

1. music_chat 为什么不需要检索？查询“120 BPM”为什么可以不生成向量？
2. model_key 与项目范围解决的是哪两个不同问题？
3. persist_set 为什么不能只相信模型或先前 validator？
4. 同一 run_id 重试、另一个 run_id 发送相同文本，是否是同一业务请求？
5. 为什么一个固定分支工作流仍用了 LangGraph？不用它时要自己维护哪些状态？

答案应能指向源码和测试，并说出边界。下一步按[面试指南](howto-ai-application-interview.md)练一次带证据的讲解。
