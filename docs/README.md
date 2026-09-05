# DropIt 文档

以当前工作区的后端代码为准：一个 LangChain Agent，三个音乐工具，Essentia 分析，CLAP 检索，SQLite 持久化。

## 阅读顺序

1. [运行并观察一次工作流](tutorial-run-and-observe.md)：从安装到导入、查询和排 Set。
2. [架构、边界与取舍](explanation-agent-architecture.md)：理解模块关系与音乐 RAG。
3. [API、Tool 与数据参考](reference-api-tools-and-data.md)：查接口、状态、参数和环境配置。
4. [项目演示与面试讲解](howto-ai-application-interview.md)：基于代码证据介绍项目。

## 当前范围

- 音频导入、音乐属性分析、音频向量入库与后台任务恢复。
- 对话调用 `search_library`、`find_similar_tracks`、`generate_dj_set`。
- 项目曲库与全局曲库查询，Set 保存、调序、确认和 JSON/CSV/M3U 导出。
- 不包含普通文本知识库、外部曲库搜索、自动混音或独立的 Curator/Planner/Critic 多 Agent。
- 当前没有用户登录或多租户鉴权。项目范围限制不能替代用户隔离，不应直接暴露到公网。

旧的 MVP PRD 已由这些文档取代；原有前端设计参考 [DESIGN.md](../DESIGN.md) 保留。
`backend/README.md` 的运行、架构与测试说明已合并到此目录，避免维护两份相同说明。

## 工作区与验证边界

本次重构和自动化验证集中在后端。前端没有随目录分层一起重构；前端构建通过也不能代替
Agent、音乐检索和音频分析链路的后端测试。

自动化测试覆盖业务与模型适配器契约；真实 Essentia + 预训练 CLAP 测试需额外启用。
本次重构尚未在当前 Windows 环境完成真实模型端到端验收，也未建立检索质量基准。
测试命令和前提见[运行教程](tutorial-run-and-observe.md)。
