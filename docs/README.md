# DropIt 学习路线：从跑通到能解释设计

这套文档面向 AI 应用开发学习与面试准备。目标不是背框架术语，而是能沿源码说明：输入怎样变成业务操作、RAG 如何提供证据、失败如何处理，以及当前实现的局限。

当前实现：React + FastAPI；一个显式 LangGraph；三个业务能力；librosa 特征、云端歌曲描述与文本 embedding；SQLite 事实/运行状态 + Chroma 向量。不是多 Agent、自由 ReAct、全离线音乐理解或自动混音系统。

## 先按这条路线学

| 顺序 | 文档 | 完成标准 |
| --- | --- | --- |
| 1 | [动手教程：跑通并观察](tutorial-run-and-observe.md) | 确认前后端、区分分析/描述/索引，观察搜索与 Set 的成功或失败 |
| 2 | [架构解释：跟一次请求读源码](explanation-agent-architecture.md) | 讲清模型/代码、State/Context/Command、检索与修复、恢复边界 |
| 3 | [面试指南：讲解、演示和追问](howto-ai-application-interview.md) | 90 秒介绍、五分钟演示，每个结论能指向函数和测试 |
| 随时查 | [接口、命令与数据参考](reference-api-tools-and-data.md) | 精确查询 API、SSE、状态、默认值、配置和验证入口 |

教程告诉你“做什么并观察什么”，架构页解释“为什么”，参考页给精确契约，面试页帮助组织证据；不用把四篇从头背一遍。

## 阅读源码也分两轮

第一轮只跟业务主线：

```text
frontend/src/api.js → backend/src/main.py
→ agent/agent.py → agent/graph.py
→ agent/retrieval.py 或 agent/set_planning.py → 回答与 SSE
```

第二轮再理解可靠性：

- 曲库准备：[后台任务](../backend/src/workers/analyze_track.py) → [音频特征](../backend/src/music/librosa_analyzer.py) → [模型适配](../backend/src/music/text_models.py) → [索引](../backend/src/music/indexer.py)。
- 数据与副作用：[SQLite](../backend/src/repositories/sqlite.py)、[Chroma](../backend/src/repositories/chroma.py)、[数据库迁移](../backend/src/migrations)。
- 历史与恢复：[短期记忆](../backend/src/agent/memory.py)、[checkpoint](../backend/src/agent/checkpoints.py)、[恢复/并发测试](../backend/src/tests/test_p0_set_and_checkpoints.py)。

源码目录已是 `backend/src`，[backend 包入口](../backend/__init__.py)兼容旧的 `backend.*` 导入。旧 tools.py、intent.py 和 services 分层不是当前阅读路线；遇到旧文档或历史评测中的名字，要先核对版本。

## 学完应能独立回答

1. 三个业务能力为什么不等于三个节点或三个子图？为什么不需要 @tool？
2. 哪些搜索只查元数据，哪些才调用 embedding？向量是否直接存在 SQLite？
3. 模型负责什么，业务范围和写入权限由谁决定？
4. 无效 Set 为什么不能保存？修复两轮保证了什么、没保证什么？
5. checkpoint、run_id 幂等、租约 fencing 分别解决什么？为什么仍不能保证外部调用 exactly-once？
6. 当前质量指标能证明什么？为什么本地曲库不等于所有数据离线？

先自己解释，再查架构页和对应测试。答不清时回到一条实际请求，不急着引入新组件。

## 当前能力与明确边界

已经实现导入/分析/描述/索引后台任务、项目与全局查询、三条受控业务路径、Set 校验与最多两轮修复、持久步骤与结果去重、SSE 和短期上下文管理。

尚未具备用户/租户鉴权、前端自动 run_id 重试、长期偏好学习、分布式任务队列、自动混音、开放式推荐质量保证。项目范围检查不能替代用户授权，应用不应直接暴露到公网。详细限制见架构页，不把后续方案当成当前实现。

## 历史材料：用于复盘，不替代当前说明

- [P0 To-Be-Solved List](p0%20lists.md)：问题、验收目标和后续待办；保留原始问题演进。
- [七轮质量评测](quality-eval-7-rounds.md)及[逐用例 JSON](quality-eval-7-rounds.json)：2026-09-12 的历史快照，不是本次文档重写重新跑出的结果；旧 Tool 参数和指标要结合当时版本解读。
- [工具路由历史问题](question-111-tool-routing.md)：一条已记录的问题，不替代当前 graph.py 的命令契约。
- [前端设计约定](../DESIGN.md)：产品界面背景；应用源码与接口仍以当前代码为准。

重写正文不覆盖这些历史记录。质量复测需显式选择输出路径、确认真实 API 成本；默认单测使用假模型，构建成功也不能替代云端验收。

## 如何维护这些文档

变更 API/字段/配置先更新参考页；变更控制流和职责更新架构页；变更启动和 UI 操作更新教程；变更可证明的能力更新面试页。链接、API 路径、Settings 名称、checkpoint 步骤与教程纯 Python 实验由[文档测试](../backend/src/tests/test_learning_docs.py)检查，不能替代人工审查业务语义。

返回[项目首页](../README.md)。
