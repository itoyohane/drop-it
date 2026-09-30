# 问题记录：无效意图误调用曲库工具

## 问题是什么

输入纯数字 `111` 时，偶发被 `search_library` 当作无参查询，返回前 20 首曲目；同一会话第一次未调用工具，第二次却调用了工具。

## 为什么会出现

规则路由没有匹配 `111`，只能交给 Ollama；无效输出虽降级为 `music_chat`，通用 Prompt 却仍声称拥有三个工具，导致主模型可能生成 `search_library` 调用，甚至把内部 DSML 协议当普通文本输出。

## 怎么解决

在 `graph.py` 的 intent routing 区新增显式 `music_chat` 规则：纯数字、符号和常见寒暄直接进入无工具路由；Ollama 无效分类也继续降级到该路由。Agent 保持统一执行流程；通用 Prompt 改为条件化描述工具，并禁止输出 DSML 等内部协议。

## 验证与复盘

回归测试覆盖 `111` 不调用 Ollama、无效 Ollama 输出降级以及 `music_chat` 不绑定工具，并检查 Prompt 不再声称始终拥有工具。关键实现见 `backend/src/agent/graph.py`，测试见 `backend/src/tests/test_intent_memory.py` 和 `backend/src/tests/test_services.py`。

# 问题记录：网页开发请求漏判与 MiniCPM5 分类截断

> 修复与验证日期：2026-09-30。

## 问题是什么

输入“帮我完成一个网页，内容是鹈鹕骑自行车”，音乐助手却输出网页代码。该请求应进入 `overstep`，而不是由主模型执行。

## 为什么会出现

开发网页的表达没有命中“写代码、编程”等规则，随后交给 MiniCPM5。修复前实测返回 `finish_reason=length`、128 个输出 tokens、269 字符的思考内容，但正文 `content` 为空。分类器没有拿到 JSON，便降级为 `music_chat`。这条路由只关闭业务工具，主模型仍可直接生成代码，因此禁用工具不能代替超纲识别。

## 怎么解决

最初修改 `intent.py`，合并新版架构后保留在 `backend/src/agent/graph.py` 的 intent routing 区：新增开发动作与网页、网站等对象的组合规则，直接拒绝网页开发，同时保留网页背景音乐检索等音乐任务；分类请求设置 `reasoning_effort=none`，使用限定五种标签的 JSON Schema，输出预算调整为 256 tokens；严格校验分类字段、置信度及截断状态。分类失败仍进入无工具 `music_chat`，但指导主模型只请求澄清，不执行原始任务。此项修复不改变 Agent 执行流程。

## 验证与复盘

完整后端回归：61 项通过、1 项音频集成测试按配置跳过；网页请求的 Agent 回归确认不调用主模型或工具。绕过规则直接调用真实 MiniCPM5，原请求返回完整 `overstep` JSON、16 个输出 tokens，无思考输出。另抽测网页变体、接歌及音乐知识问题均得到完整 JSON；速度筛选请求仍有语义误分类，格式修复不代表模型分类准确率已达标。测试见 `backend/src/tests/test_intent_memory.py`、`backend/src/tests/test_services.py`。

### 运行实例复盘

修复后仍出现代码，是因为网页实际连接 `.codex/worktrees/6c22/drop-it` 的旧服务，而修复位于 `Documents/ChatGPT/drop-it`；运行工作区仍使用 128 tokens 的旧分类器，实际请求记录为 `music_chat`。已将修复快进同步到运行工作区，由 Uvicorn 热重载生效，保留曲库和原会话。实际 SSE API 验证原句和“给我整一个有鹈鹕踩单车的网站”均进入 `overstep` 并拒答，无代码和工具事件；临时验证会话已删除。意图代码现已随架构合并移入 `backend/src/agent/graph.py`。交付验证应包含实际运行实例，不能只检查另一目录的源码和离线测试。
