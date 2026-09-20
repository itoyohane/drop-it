# 问题记录：无效意图误调用曲库工具

## 问题是什么

输入纯数字 `111` 时，偶发被 `search_library` 当作无参查询，返回前 20 首曲目；同一会话第一次未调用工具，第二次却调用了工具。

## 为什么会出现

规则路由没有匹配 `111`，只能交给 Ollama；无效输出虽降级为 `music_chat`，通用 Prompt 却仍声称拥有三个工具，导致主模型可能生成 `search_library` 调用，甚至把内部 DSML 协议当普通文本输出。

## 怎么解决

在 `intent.py` 新增显式 `music_chat` 规则：纯数字、符号和常见寒暄直接进入无工具路由；Ollama 无效分类也继续降级到该路由。Agent 保持统一执行流程，仅按路由结果绑定工具；通用 Prompt 改为条件化描述工具，并禁止输出 DSML 等内部协议。

## 验证与复盘

回归测试覆盖 `111` 不调用 Ollama、无效 Ollama 输出降级以及 `music_chat` 不绑定工具，并检查 Prompt 不再声称始终拥有工具。关键实现见 `backend/src/agent/intent.py`，测试见 `backend/src/tests/test_intent_memory.py` 和 `backend/src/tests/test_services.py`。
