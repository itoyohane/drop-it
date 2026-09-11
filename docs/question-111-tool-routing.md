# 问题记录：无效意图误调用曲库工具

## 问题是什么

输入纯数字 `111` 时，偶发被 `search_library` 当作无参查询，返回前 20 首曲目；同一会话第一次未调用工具，第二次却调用了工具。

## 为什么会出现

规则路由没有匹配 `111`，Ollama 未返回有效 JSON 后降级为 `music_chat`。但 `music_chat` 只是写入 Agent Prompt 的提示，Agent 仍暴露全部三个业务工具；主模型受历史上下文和采样参数影响，可能自行调用默认的 `search_library(limit=20)`。

## 怎么解决

在 `intent.py` 的路由结果中明确工具权限：`music_chat`（包括无效分类的降级结果）改用独立的无工具 Prompt，并直接调用主模型；只有搜索、相似歌曲和 DJ Set 意图才进入工具 Agent。返回前同时拦截意外生成的 DSML/工具协议文本。

## 验证与复盘

新增回归测试覆盖 Ollama 无效分类输入 `111` 和 DSML 文本泄漏，确认主模型仍能回答且 `tool_events=[]`。关键实现见 `backend/agent/intent.py` 与 `backend/agent/agent.py`，测试见 `backend/tests/test_services.py`。
