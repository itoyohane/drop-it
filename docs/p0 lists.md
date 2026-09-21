P0 To-Be-Solved List
1. Agent 执行过程不显式
- 已完成（P0 item 1）
完成前问题
此前主要依赖 create_agent 让模型自行选择工具。
要改什么
修改：
backend/src/agent/agent.py
新增：
backend/src/agent/state.py
backend/src/agent/commands.py
backend/src/agent/graph.py
改成什么
使用直接编译的 LangGraph StateGraph 构建受控执行图。IntentRecognizer 先做硬路由，图使用条件边选择以下分支：
search_library → respond → END
resolve_reference → find_similar_tracks → respond → END
retrieve_candidates → plan_set → persist_set → respond → END
respond_chat → END
reject_response → END

模型只负责路由专属 typed command 提取和最终自然语言回答；检索、参考歌解析、规划、持久化和终止由确定性节点负责。
定义显式 AgentState：
class AgentState(TypedDict):
    run_id: str
    route: AgentRoute
    command: typed route command | None
    history: list[dict[str, str]]
    candidate_track_ids: list[str]
    playlist: Playlist | None
    final_response: str
    error_code: str | None

项目和会话范围、Store 与 Registry 依赖位于不可变的 server-only AgentRuntimeContext，不进入 command schema。
为什么改
直接对应 JD 中的：
- 任务规划与调度
- 执行引擎
- Planning
- Tool Use
- LangGraph 项目实践
- 复杂任务和长链路任务
它还能让执行过程可测试、可追踪、有终止条件，不依赖模型自由发挥。
完成标准
- 每类任务都有明确 Graph 分支。
- 所有分支都有终止节点。
- 节点可以独立测试。
- 模型不能任意改变项目范围。
- 简单搜索不经过多余的 Agent 循环。
预计时间
实现日期：2026-09-20。
2. Set 校验失败后仍然会返回结果
- 已完成（P0 item 2）
完成前问题
[tools.py (line 205)](../backend/src/agent/tools.py:205) 当时已经计算：
- BPM 大跳跃
- Camelot 不兼容
- 时长误差
但即使 Transition Rules 状态为 failed，Playlist 仍会被返回和保存。
当时流程是：
生成 Set → 记录失败信息 → 仍然保存
要改什么
从 tools.py 提取校验逻辑，新增：
backend/src/agent/set_validation.py
backend/src/agent/set_repair.py
修改：
backend/src/agent/tools.py
backend/src/repositories/sqlite.py
改成什么
改成校验驱动的受控 Reflection：
生成 Set
  ↓
SetValidator
  ├─ 通过 → 保存
  └─ 失败 → SetRepairer
                ↓
             再次校验
                ↓
        最多修复两轮
校验器必须检查：
track_membership
duplicate_tracks
bpm_range
duration_tolerance
bpm_transition
camelot_compatibility
energy_curve
required_tracks
返回结构化错误：
{
  "valid": false,
  "issues": [
    {
      "code": "camelot_break",
      "position": 4,
      "message": "第 4 首到第 5 首调性不兼容"
    }
  ]
}
无法修复时：
- 不保存 Playlist。
- 返回 constraint_conflict。
- 告诉用户需要放宽哪个条件。
为什么改
直接对应 JD 中的：
- Reflection
- 任务执行失败分析
- 工具调用和效果异常改进
- 准确性和稳定性保障
这比通用 ReAct 的“让模型反思”更适合强约束音乐场景。
完成标准
- 无效 Set 保存数为 0。
- 修复循环最多两轮。
- 每个失败都有错误代码。
- 相同输入和曲库产生确定性结果。
- 无法满足时不静默放宽用户限制。
预计时间
实现日期：2026-09-21。
3. 长链路任务不能从中间恢复
- 已完成（P0 item 3）
完成前问题
音频分析任务可以在服务重启后重新排队，但 Agent 对话任务当时没有节点级 Checkpoint。
如果 Set 生成在最后一步失败，可能需要重新：
- 解析请求
- 检索候选
- 调用模型
- 执行规划
要改什么
在 LangGraph 执行链路中增加 Checkpoint。
新增或扩展：
backend/src/agent/checkpoints.py
backend/src/repositories/sqlite.py
backend/src/migrations/009_agent_runs.sql
backend/src/migrations/010_agent_run_claims.sql
改成什么
保存节点状态：
command_parsed
candidates_retrieved
set_planned
set_validated
set_repaired
playlist_persisted
completed
每个节点保证幂等：
相同 run_id + 相同 step
→ 已完成则复用结果
→ 未完成才重新执行
服务重启后：
读取最后成功节点 → 从下一节点继续
为什么改
直接对应 JD 中的：
- 任务调度
- 执行引擎
- 复杂任务
- 长链路任务
- 稳定性
- 使用成本
它能避免重复模型调用和重复写入 Playlist。
完成标准
- 每次任务都有 run_id。
- 服务中断后能够继续执行。
- 同一节点不会重复产生写操作。
- 重试不会生成多个 Playlist。
- Checkpoint 数据能够用于调试。
预计时间
实现日期：2026-09-21。
4. 评测没有覆盖完整任务、效率和成本
- 待解决
现在是什么
现有评测已经覆盖：
- 意图准确率
- 工具成功率
- RAG 自检索 Hit@3
- 数据真实性
但还没有完整覆盖：
- 端到端任务完成率
- 工具参数正确率
- 多步调用顺序
- Set 强约束满足率
- 多次运行稳定性
- P50/P95 延迟
- Token 与成本
- 失败发生在哪个节点
要改什么
新增：
backend/evals/cases/agent_tasks.jsonl
backend/evals/run_agent_eval.py
backend/src/tests/test_agent_eval.py
扩展：
backend/evals/run_quality_eval.py
改成什么
建立 60 条真实任务：
15 条搜索
10 条相似推荐
20 条 Set 生成
10 条多轮上下文
5 条越权和异常输入
增加指标：
Command Accuracy
Argument Accuracy
Tool Sequence Accuracy
Task Completion Rate
Set Constraint Pass Rate
Unauthorized Action Block Rate
Stability
P50/P95 Latency
Token Cost
Repair Success Rate
每次 Agent Run 记录：
run_id
command
graph_steps
tool_calls
latency_ms
token_usage
estimated_cost
repair_attempts
error_code
为什么改
完全对应 JD 第三条：
- 调试体系
- 评测体系
- 质量保障
- 任务完成率
- 准确性
- 稳定性
- 响应效率
- 使用成本
- 问题分析和持续优化
这也是项目从“功能展示”升级为“工程实验”的关键。
完成标准
- 一条命令生成 JSON 和 Markdown 报告。
- 能对比改造前后的指标。
- 指标低于阈值时返回非零退出码。
- 每个失败都有 Graph 节点和错误分类。
- 核心用例重复运行 3 次评估稳定性。
预计时间
1.5～2 天。
5. Memory 只有短期会话缓存
- 待解决
现在是什么
当前 [memory.py (line 17)](../backend/src/agent/memory.py:17) 包含：
- TTL 短期记忆
- 消息数限制
- 历史恢复
- 上下文压缩
但 Working Memory、Conversation Memory、长期偏好都混在对话历史概念里。
要改什么
保留现有短期记忆，新增：
backend/src/agent/working_memory.py
backend/src/agent/preference_memory.py
backend/migrations/010_memory_items.sql
改成什么
拆成三层：
Working Memory
→ 当前 Graph 状态、候选歌曲、约束和修复结果

Conversation Memory
→ 当前会话历史、摘要和“刚才那首”的指代

Long-term Preference Memory
→ 用户明确确认的 DJ 长期偏好
长期记忆只保存明确表达：
暖场通常使用 118～124 BPM
不推荐某位艺人
优先保持 Camelot 兼容
偏好逐步升温
冲突优先级：
当前请求 > 当前项目设置 > 长期用户偏好 > 系统默认值
为什么改
直接对应 JD 中的：
- Context Management
- Memory
- 长链路任务
- Agent 效果稳定性
分层后可以清晰回答面试官：
什么信息存在哪里、保存多久、什么时候读取、冲突如何处理、如何防止记忆污染。

完成标准
- 临时条件不会错误写入长期记忆。
- 当前请求可以覆盖历史偏好。
- 不同会话的指代不会串线。
- 用户可以查看和删除长期记忆。
- 删除后不再影响后续任务。
预计时间
1～1.5 天。
实施顺序
1. LangGraph 执行图
2. SetValidator + SetRepairer
3. Checkpoint 与幂等
4. Eval 2.0 与 Run Trace
5. 三层 Memory
总工期：约 6～8 个专注工作日。
当前进度：第 1～3 项已完成；下一步按顺序处理第 4 项 Eval 2.0 与 Run Trace。
