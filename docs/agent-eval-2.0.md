# Agent Eval 2.0 与组件质量评测

评测对齐当前 main：受控 LangGraph 节点直接调用 `DropItToolRegistry` 和 Set 规划函数，
不依赖已删除的 LangChain Tool 适配器。模块位于 `backend/src/evals/`，
兼容命令入口仍是 `python -m backend.evals.…`。

## 两个入口

- `run_agent_eval`：真实 Agent 能力评测，包含意图路由、结构化参数提取、检索图执行、
  多轮上下文、越权拦截和最终回答。固定 37 条用例：搜索 15、相似推荐 10、
  多轮 7、越权 5。不包含 Set 编排。
- `run_quality_eval`：组件评测，单独检查向量完整性、25 条意图样本、8 条合法业务调用、
  RAG 描述自检索 Hit@3 和返回证据事实一致性。它不会隐式执行 Agent 评测。

Agent 套件已移除 20 条独立 Set 用例和 3 条包含 Set 的多轮用例，保留其他用例 ID。
生产 Set 功能及其单元/集成测试不受影响；独立的组件评测仍保留 Set 业务调用。

用例在 `backend/src/evals/cases/agent_tasks.jsonl`。真实曲目 ID、参考标题、BPM 等
通过运行时 `fixture` 模板生成，不将真实用户数据提交到仓库。
自动化测试用假模型驱动生产图；CLI 的 fixture 模式则只验证数据、评分和报告格式，
不会执行生产图，不能作为模型质量成绩。

## 快速运行与配置

从仓库根目录运行离线 smoke test（不联网）：

```powershell
python -m backend.evals.run_agent_eval --mode fixture --rounds 1 --limit 3 `
  --json-out eval-results/smoke.json --report-out eval-results/smoke.md
```

真实评测复用生产 Settings 和意图 fallback 配置。worktree 通常没有主目录的
`.env` 和被 Git 忽略的 `data/`，必须指向已有数据：

```powershell
python -m backend.evals.run_agent_eval --mode live `
  --env-file "C:/path/to/drop-it/.env" `
  --data-dir "C:/path/to/drop-it/data" `
  --chroma-dir "C:/path/to/drop-it/data/chroma" `
  --rounds 1 `
  --json-out eval-results/agent-main.json --report-out eval-results/agent-main.md `
  --threshold task_completion_rate=80
```

将示例路径替换为实际位置。需要 `DEEPSEEK_API_KEY`、`DASHSCOPE_API_KEY`、
当前 embedding 模型的已分析歌曲和可用的模型服务。启用意图 fallback 时也需要 Ollama。
`--project-id` 可指定项目；省略时选择当前模型向量就绪曲目最多的项目。
没有合格项目时拒绝运行，不会自动创建空库。
至少需要两首合格曲目以支持相似检索；当前 Agent 套件不要求曲库满足 Set 编排约束。
若某轮被错误路由到 `generate_dj_set`，CLI 会在执行业务图前停止该轮，记录
`out_of_scope_route` 并计为失败，不会实际规划、修复或保存 Set。

`--env-file` 只指定配置文件，不改变工作目录；文件内相对的 `DROPIT_DATA_DIR`
仍相对于当前工作目录解析。显式 `--data-dir` / `--chroma-dir` 优先于配置和环境变量。
这也是 worktree 中出现 `source database does not exist: data/dropit.db` 的常见原因。
缺库错误现在会显示绝对路径、当前目录和修正参数。

## 执行轮数、时间和产物

不传 `--rounds` 时遵守每个用例的 `repeats`：7 个核心用例各 3 次，其余各 1 次，
共 51 次任务执行。`--rounds N` 覆盖每个选中用例的重复数，所以全量 `--rounds 1`
是 37 次、`--rounds 3` 是 111 次。`--limit N` 只选择文件前 N 条，适合启动检查，
不能当作均衡抽样。每次重复创建独立对话，多轮任务内各轮使用不同 Agent run_id。

任务顺序执行；实际耗时取决于模型、API 限流和对话轮数。搜索/相似推荐的
每轮通常各有参数提取和最终回答两次聊天模型请求，模糊意图可能再调用 Ollama，
语义检索还会调用 embedding API。可先跑 `--rounds 1 --limit 3`，根据实测单任务
延迟估算全量耗时；默认没有固定分钟数承诺。

每条任务完成即打印进度并 flush 到 JSONL。产物包括：
默认输出目录为已被 Git 忽略的 eval-results，避免把真实曲库评测产物提交到仓库；
组件评测也使用该目录，不默认覆盖 docs 中历史七轮报告。

- `--json-out`：完整结果 JSON，含指标、逐任务 trace、阈值和 baseline 差异。
- `--report-out`：Markdown 指标、失败分类/节点汇总。
- `--traces-out`：增量 JSONL，默认与 JSON 同目录且后缀为 `.runs.jsonl`。
  中断后已完成任务仍可查；中断任务不会生成完整汇总，也不自动断点续跑。

重复运行需要换输出文件名，避免覆盖既有增量 trace。阈值失败仍写完整报告，退出码为 1；
通过为 0；初始化或配置错误也会非零退出。
`--baseline previous.json` 比较已有百分比指标，不自动判定回退。
baseline 应使用相同数据、模型配置、用例范围和重复策略；旧版包含 Set 的 80 次结果
不能与当前 51 次结果直接比较。JSON 的 `evaluation_scope` 标记
`profile=agent_capabilities` 和 `excluded_capabilities=["set_planning"]`。
`--threshold` 可使用 `command_accuracy`、`argument_accuracy`、`tool_sequence_accuracy`、
`tool_execution_success_rate`、`graph_sequence_accuracy`、`task_completion_rate`、
`unauthorized_action_block_rate` 和 `stability`。为兼容旧报告保留的
`set_constraint_pass_rate`、`repair_success_rate` 在当前套件中均为 N/A，
不要为它们配置通过阈值。

## 真实执行证据与指标

`DropItAgent` 提供默认关闭的内部 `trace_callback`，eval 观察实际编译图的节点更新，
不改 SSE 协议。参数取自生产 Pydantic command，而不是从最终文本反推。
命令参数提取显式使用 `function_calling`，并仅在这次请求中关闭 DeepSeek thinking，
避免默认思考模式与强制指定提取函数的 `tool_choice` 冲突；仍保留严格 schema 校验。
最终回答不复用该请求的模式覆盖，保持原始模型配置。

| 路由 | 实际节点 |
| --- | --- |
| 搜索 | search_library → respond |
| 相似推荐 | resolve_reference → find_similar_tracks → respond |
| 音乐问答 | respond_chat |
| 越权拒绝 | reject_response |

参考曲目的标题解析不是一次虚构的 `search_library` 工具调用；业务事件只有生产发出的
done/failed 事件。trace 保存实际节点、typed command、工具参数/状态、每轮 run_id、
持久化 checkpoint 步骤、修复次数、结构化校验问题、真实失败节点和按来源/轮次/节点记录的
`error_chain`。最终图流报告 `graph_failed` 时，它会成为主要错误，先前节点错误仍保留在链中；
较早轮次的失败也会保留，即使之后的轮次成功。

- Command Accuracy：最终轮的实际意图与预期 exact match。
- Argument Accuracy：用例参数断言通过数 / 总断言数；无参数断言时分母为 0。
- Tool Sequence Accuracy：整个任务的业务事件名称序列 exact match，只比较名称顺序。
- Tool Execution Success Rate：至少发出一个业务工具事件的任务中，所有事件状态都明确为 `done` 的比例；
  没有工具调用的拒答不计入分母，缺少状态也不算成功。
- Graph Sequence Accuracy：预期节点按顺序出现在实际节点序列中。
- Task Completion Rate：路由、参数、工具/图顺序、complete/interception 预期全部通过，
  没有错误码；收到 complete 不代表任务成功。
- Set Constraint Pass Rate：兼容字段，当前 Agent 套件无 Set 用例，显示 N/A（0/0）。
- Unauthorized Action Block Rate：越权任务被拒绝且没有业务调用的比例。
- Stability：有重复执行的用例中，命令、业务参数/状态、任务成功与错误码保持一致的比例；
  它衡量重复结果的一致性，不代表任务成功率。`--rounds 1` 没有重复样本，不构成稳定性证据。
- P50/P95 Latency：任务端到端延迟（不含初始数据快照时间）。
- Repair Success Rate：兼容字段，当前 Agent 套件不执行 Set 修复，显示 N/A。

底层 Set 评分辅助函数及生产图回归测试继续保留，但不会进入当前 Agent CLI 套件。
套件校验也拒绝在多轮等其他类别中混入预期 Set 命令、工具或规划节点。

聊天模型回调读取供应商 usage metadata；缺失标为 unavailable，部分缺失标为 partial。
`--input-price-per-million` 和 `--output-price-per-million` 必须成对给出有限非负的
美元单价，否则不估价。不内置供应商价格。用量/费用仅包含可观测聊天模型请求，
不包含 embedding 和意图分类，partial 金额仅为已知小计，不是完整账单。

## 数据隔离与组件汇总

两个入口共用 `EvaluationSnapshot`：SQLite Online Backup 包含已提交 WAL 数据；
Chroma 元数据库同样备份，向量段复制到临时目录。评测消息、租约、checkpoint、
歌单和 Chroma 客户端可能执行的迁移都只写临时副本，退出时关闭并清理。
创建快照期间应暂停导入/分析，避免跨 SQLite/Chroma 文件的并发更新；两个存储没有
共同事务，因此不能保证持续写入时的跨库原子快照。复制占用额外磁盘和启动时间。

组件评测也支持显式路径，可附带已有 Agent 结果，分别显示指标，不混合分母：

```powershell
python -m backend.evals.run_quality_eval --rounds 2 --rag-cases-per-round 5 `
  --env-file "C:/path/to/drop-it/.env" --data-dir "C:/path/to/drop-it/data" `
  --agent-eval-json eval-results/agent-main.json `
  --json-out eval-results/quality-main.json --report-out eval-results/quality-main.md
```

组件报告轮数和结果说明从当次数据生成，不再复述历史固定分数。RAG Hit@3 是原描述
自检索/索引闭环，不等于开放查询的人工相关性成绩；事实一致性也不等于自由回答的真实性。
真实 live 产物可能含歌曲信息和模型回答，应保存在本地，不提交密钥或真实用户数据。

## 代码位置与测试

- `backend/src/evals/run_agent_eval.py`：用例、runner、评分、报告、阈值及 CLI。
- `backend/src/evals/runtime.py`：显式配置与隔离快照。
- `backend/src/evals/tracing.py`：聊天模型用量回调。
- `backend/src/evals/run_quality_eval.py`：组件基准及独立 Agent 汇总。
- `backend/src/tests/test_agent_eval*.py`、`test_eval_runtime.py`、`test_quality_eval.py`：
  指标单测、真实生产图的假模型集成测试、CLI 和隔离测试。

```powershell
python -m pytest backend/src/tests -q --basetemp .pytest-run -p no:cacheprovider
```
