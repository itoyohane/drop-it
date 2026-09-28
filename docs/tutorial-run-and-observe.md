# 动手教程：跑通 DropIt，再跟踪一次请求

完成后你应看到健康接口、分析后的歌曲、检索结果和 Set 的成功或明确冲突，并能把一次操作对应到源码。项目原理见[架构解释](explanation-agent-architecture.md)，参数见[参考](reference-api-tools-and-data.md)，返回[学习首页](README.md)。

所有命令从仓库根目录执行。下面使用 Windows PowerShell；Python 用项目 Docker 同样的 3.11，Node 至少 20.19 或 22.12（Vite 7 的要求），安装 npm。准备几首有权使用的本地音频；首次先用 WAV，避免把格式解码问题和模型问题混在一起。真实模型调用需要自己的 Key，可能联网计费；前两步不需要 Key。

## 1. 安装完整运行依赖

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-audio.txt
npm.cmd install
```

`requirements-audio.txt` 包含基础依赖与 librosa；只装 requirements.txt 不足以保证真实音频分析可用。每个新终端都要激活同一虚拟环境，因为 npm 的 dev:api 脚本调用 PATH 中的 python。若系统策略不允许 Activate.ps1，可在 cmd 终端运行 `.venv\Scripts\activate.bat` 后继续，不必修改整机执行策略。

## 2. 启动并得到第一个可见结果

```powershell
$env:DROPIT_INTENT_FALLBACK_ENABLED="false"
$env:LANGSMITH_TRACING="false"
npm.cmd run dev
```

禁用可选 Ollama 兜底是为了先减少外部依赖，不改变三个明确业务分支。等终端显示两个服务已监听后，打开 `http://127.0.0.1:5173/`。在另一个终端检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8765/api/health
Invoke-RestMethod http://127.0.0.1:5173/api/health
```

应见 `status=ok`、`agent_framework=langgraph.state_graph` 和三个业务名。直接访问与代理访问应一致；未配置 Key 时 model_configured=false 是正常结果，不代表后端坏了。交互 API 文档在 `http://127.0.0.1:8765/api/docs`。

如果 5173 占用，先 Ctrl+C 停止刚启动的服务，再设置 `$env:DROPIT_WEB_PORT=5174` 后重启；访问地址也改为 5174。API 仍是 8765，改 Web 端口不会改 API 端口。不要同时启动两套后端。

到这里已验证安装、API、前端和代理，还没验证音频或模型。

## 3. 配置真正的 AI 链路

在根目录创建或编辑 `.env`，保留原有配置，加入自己的值；下面全是占位内容，不能原样调用：

```dotenv
DEEPSEEK_API_KEY=YOUR_DEEPSEEK_KEY
DASHSCOPE_API_KEY=YOUR_DASHSCOPE_KEY
DROPIT_INTENT_FALLBACK_ENABLED=false
LANGSMITH_TRACING=false
```

描述服务默认复用聊天 Key，可另设 `DEEPSEEK_DESCRIPTION_API_KEY`。聊天/描述/embedding 模型名和 URL 以[配置参考](reference-api-tools-and-data.md)为准；源码默认模型名不保证账号有权限。若服务报模型不存在，使用账号实际可用、接口兼容的模型，embedding 维度也要匹配。此教程不自动购买额度或建立账号。

`.env` 已被 Git 忽略；不要提交 Key，不要截图完整配置，也不要使用 VITE_ 前缀暴露给浏览器。修改后 Ctrl+C 再启动，使新建的模型适配器读取配置。health 的 model_configured=true 只证明 Key 非空，不证明凭证有效、配额足够或模型可访问。

音频在本地分析，但数值特征、文本、对话证据会分别发送给对应模型服务；这是本地曲库应用，不是全离线应用。

## 4. 建立曲库，分别观察三个阶段

在页面创建一个具体项目，导入本地音频文件夹。界面先解析来源，再上传文件；后台 Job 处理歌曲。默认支持 mp3/wav/flac/aiff/aif/m4a，实际能否解码也取决于环境；WAV 是首次验证的较简单选择。

等待任务结束，检查项目曲库，不能只看上传成功。页面不展示全部索引字段时，在 `/api/docs` 调用 `GET /api/projects/{project_id}/library` 查看歌曲 JSON；project_id 从项目列表响应取得：

- 数值分析：analysis_status 为 analyzed，出现 BPM、key、camelot_key、energy 和 analyzer。
- 描述：description 非空、description_model 是当前版本。
- 向量：embedding_status 为 ready、embedding_model 是当前版本；真实向量还须存在于 Chroma。

导入返回 `202` 只是受理。Job 失败时看每首歌的 analysis_error/embedding_error：有 BPM 但描述或索引失败，说明前一个阶段可能已成功，不必把所有问题归咎于 librosa。源码跟到 `JobRunner._analyze` 看阶段保留与重试。

通过 `/api/docs` 或页面查询任务；单个任务接口是 `GET /api/jobs/{job_id}`。普通分析请求 `{ "track_ids": [] }` 重试未 ready 歌曲；明确指定 IDs 会强制重做特征，不要把它当仅重试 embedding。

## 5. 按顺序验证五个聊天分支

在具体项目的会话中依次发送：

1. “搜索曲库里 120 到 128 BPM 的歌曲，不需要风格条件。”观察 search_library 事件；这类元数据搜索可以不生成 query 向量，但聊天参数提取与回答仍需要模型。
2. “搜索适合暖场、能量偏低的歌曲。”观察语义 query 与筛选是否分开；返回结果来自当前曲库，不是外部音乐目录。当前描述主要依据测量特征，不保证懂任意风格词。
3. 从曲库 API 响应复制一首歌曲的 id，发送“找几首和 track_id:【实际ID】相似的歌”。检查不返回参考歌本身。不要原样发送占位符，也不要只说“这首”却不给上下文。
4. “请生成 10 分钟、BPM 110 到 140、能量逐步上升的 DJ Set，不限制风格。”先选目标较短、曲目充足的项目；检查提取后的 style_query 是否为空。候选不足或硬约束冲突是合理失败，不能要求模型补不存在的歌。
5. “解释 Camelot wheel。”应为 music_chat，不出现本地曲库工具；“帮我写 Python 代码。”应拒答，不调用主回答模型或业务函数（入口仍要求已配置 Key）。

Set 成功时应有真实 Playlist，可查看、调序、确认和导出；失败时应有明确原因，constraint_conflict 不应返回或保存无效歌单。当前没有自动混音，10 分钟按原曲时长求和计算；调序/确认不代表重新通过全部硬约束。

浏览器开发者工具 Network 中找到 `/chat/stream`，观察 user_saved、status、tool、token、error、complete。token 和 tool 的具体先后取决于执行结果；恢复或错误请求可能没有 token。最终以 complete 中的保存消息与 error_code 为准。

## 6. 不调用模型的源码学习实验

下面只构造三首合成 Track 测试编排和校验，不会读取音频、联网或写曲库。它证明业务算法可脱离 LLM 运行，不证明真实推荐质量。在根目录的已激活终端执行：

```powershell
@'
from backend.models import Track
from backend.agent.set_planning import plan_set, validate_and_repair_set

tracks = [
    Track(id=name, title=name, artist="Demo", filename=f"{name}.wav",
          path=f"demo/{name}.wav", duration_sec=200, bpm=124,
          key="A minor", camelot_key="8A", energy=energy,
          analysis_status="analyzed")
    for name, energy in [("a", .2), ("b", .5), ("c", .8)]
]
playlist = plan_set(
    "demo", tracks, request="10 minute demo", duration_min=10,
    bpm_min=110, bpm_max=140, energy_curve="build", style_query="",
)
playlist, validation, attempts = validate_and_repair_set(playlist, tracks)
assert validation.valid and playlist.duration_sec == 600
print([row.track.id for row in playlist.tracks], playlist.duration_sec, validation.valid)
'@ | python -
```

预期输出 `['a', 'b', 'c'] 600 True`。然后把 request 中的目标时长改为 30 分钟（修改 duration_min 为 30）：三首总长不足，观察 SetConstraintConflictError。这不是“模型不聪明”，而是证据和约束不可满足。先看 validator 的 issues，再看 repairer 为什么不制造新歌曲。

## 7. 用 API 验证同一 run_id 的重试

此实验会创建一个演示项目、会话和真实模型请求；在独立演示数据中操作，可能计费。不必模拟宕机即可验证重复请求不会重复写同一轮消息：

```powershell
$demoApi="http://127.0.0.1:8765"
$demoProject=Invoke-RestMethod "$demoApi/api/projects" -Method Post -ContentType "application/json" -Body '{"name":"Run demo"}'
$demoConversation=Invoke-RestMethod "$demoApi/api/projects/$($demoProject.id)/conversations" -Method Post -ContentType "application/json" -Body '{"title":"Resume demo"}'
$demoRun=[guid]::NewGuid().ToString("N")
$demoBody=@{message="解释 Camelot wheel。";run_id=$demoRun} | ConvertTo-Json -Compress
$demoPath="$demoApi/api/projects/$($demoProject.id)/conversations/$($demoConversation.id)/chat"
$demoFirst=Invoke-RestMethod $demoPath -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($demoBody))
$demoAgain=Invoke-RestMethod $demoPath -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($demoBody))
$demoFirst.message.id -eq $demoAgain.message.id
$demoAgain.resumed
```

成功请求重复发送预期两项为 True。若首次 error_code 非空，先处理模型错误，不能拿失败结果证明模型正常。这个实验验证完成结果复用；中断节点恢复、并发 fencing 和歌单去重由下一节测试覆盖。不要声称已经证明任意断网恢复；当前浏览器并未发送 run_id 重试。

## 8. 自动检查与真实质量复测

```powershell
npm.cmd test
npm.cmd run build
```

前者运行前端 SSE 测试与后端轻量测试；后者把前端写入 backend/dist。需要单独定位时：

```powershell
python -m pytest backend/src/tests/test_agent_graph.py -q --basetemp .pytest-run -p no:cacheprovider
python -m pytest backend/src/tests/test_p0_set_and_checkpoints.py -q --basetemp .pytest-run -p no:cacheprovider
```

轻量测试隔离真实库并使用假模型，不能替代云端验收。默认跳过的 test_real_audio_models 仍是遗留本地适配器测试，打开 DROPIT_TEST_AUDIO_MODELS 并不是本项目默认云端链路验证方法。

[评测脚本](../backend/src/evals/run_quality_eval.py)会读取当前曲库并调用配置的 embedding/可选意图模型。准备好真实索引、确认数据范围和成本后，可另存报告，避免覆盖历史文件：

```powershell
python -m backend.evals.run_quality_eval --rounds 1 --json-out data/quality-eval-current.json --report-out data/quality-eval-current.md
```

它用内存代理截获评测歌单写入，不向真实曲库保存评测 Set；但仍会调用外部 API，读取音乐事实，也会写输出报告。自检索 Hit@3 只是索引闭环，不能代替独立查询相关性、听感或最终回答真实性评测。历史七轮结果不是当前 checkout 刚跑的结果。

验证生产静态服务：先停止 dev，再执行 `npm.cmd run start`，打开 `http://127.0.0.1:8765/`。该脚本监听 0.0.0.0，请仅在可信本地环境使用；当前没有登录鉴权，不要直接公网部署。

## 9. 排障按层次，不按猜测

| 现象 | 先检查 |
| --- | --- |
| 端口占用或页面打不开 | 两个服务是否启动、5173/8765 是否占用；不要只改代理后仍沿用旧 URL |
| 页面能开，/api 失败 | 对比直连 health 与代理 health，检查 DROPIT_API_TARGET |
| 聊天 503 | Key 是否加载到实际服务进程；重启；health 配置标志 |
| 配置标志 true，仍调用失败 | 服务端日志/请求编号、账号权限、模型名、额度、网络；不输出完整 Key |
| 导入 202 后迟迟无结果 | 查 Job 与曲目阶段错误；大文件分析不等于聊天线程阻塞 |
| BPM 有值但语义搜索失败 | description、embedding_status/model_key、API Key、Chroma 向量，而非只看分析状态 |
| 参考歌找不到或重名 | 使用当前项目内真实 track_id；标题精确匹配不做猜测 |
| Set 冲突 | 曲库数量、总长、BPM/Camelot/能量和必选约束；看结构化问题，不靠放宽 prompt 强行保存 |
| 流显示 error 后又 complete | complete 表示收尾，仍看 error_code 和 playlist，不当成成功 |

完成后记录一份个人验收：运行环境、源码 commit、导入数量、一个实际 query 与结果、一个可行/不可行 Set、一个错误请求。只写实际观察到的结果。下一步用[架构页](explanation-agent-architecture.md)解释每个现象来自哪段代码。
