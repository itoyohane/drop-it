# 运行并观察一次 DropIt 工作流

本教程会启动后端、检查歌曲描述与文本向量 API 配置，并说明如何验证音频分析、曲库检索和 Agent。

## 前提

- Python 3.11+；librosa 支持 Windows、macOS 和 Linux。
- 至少准备两首支持格式的本地歌曲，找相似至少需要一首参考歌和一首候选歌。
- 对话/描述需要 DeepSeek 兼容 API Key，语义检索需要 DashScope API Key；librosa 数值分析本身不需要 API Key。

## 第 1 步：用 Docker 启动

在仓库根目录运行：

```bash
docker compose -f compose.backend.yaml build
docker compose -f compose.backend.yaml run --rm api python -m backend.music.download_models
docker compose -f compose.backend.yaml up
```

第二条命令现在只检查 DeepSeek、DashScope 与 Chroma 配置；描述和 embedding 使用远程 API，不下载本地权重。
第三条命令启动单个 Uvicorn worker，避免多个进程竞争同一 SQLite 任务。

打开 `http://127.0.0.1:8765/api/docs`。`GET /api/health` 应返回 `status: ok`、
`embedding_provider: librosa-deepseek-dashscope-chroma-rag` 和三个工具名。生产模式会关闭 OpenAPI 页面。

## 第 2 步：配置聊天模型

在根目录新建未提交的 `.env`：

```dotenv
DEEPSEEK_API_KEY=替换为服务端密钥
DEEPSEEK_DESCRIPTION_API_KEY=可选，歌曲描述专用密钥
DASHSCOPE_API_KEY=替换为 DashScope 密钥
DROPIT_MODEL=替换为账号实际可用的模型名
DROPIT_DESCRIPTION_MODEL=deepseek-v4.1-flash
DROPIT_TEXT_EMBEDDING_MODEL=qwen3.7-text-embedding
DEEPSEEK_BASE_URL=https://api.deepseek.com
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

Compose 读取当前 shell 环境做变量替换，不会自动把 `.env` 复制进镜像。修改后重启服务。
如果只测试曲库元数据和 librosa 管线，可留空；缺少对应 API key 时，描述/embedding 阶段会在曲目状态中报告失败。

## 第 3 步：导入并等待分析

通过 OpenAPI 页面依次执行：

1. `POST /api/projects` 创建项目。
2. `POST /api/projects/{id}/sources/resolve` 绑定一个来源。`folder_key` 是客户端生成的稳定标识，至少 16 字符；它不是服务端文件夹路径。
3. `POST /api/projects/{id}/library/import-files`，传入返回的 `source_id` 与音频文件。
4. 轮询返回 Job 的 `GET /api/jobs/{job_id}`，直到 `completed` 或 `failed`。
5. `GET /api/projects/{id}/library` 检查两个独立状态：`analysis_status=analyzed` 和 `embedding_status=ready`。

分析任务依次保存 librosa 结果、DeepSeek 歌曲文本描述和 DashScope 描述向量。因此 API 或 embedding 失败不会抹掉 BPM、调性和能量。
进程中断后的 queued/running 任务会在下次启动时恢复。显式把 `track_ids` 传给 analyze 接口会强制重跑；
空数组只补齐尚未就绪的阶段。

对应代码路径是 `workers/analyze_track.py → music/librosa_analyzer.py →
music/text_models.py → music/indexer.py`。Worker 直接管理分阶段执行与重试，indexer 负责向量校验和写库分界。

## 第 4 步：从对话调用三个工具

创建项目对话后，依次尝试：

```text
列出当前项目 120 到 128 BPM 的歌曲。
找出与「准确歌名」最相似的三首歌。
用上面的候选排一个 45 分钟、逐步升温、118 到 132 BPM 的 Set。
```

第一句应让 Agent 用空 query 加数值过滤调用 `search_library`。第二句先解析歌曲 ID，再调用
`find_similar_tracks`；重名时应先向用户确认。第三句调用 `generate_dj_set`，成功的 complete
事件包含保存后的 Playlist。自然语言声音描述会与生成的歌曲描述在同一文本向量空间检索。

全局曲库对话可以搜索和找相似，但不能保存 Set；需要进入项目。任何结果都只应出现 Tool 返回的
track_id，不应出现模型编造的歌曲。

## 验证

不安装大模型的离线测试：

```bash
python -m pip install -r requirements.txt
python -m pytest backend/tests -q
```

它用注入的轻量分析器、描述器和向量器验证任务、项目范围、检索、工具、API、短期记忆和迁移。
它不下载模型，也不验证预训练模型的推荐质量。

真实模型 smoke test：

```bash
python -m pip install -r requirements-audio.txt
python -m backend.music.download_models
```

该命令不会下载权重；真实导入 smoke test 需要配置 DeepSeek 与 DashScope key，使用合成点击/和弦音频，
只证明 API 链路可用，不代表真实曲库 BPM/调性准确率或检索排序质量。

## 常见故障

### librosa 无法解码音频

确认安装 `requirements-audio.txt`，并检查对应格式的 soundfile/audioread 解码支持。

### 提示没有当前歌曲描述索引

确认 DeepSeek 与 DashScope key、base URL 和模型名正确，并查看曲目的 `embedding_error`。
模型、revision 或 embedding 维度改变会产生新 model key；旧向量不会复用，需要重新分析。

### 搜索能用，风格检索不能用

空 query 的元数据查询不需要向量；非空声音描述需要歌曲描述向量。先等待 embedding ready。

### Set 曲目数或时长不足

检查歌曲是否 analyzed、BPM 是否落在范围内。风格 query 会进一步缩小候选，规划器不会悄悄
放宽用户条件。Set 使用整曲时长，没有扣除混音重叠。

### 前端行为与文档不一致

本次分层重构只调整后端。先用 OpenAPI 和后端测试确认接口，再根据
[接口参考](reference-api-tools-and-data.md)检查前端调用和 SSE 事件处理。
