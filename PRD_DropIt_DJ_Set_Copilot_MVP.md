# DropIt DJ Set Copilot

> 最小可行产品（MVP）｜LangGraph · Subagent · RAG

| 项目 | 内容 |
| --- | --- |
| 项目定位 | 面向实习作品集的本地 DJ Set 生成助手 |
| 交付目标 | 从本地歌曲导入到可审核 Playlist 的完整 Agent 闭环 |
| 开发周期 | 7–10 天；10–30 首歌曲的受控 Demo |

## 1. 背景与目标

目标：将分散的本地歌曲转为可搜索、可解释的曲库，并根据演出 Brief 自动生成 Set 草案。

作品集重点：展示 LangGraph 状态编排、三个专业 Subagent、RAG 检索和人工审核闭环。

## 2. 用户流程与范围

用户流程：

```text
选择本地歌曲文件夹
→ 读取标签与基础音频特征
→ 建立曲库检索索引
→ 填写 Brief
→ 生成与审核 Set
→ 人工确认
→ 导出 Playlist
```

本版包含：

- 支持本地 MP3、WAV、FLAC、AIFF、M4A
- 输出 JSON、CSV、M3U

本版不做：

- 音乐平台下载
- DRM
- rekordbox/Serato 内部数据库写入
- 现场实时控制
- 自动混音

## 3. 核心功能

| 模块 | 输入 | 输出 |
| --- | --- | --- |
| 曲库导入 | 本地音频文件夹 | 标签、时长、BPM、基础能量特征 |
| RAG 检索 | 自然语言需求 + 数值条件 | 可解释的候选歌曲集合 |
| Set 生成 | 时长、BPM、能量、风格要求 | 顺序、理由、替代歌曲 |
| 审核导出 | Critic 结果 + 用户确认 | 约束报告与 Playlist 文件 |

## 4. Agent 与 RAG 设计

### Curator Subagent

负责把音频特征转为 Mood、Energy、Set Role 与检索文本；不直接修改原始文件。

### Planner Subagent

调用曲库检索与筛选工具，按 Brief 生成 Set 和候选替换曲。

### Critic Subagent

检查时长、重复、BPM 跳跃、能量曲线和用户硬约束；失败时要求 Planner 最多修改两次。

### RAG

每首歌曲形成一条含 BPM、Key、Energy、Mood、备注的文档。语义检索负责召回，SQLite 过滤精确数值。

## 5. 技术栈

| 层 | 选型 |
| --- | --- |
| 编排与 Agent | Python 3.11+、LangGraph、LangChain、Pydantic |
| 模型与检索 | DeepSeek/通义千问兼容接口、BGE Embedding、Chroma |
| 数据与音频 | SQLite、Mutagen、librosa、ffprobe、可选 FFmpeg |
| 界面与工程 | Streamlit、pytest、Ruff、uv 或 pip |

## 6. 代码框架

```text
app/
  graph/      state.py, nodes.py, routes.py, builder.py
  agents/     curator.py, planner.py, critic.py
  tools/      audio_tools.py, retrieval_tools.py, export_tools.py
  rag/        indexer.py, retriever.py, documents.py
  models/     track.py, brief.py, playlist.py
  storage/    database.py, repositories.py
  main.py     Streamlit 入口
tests/        图路由、检索、规则和导出测试
data/         演示歌曲、向量库与导出文件
```

## 7. 验收标准与计划

### 验收标准

- 20 首测试歌曲可完成导入与索引
- Set 无重复，时长误差不超过 10%
- Critic 至少触发一次可追踪的修改循环
- 用户能导出 M3U/JSON

### 开发节奏

- 第 1–2 天：完成导入和数据模型
- 第 3–4 天：完成 RAG
- 第 5–6 天：完成 LangGraph 与 Subagent
- 第 7–8 天：完成 Streamlit、测试和 README

### 风险控制

- BPM/Key 可能不准确，结果必须可编辑
- LLM 只可使用 Retriever 返回的 `track_id`，避免编造歌曲
- 图的重试次数固定为 2
