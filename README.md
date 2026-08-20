# DropIt DJ Set Copilot

DropIt 是一个 React + FastAPI 的 DJ Set 生成助手网页原型。浏览器负责曲库上传、Brief 与审核界面，FastAPI 负责音频索引、规划、规则审核和导出。

## 已实现的 MVP 闭环

- 从浏览器选择音频文件夹，支持 MP3、WAV、FLAC、AIFF、M4A
- 使用 Mutagen 读取可用标签与时长
- 将曲库保存到本地 SQLite，并生成可检索的 BPM、Key、Energy、Mood、Set Role
- 根据时长、BPM、能量曲线、风格和补充说明生成 Set
- 显示 Curator、Planner、Critic 的可追踪协作记录，固定展示一次 Critic 修订闭环
- 支持人工上移、下移曲目并确认 Set
- 导出 M3U、JSON、CSV

当前 BPM、Key、Energy 是可重复的轻量原型特征，用于 10-30 首歌曲的作品集演示。后续可在 `backend/services.py` 中替换为 librosa、Essentia 或外部分析服务，不影响前端和数据契约。

## 本地运行

要求 Node.js 20+ 与 Python 3.11+。

```bash
npm install
python -m pip install -r requirements.txt
npm run dev
```

打开 `http://127.0.0.1:5173`，然后选择一个本地音频文件夹。开发模式会并行启动 Vite 和 FastAPI。

## 生产运行

React 构建产物由 FastAPI 直接托管，只需要启动一个服务：

```bash
npm run build
npm start
```

打开 `http://127.0.0.1:8765`。

## 验证

```bash
npm run build
npm test
```

FastAPI 文档位于 `http://127.0.0.1:8765/docs`，前端开发地址为 `http://127.0.0.1:5173`。

## 工程结构

```text
backend/       FastAPI、SQLite、模型与 Set 生成服务
src/           React 界面与本地 API 客户端
```
