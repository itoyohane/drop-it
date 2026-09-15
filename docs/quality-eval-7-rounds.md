# DropIt 七轮质量评测报告

> 生成时间：`2026-09-12T13:22:15.791196+00:00`。原始逐用例结果见 [quality-eval-7-rounds.json](quality-eval-7-rounds.json)。

## 最终指标

| 指标 | 结果 | 样本/分母 | 操作性定义 |
| --- | --- | --- | --- |
| 歌曲向量化入库 | 78 首 | 完整性 100.00% | SQLite 当前模型 ready 与有效 Chroma 向量的交集 |
| 意图路由准确率 | 84.00% | 147/175 | 5 类平衡标注集 exact match |
| 工具调用成功率 | 100.00% | 56/56 | 返回 ok 且契约、范围、约束复核通过 |
| RAG 调用准确性 | 100.00% | 35/35 | 原描述自检索目标歌曲 Hit@3 |
| RAG 真实性 | 100.00% | 1260/1260 个事实 | 返回证据逐字段与 SQLite 权威事实一致 |

可用于项目描述的严格表述：

> 完成 `78` 首歌曲的有效向量化入库；在 7 轮、`175` 次意图判定、`56` 次合法工具调用和 `35` 次真实向量检索中，意图路由准确率达到 `84.00%`，工具调用成功率达到 `100.00%`，RAG Top-3 调用准确性与证据真实性分别达到 `100.00%` 和 `100.00%`。

## 七轮结果

| 轮次 | 有效向量 | 意图准确率 | 工具成功率 | RAG Hit@3 | RAG 真实性 |
| --- | --- | --- | --- | --- | --- |
| 1 | 78 | 84.00% | 100.00% | 100.00% | 100.00% |
| 2 | 78 | 84.00% | 100.00% | 100.00% | 100.00% |
| 3 | 78 | 84.00% | 100.00% | 100.00% | 100.00% |
| 4 | 78 | 84.00% | 100.00% | 100.00% | 100.00% |
| 5 | 78 | 84.00% | 100.00% | 100.00% | 100.00% |
| 6 | 78 | 84.00% | 100.00% | 100.00% | 100.00% |
| 7 | 78 | 84.00% | 100.00% | 100.00% | 100.00% |

## 结果解读

意图路由每轮固定错 4 条：1 条隐式曲库查询、2 条隐式接歌请求和 1 条外部实时榜单请求。它们均未命中关键词规则，且 Ollama 未返回有效 JSON，于是系统按设计安全降级为 `music_chat`。因此 84.00% 是稳定、可复现的系统性边界，而不是偶发抖动。优先修复方向是约束 Ollama 的结构化输出兼容性，并补充这些表达的规则或训练样本。

工具与 RAG 的 100% 表示本次合法用例和索引闭环全部通过，并不表示任意线上输入都必然成功。尤其 RAG 使用原描述自检索，适合验证索引链路，不应包装成开放式语义检索的人工相关性准确率。

## 总体测试方法与思路

评测固定生产代码、当前 `.env` 中的模型名和当前 SQLite/Chroma 数据快照。脚本只读生产曲库；`generate_dj_set` 的保存动作由内存代理截获，因此不会向生产数据库写入评测歌单。每轮按以下顺序执行：

1. **向量完整性审计**：读取 `analysis_status=analyzed`、`embedding_status=ready` 且模型 key 与当前配置一致的歌曲，再与 Chroma 实际向量取交集；检查维度、有限值和非零范数。
2. **意图路由**：对 `search_library`、`find_similar_tracks`、`generate_dj_set`、`music_chat`、`overstep` 各 5 条标注语句进行 exact-match，并保留实际标签、置信度和规则/Ollama 来源。
3. **工具调用**：执行 8 个合法用例，覆盖三个 Tool、空结果、语义搜索、范围过滤和 Set 约束；只有 `ToolResult.ok=true` 且输出契约复核通过才计成功。
4. **RAG 准确性**：每轮选择不同的 5 首已索引歌曲，以其已入库描述发起新的真实 DashScope 查询；目标歌曲进入 Top-3 计为命中。该指标验证 embedding API、模型版本、Chroma 读取、项目作用域和余弦排序的整条链路。
5. **RAG 真实性**：把每个返回结果的 12 个可见事实原子与 SQLite 源记录逐字段比较。真实性只衡量可自动验证的检索证据，不把模型主观文案当事实。

这种 RAG 准确率属于**索引闭环/自检索 Hit@3**，能证明技术链路正确，但不能替代人工构建的自然语言相关性数据集；若要宣称开放式用户查询准确率，应另建独立 query→relevant track 标注集。

## 两轮完整测试链路

### 第 1 轮

#### A. 向量链路

`SQLite 状态筛选 → model key 校验 → Chroma 按 track_id 读取 → 维度/有限值/非零范数校验 → 有效计数`

| 导入歌曲 | SQLite 当前 ready | Chroma 可读 | 最终有效 | 维度 | 完整性 |
| --- | --- | --- | --- | --- | --- |
| 275 | 78 | 78 | 78 | 1024 | 100.00% |

#### B. 意图链路

`标注输入 → 规则优先匹配 → 未命中时 Ollama 分类 → 标签 exact match → 汇总 Accuracy/Macro-F1`

| ID | 输入 | 期望 | 实际 | 来源 | 结果 |
| --- | --- | --- | --- | --- | --- |
| search-01 | 曲库里有哪些 120 到 128 BPM 的歌？ | search_library | search_library | rule_or_safe_fallback | 通过 |
| search-02 | 帮我找歌，调性是 8A。 | search_library | search_library | rule_or_safe_fallback | 通过 |
| search-03 | Search my local library for high-energy tracks. | search_library | search_library | rule_or_safe_fallback | 通过 |
| search-04 | 搜索艺人 NURKO 的作品。 | search_library | search_library | rule_or_safe_fallback | 通过 |
| search-05 | 我想看看本地收藏里速度适中的音乐。 | search_library | music_chat | rule_or_safe_fallback | 失败 |
| similar-01 | 找几首和这首相似的歌。 | find_similar_tracks | find_similar_tracks | rule_or_safe_fallback | 通过 |
| similar-02 | 这首后面下一首放什么？ | find_similar_tracks | find_similar_tracks | rule_or_safe_fallback | 通过 |
| similar-03 | Find something that sounds like this track. | find_similar_tracks | find_similar_tracks | rule_or_safe_fallback | 通过 |
| similar-04 | 这首适合接什么？ | find_similar_tracks | music_chat | rule_or_safe_fallback | 失败 |
| similar-05 | 以刚才那首为参考推荐几个替代曲目。 | find_similar_tracks | music_chat | rule_or_safe_fallback | 失败 |
| set-01 | 帮我编排一个 45 分钟的 DJ Set。 | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| set-02 | 做一份逐步升温的歌单。 | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| set-03 | 排歌：前半段平稳，后半段达到峰值。 | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| set-04 | 给今晚的暖场安排 30 分钟音乐。 | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| set-05 | Create a one-hour set between 118 and 132 BPM. | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| chat-01 | 解释一下 Camelot wheel。 | music_chat | music_chat | rule_or_safe_fallback | 通过 |
| chat-02 | 侧链压缩在电子音乐里有什么作用？ | music_chat | music_chat | rule_or_safe_fallback | 通过 |
| chat-03 | DJ 过渡时为什么要注意拍点？ | music_chat | music_chat | rule_or_safe_fallback | 通过 |
| chat-04 | 大调和小调听感上通常有什么区别？ | music_chat | music_chat | rule_or_safe_fallback | 通过 |
| chat-05 | 你好，聊聊音乐吧。 | music_chat | music_chat | ollama | 通过 |
| overstep-01 | 帮我写一段 Python 爬虫代码。 | overstep | overstep | rule_or_safe_fallback | 通过 |
| overstep-02 | 分析一下这只股票是否值得买。 | overstep | overstep | rule_or_safe_fallback | 通过 |
| overstep-03 | 总结今天最新政治新闻。 | overstep | overstep | rule_or_safe_fallback | 通过 |
| overstep-04 | 根据症状给我医疗诊断。 | overstep | overstep | rule_or_safe_fallback | 通过 |
| overstep-05 | 查一下 Spotify 实时排行榜。 | overstep | music_chat | rule_or_safe_fallback | 失败 |

本轮意图结果：`21/25`，Accuracy `84.00%`，Macro-F1 `84.84%`。

#### C. 工具链路

`合法参数 → LangChain Tool schema → 业务函数 → ToolResult → 输出契约/项目范围/约束复核`

| ID | 工具 | 参数 | 返回 | 契约 | 摘要 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| tool-01 | search_library | `{"query": "", "filters": {"bpm_min": 120, "bpm_max": 135}, "limit": 5}` | True | True | 找到 5 首曲目。 | 通过 |
| tool-02 | search_library | `{"query": "", "filters": {"title": "Last Wil"}, "limit": 5}` | True | True | 找到 1 首曲目。 | 通过 |
| tool-03 | search_library | `{"query": "high energy electronic music", "filters": null, "limit": 5}` | True | True | 找到 5 首曲目。 | 通过 |
| tool-04 | search_library | `{"query": "", "filters": {"title": "__dropit_eval_no_match__"}, "limit": 5}` | True | True | 找到 0 首曲目。 | 通过 |
| tool-05 | find_similar_tracks | `{"track_id": "e1c0468c81291ceb3d79", "limit": 3, "filters": null}` | True | True | 找到 3 首曲目。 | 通过 |
| tool-06 | find_similar_tracks | `{"track_id": "e1c0468c81291ceb3d79", "limit": 5, "filters": {"bpm_min": 60, "bpm_max": 220}}` | True | True | 找到 5 首曲目。 | 通过 |
| tool-07 | generate_dj_set | `{"request": "quality eval round 1 build", "duration_min": 10, "bpm_min": 60, "bpm_max": 220, "energy_curve": "build", "style_query": ""}` | True | True | 已生成 2 首、约 9 分钟的 Set。 | 通过 |
| tool-08 | generate_dj_set | `{"request": "quality eval round 1 scoped", "duration_min": 10, "bpm_min": 60, "bpm_max": 220, "energy_curve": "steady", "style_query": "", "track_ids": ["040d8aecd4b178302f91", "c905b7a001da1a61945c", "482792d0e401f7a09ea2", "d92887a5d63be97e5351", "252c1f7d28ed1ead6f0d", "5e146bf976728898b357", "7e3ac38d08fafc30b836", "8be4e1da6336d64698a3", "416b57d312cf3e3ecf16", "668fc1ecb6b4b414c6c0", "8e656e9bafae84ab376d", "a8df75fc21f64d3dc7ea", "bde89636f09f1ccb27ab", "ca5015bf4301c9e18a0d", "e1556d1e22847958ac6f", "e6a6a00281a127875d96", "43ab50199dc212585f04", "39d806aa331d1ba936f8", "7ab79c6cc8e754f9506b", "816887e3f689b6883ca8"]}` | True | True | 已生成 3 首、约 10 分钟的 Set。 | 通过 |

本轮工具结果：`8/8`，成功率 `100.00%`。

#### D. RAG 链路

`选定 gold 歌曲 → 读取已索引描述 → DashScope 重新编码查询 → 项目向量读取 → 余弦 Top-3 → Hit@3 → 返回事实逐字段对账`

| ID | Gold | Top-3（排名顺序） | Gold 排名 | 命中 | 事实核验 | 错误 |
| --- | --- | --- | --- | --- | --- | --- |
| rag-01 | Light In The Dark (`015c9014efbd753ee838`) | 1. Light In The Dark → 2. Cursed → 3. Tomorrow | 1 | True | 36/36 | — |
| rag-02 | Ones I Used To Love (`040d8aecd4b178302f91`) | 1. Ones I Used To Love → 2. More Than I Can Say → 3. Homesick (feat. SOUNDR) | 1 | True | 36/36 | — |
| rag-03 | Feel Again (`068777e4dcf62b2fc032`) | 1. Feel Again → 2. You Can Be My Light (feat. Monika Santucci) → 3. All We Get | 1 | True | 36/36 | — |
| rag-04 | WARP DRIVE (`0d96d1ecce8e1483e9f5`) | 1. WARP DRIVE → 2. Leave The Light On → 3. Aeipathy | 1 | True | 36/36 | — |
| rag-05 | Grand Escape (Teminite Remix) (`13b033711643e23b252a`) | 1. Grand Escape (Teminite Remix) → 2. Space Between → 3. Snowblind | 1 | True | 36/36 | — |

本轮 RAG：Hit@3 `100.00%`，MRR `100.00%`，真实性 `100.00%`。

### 第 7 轮

#### A. 向量链路

`SQLite 状态筛选 → model key 校验 → Chroma 按 track_id 读取 → 维度/有限值/非零范数校验 → 有效计数`

| 导入歌曲 | SQLite 当前 ready | Chroma 可读 | 最终有效 | 维度 | 完整性 |
| --- | --- | --- | --- | --- | --- |
| 275 | 78 | 78 | 78 | 1024 | 100.00% |

#### B. 意图链路

`标注输入 → 规则优先匹配 → 未命中时 Ollama 分类 → 标签 exact match → 汇总 Accuracy/Macro-F1`

| ID | 输入 | 期望 | 实际 | 来源 | 结果 |
| --- | --- | --- | --- | --- | --- |
| search-01 | 曲库里有哪些 120 到 128 BPM 的歌？ | search_library | search_library | rule_or_safe_fallback | 通过 |
| search-02 | 帮我找歌，调性是 8A。 | search_library | search_library | rule_or_safe_fallback | 通过 |
| search-03 | Search my local library for high-energy tracks. | search_library | search_library | rule_or_safe_fallback | 通过 |
| search-04 | 搜索艺人 NURKO 的作品。 | search_library | search_library | rule_or_safe_fallback | 通过 |
| search-05 | 我想看看本地收藏里速度适中的音乐。 | search_library | music_chat | rule_or_safe_fallback | 失败 |
| similar-01 | 找几首和这首相似的歌。 | find_similar_tracks | find_similar_tracks | rule_or_safe_fallback | 通过 |
| similar-02 | 这首后面下一首放什么？ | find_similar_tracks | find_similar_tracks | rule_or_safe_fallback | 通过 |
| similar-03 | Find something that sounds like this track. | find_similar_tracks | find_similar_tracks | rule_or_safe_fallback | 通过 |
| similar-04 | 这首适合接什么？ | find_similar_tracks | music_chat | rule_or_safe_fallback | 失败 |
| similar-05 | 以刚才那首为参考推荐几个替代曲目。 | find_similar_tracks | music_chat | rule_or_safe_fallback | 失败 |
| set-01 | 帮我编排一个 45 分钟的 DJ Set。 | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| set-02 | 做一份逐步升温的歌单。 | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| set-03 | 排歌：前半段平稳，后半段达到峰值。 | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| set-04 | 给今晚的暖场安排 30 分钟音乐。 | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| set-05 | Create a one-hour set between 118 and 132 BPM. | generate_dj_set | generate_dj_set | rule_or_safe_fallback | 通过 |
| chat-01 | 解释一下 Camelot wheel。 | music_chat | music_chat | rule_or_safe_fallback | 通过 |
| chat-02 | 侧链压缩在电子音乐里有什么作用？ | music_chat | music_chat | rule_or_safe_fallback | 通过 |
| chat-03 | DJ 过渡时为什么要注意拍点？ | music_chat | music_chat | rule_or_safe_fallback | 通过 |
| chat-04 | 大调和小调听感上通常有什么区别？ | music_chat | music_chat | rule_or_safe_fallback | 通过 |
| chat-05 | 你好，聊聊音乐吧。 | music_chat | music_chat | ollama | 通过 |
| overstep-01 | 帮我写一段 Python 爬虫代码。 | overstep | overstep | rule_or_safe_fallback | 通过 |
| overstep-02 | 分析一下这只股票是否值得买。 | overstep | overstep | rule_or_safe_fallback | 通过 |
| overstep-03 | 总结今天最新政治新闻。 | overstep | overstep | rule_or_safe_fallback | 通过 |
| overstep-04 | 根据症状给我医疗诊断。 | overstep | overstep | rule_or_safe_fallback | 通过 |
| overstep-05 | 查一下 Spotify 实时排行榜。 | overstep | music_chat | rule_or_safe_fallback | 失败 |

本轮意图结果：`21/25`，Accuracy `84.00%`，Macro-F1 `84.84%`。

#### C. 工具链路

`合法参数 → LangChain Tool schema → 业务函数 → ToolResult → 输出契约/项目范围/约束复核`

| ID | 工具 | 参数 | 返回 | 契约 | 摘要 | 结果 |
| --- | --- | --- | --- | --- | --- | --- |
| tool-01 | search_library | `{"query": "", "filters": {"bpm_min": 120, "bpm_max": 135}, "limit": 5}` | True | True | 找到 5 首曲目。 | 通过 |
| tool-02 | search_library | `{"query": "", "filters": {"title": "Last Wil"}, "limit": 5}` | True | True | 找到 1 首曲目。 | 通过 |
| tool-03 | search_library | `{"query": "high energy electronic music", "filters": null, "limit": 5}` | True | True | 找到 5 首曲目。 | 通过 |
| tool-04 | search_library | `{"query": "", "filters": {"title": "__dropit_eval_no_match__"}, "limit": 5}` | True | True | 找到 0 首曲目。 | 通过 |
| tool-05 | find_similar_tracks | `{"track_id": "e1c0468c81291ceb3d79", "limit": 3, "filters": null}` | True | True | 找到 3 首曲目。 | 通过 |
| tool-06 | find_similar_tracks | `{"track_id": "e1c0468c81291ceb3d79", "limit": 5, "filters": {"bpm_min": 60, "bpm_max": 220}}` | True | True | 找到 5 首曲目。 | 通过 |
| tool-07 | generate_dj_set | `{"request": "quality eval round 7 build", "duration_min": 10, "bpm_min": 60, "bpm_max": 220, "energy_curve": "build", "style_query": ""}` | True | True | 已生成 2 首、约 9 分钟的 Set。 | 通过 |
| tool-08 | generate_dj_set | `{"request": "quality eval round 7 scoped", "duration_min": 10, "bpm_min": 60, "bpm_max": 220, "energy_curve": "steady", "style_query": "", "track_ids": ["040d8aecd4b178302f91", "c905b7a001da1a61945c", "482792d0e401f7a09ea2", "d92887a5d63be97e5351", "252c1f7d28ed1ead6f0d", "5e146bf976728898b357", "7e3ac38d08fafc30b836", "8be4e1da6336d64698a3", "416b57d312cf3e3ecf16", "668fc1ecb6b4b414c6c0", "8e656e9bafae84ab376d", "a8df75fc21f64d3dc7ea", "bde89636f09f1ccb27ab", "ca5015bf4301c9e18a0d", "e1556d1e22847958ac6f", "e6a6a00281a127875d96", "43ab50199dc212585f04", "39d806aa331d1ba936f8", "7ab79c6cc8e754f9506b", "816887e3f689b6883ca8"]}` | True | True | 已生成 3 首、约 10 分钟的 Set。 | 通过 |

本轮工具结果：`8/8`，成功率 `100.00%`。

#### D. RAG 链路

`选定 gold 歌曲 → 读取已索引描述 → DashScope 重新编码查询 → 项目向量读取 → 余弦 Top-3 → Hit@3 → 返回事实逐字段对账`

| ID | Gold | Top-3（排名顺序） | Gold 排名 | 命中 | 事实核验 | 错误 |
| --- | --- | --- | --- | --- | --- | --- |
| rag-01 | Colorblind (`6d79b5100546acb24081`) | 1. Colorblind → 2. Sideways → 3. Good News (feat. Kyle Reynolds) | 1 | True | 36/36 | — |
| rag-02 | When I Fall (`723358e15ced6ba76711`) | 1. When I Fall → 2. Into Pieces (Wooli x Grabbitz Remix) → 3. Tomorrow | 1 | True | 36/36 | — |
| rag-03 | Evergreen (`737e09afd845c28dc356`) | 1. Evergreen → 2. Going Down → 3. Walk On Water (Wooli & Trivecta Remix) | 1 | True | 36/36 | — |
| rag-04 | Not My Night (`748a4934059d57903f44`) | 1. Not My Night → 2. Alaska → 3. Evergreen | 1 | True | 36/36 | — |
| rag-05 | Stardust (with HALIENE) (`7553a2ed6fb640ca2ca5`) | 1. Stardust (with HALIENE) → 2. Tomorrow → 3. Evergreen | 1 | True | 36/36 | — |

本轮 RAG：Hit@3 `100.00%`，MRR `100.00%`，真实性 `100.00%`。

## 复现

```powershell
python -m backend.evals.run_quality_eval --rounds 7
```

需要当前项目依赖、可访问的 Ollama、`DASHSCOPE_API_KEY`，以及 `data/dropit.db` 与 `data/chroma`。脚本不会输出任何密钥。网络或模型服务失败会作为该轮失败记录进入 JSON，而不会被静默跳过。

## 解释边界

- 本次选择向量歌曲最多的项目 `md`（`74` 首）执行工具与 RAG；向量入库总数按全局有效交集统计。
- 意图测试集是人工编写的平衡集，不是线上流量分布；因此 Accuracy 适合做版本回归，不应直接外推为真实用户总体准确率。
- 工具成功率只纳入合法请求。越权 ID、非法范围等请求被正确拒绝属于安全测试通过，不应混入合法调用成功率分母。
- RAG 真实性核验的是 Tool 返回证据，不等同于任意生成式回答的事实正确率。
