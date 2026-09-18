# 企业文档 RAG 全链路拆解：从一份 PDF 到一条可追溯答案

> 版本：2026-09-18 · 配套仓库：`github.com/wabcyyyy/quant-rag` · 唯一事实源：`PLAN.md`（v0.9）
>
> 这是一份**教学文档**，不是 API 参考。它会回答三个问题：系统长什么样、每个部件为什么这么选、以及你如何从零把它复现出来。文中所有数字都来自仓库内可复现的实验，原始结果在 `data/eval/`（含公司内容，不入库），命令见 `PLAN §6`。

---

## 导读：这个项目在解决什么问题

它面向的不是百科式纯文本，而是**公司会议记录类文档**：飞书批量导出的 PDF、老格式 `.doc`、`.docx`，混着表格、扫描页、碎化文本层和大量“会上讨论了但没定”的模糊表述。

**一句话概括架构**：

```text
          离线（一次性）                              在线（每题）
┌──────────────────────────┐              ┌──────────────────────────┐
│ PDF / doc(x) 多源接入     │              │ QueryRewriter（零 LLM）    │
│   ├─ 碎化行重建           │              │   ├─ 实体 / 时间识别       │
│   ├─ find_tables 表格     │              │   ├─ 聚合 / 过滤意图       │
│   └─ 统一中间 JSON        │              │   └─ 改写                 │
└───────────┬──────────────┘              └───────────┬──────────────┘
            │                                         │
            ▼                                         ▼
┌──────────────────────────┐   RRF 融合   ┌──────────────────────────┐
│ 结构感知分块 + 元数据     │ ───────────▶ │ HybridRetriever          │
│ Embed → Qdrant 双索引     │              │ Dense + BM25 + 重排       │
└──────────────────────────┘              └───────────┬──────────────┘
                                                       │ 取 top-6
                                                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ Synthesizer：强制引用 [n] + 拒答 + 分题型思考分流                  │
│ 答案 [1] → （文档名 第p页） → Qdrant payload（doc_id/page/block）  │
└──────────────────────────────────────────────────────────────────┘
```

**为什么这个项目值得讲**：功能上“能问答”只是及格线；真正的差异化在**评估闭环**——7 题型黄金集、5 组消融、RAGAS 双轨判分、噪声地板口径、以及两次推翻自己结论的度量复盘。这些内容在第 6、7 章展开。

---

## 第 1 章 五分钟上手：clone 即可端到端体验

仓库自带一套**虚构公司「云帆科技」的合成会议纪要**（10 篇 PDF，232KB），与脱敏样例黄金集 `data/eval/golden_sample.json` 的 `source_doc_ids` 互相咬合。不需要任何真实语料，也不需要 API 之外的额外依赖。

### 1.1 前置条件

| 项 | 要求 |
|---|---|
| Python | 3.12（`.python-version` 钉死，勿升） |
| 依赖管理 | `uv`（`uv sync` 自动按 `uv.lock` 锁定版本） |
| 向量库 | Qdrant ≥ 1.15（内置 `qdrant/bm25` 稀疏模型） |
| 密钥 | DeepSeek 官方 LLM + SiliconFlow Embedding（`.env` 已 gitignore） |

### 1.2 四步跑通

```bash
# 1. 启动 Qdrant（仅依赖 docker-compose.yml，不依赖本机安装）
docker compose up -d

# 2. 安装依赖（核心依赖 + dev 组；eval/demo 是独立 extra）
uv sync --frozen

# 3. 把示例语料灌入独立 collection（--parsed-dir 隔离解析产物，避免误灌全量）
uv run doc-rag ingest  --raw-dir data/sample_raw --parsed-dir data/sample_parsed \
                       --kb doc_rag_sample --recreate

# 4. 跑评估（10 条全真实调用，约 ¥0.02；--fresh-answers 关缓存，测真实延迟）
uv run doc-rag eval    --kb doc_rag_sample --gold data/eval/golden_sample.json \
                       --rewrite --rerank --fresh-answers
```

**预期结果（2026-09-18 实测）**：

```text
入库：10 篇 / 10 块 → Qdrant[doc_rag_sample]
Recall@5 : 1.0    Recall@8 : 1.0    MRR : 1.0    nDCG@8 : 1.0
包含匹配准确率 : 1.0    拒答正确率 : 1.0    引用有效率 / 存在率 : 1.0 / 1.0
端到端 p95 ≈ 4~5s（目标 ≤ 8s，达标）
```

**注意**：10 篇小库满分只背书「管线正确、判分口径可跑通」，不代表真实语料难度；真实语料上的结论见第 7 章。

### 1.3 启动演示页

```bash
uv run --extra demo doc-rag demo --kb doc_rag_sample   # Gradio 连示例库
```

演示页包含：问题输入、示例问题下拉、答案 Markdown（引用编号）、引用列表（文档名/页码/块类型）、分阶段延迟面板。

---

## 第 2 章 系统架构总览

### 2.1 分层与职责

| 层 | 模块 | 职责 | 关键约束 |
|---|---|---|---|
| 接入层 | `ingest/pdf.py` `ingest/office.py` `ingest/pipeline.py` | 多源归一为统一中间 JSON | 逐文件容错、sha256 去重、断点续跑 |
| 存储层 | `ingest/indexer.py` | 分块 → Embed → Qdrant 双索引 upsert | `--parsed-dir` 可隔离目录 |
| 检索层 | `retrieve/hybrid.py` `retrieve/rewrite.py` `retrieve/rerank.py` | 改写 → 双路检索 → RRF → 重排 | 改写零 LLM；聚合题自动放宽预算 |
| 合成层 | `generate/llm.py` `generate/prompts.py` | 强制引用 + 拒答 + 分题型思考分流 | prompt 版本可追踪；缓存键含版本 |
| 评估层 | `eval/runner.py` `eval/goldgen.py` `eval/ragas_runner.py` | exact-match 与 RAGAS 双轨判分 | 噪声地板 1.4pt；结果文件自证模型/缓存/版本 |
| 接口层 | `api/main.py` `api/demo.py` `cli.py` | FastAPI（含 SSE 流式）+ Typer CLI + Gradio | 同缓存键互通 |

### 2.2 在线 vs 离线双轨

- **在线（每题）**：改写 → 检索 → 重排 → 合成 → 返回答案 + 引用 + 分阶段延迟。
- **离线（一次）**：接入 → 分块 → Embed → 入库 → 黄金集生成 → 评估 → 消融对比。
- 两条链路共用同一套 `HybridRetriever` 与 `Synthesizer`，保证「线上表现」和「评估口径」是同一个系统，而不是两套实现。

### 2.3 为什么自研薄编排，不引编排框架

重型轮子全部用现成库（Qdrant / BGE-M3 / PyMuPDF / RAGAS），定制层自己写：

1. **可观测性**：每个阶段耗时、token、缓存命中、prompt 指纹都落在结果文件里，框架黑盒会遮住这些。
2. **评估可复现**：prompt 版本、judge 上下文、模型身份全部可追踪，框架默认行为变一次，数字就废一次。
3. **成本纪律**：批量评估先 3 条试算再外推，超出预算即停（详见第 8 章）。

---

## 第 3 章 接入层：把异构文档归一化

### 3.1 文件路由与统一中间表示（IR）

`ingest/pipeline.py` 按扩展名路由到对应解析器，输出同一套 `IntermediateDoc`：

```jsonc
// data/parsed/<doc>.json（示意）
{
  "meta": {
    "source_type": "pdf",
    "doc_id": "a27cfc6af6ea27e7",   // 文件 sha256 前 16 位
    "title": "会议档案_办公会_2026年第42周-会议纪要",
    "doc_date": "2026-04-13"
  },
  "blocks": [
    { "type": "heading", "heading_level": 1, "text": "二、议题讨论", "page": 3 },
    { "type": "paragraph", "text": "……", "page": 3 },
    { "type": "table", "text": "| 项目 | 金额 | ……", "page": 3 }
  ]
}
```

**设计要点**：下游分块、入库只认 IR，不认来源。PDF 和 docx 的差异在接入层消化，检索层永远面对同一种结构。

### 3.2 PDF 快通道：PyMuPDF + 碎化行重建 + find_tables

飞书导出的 PDF 里约 4.5% 的文档文本层是「每字符一行」，直接分块会把句子切成单字。`ingest/pdf.py` 做三件事：

1. **碎化行重建**：按纵向间距合并相邻行，再逐字 grounding 校验，违规从 30/44 降到 0。
2. **带框表格重建**：`find_tables` 把表格还原为单个 `table` 块，整块不切。
3. **页码与 bbox 保留**：每个 block 带 `page`/`bbox`，引用才能落到「第 p 页」。

**实测**：结构感知分块的表格完整率 **100%**，固定 512 切分只有 **52.5%**。

### 3.3 docx 通道：LibreOffice headless → mammoth

老格式 `.doc` 先由 LibreOffice 归一为 docx，再走 mammoth 保留标题层级。导出产物同样归一为 IR，后续流程零差异。

### 3.4 结构感知分块：检索块 ≠ 合成块

- **分块策略**：300–600 字、按标题层级切分、表格整块不切（`chunker.py`）。
- **关键取舍**：小块召回率高，但上下文太碎；所以检索取 top-12 → 重排取 top-6 → `max_contexts` 上限控制送 LLM 的上下文量。
- **实测**：固定 512 切在 Recall@8 上反而略优（0.982 vs 0.945），但表格完整率 52.5%；修正口径后 Faithfulness 无差异（0.5pt < 1.4pt 噪声地板）。**结构感知的价值在表格完整性与元数据，不在召回**——这是消融 #2 的结论。

### 3.5 元数据与 doc_id

- `doc_id = sha256(文件)[:16]`：内容级去重，同文件重复入库自动跳过。
- 文件名正则解析大类/分组/日期；正文头部再扫一次日期兜底。
- **doc_date 覆盖率只有 ~19%**（会议档案 114/384 有完整日期）——这是时间过滤必须查覆盖率的根因，否则普通事实题加过滤会误伤 80% 语料。

---

## 第 4 章 检索层：Hybrid 为什么必选

### 4.1 纯 Dense 搞不定什么

公司文档里有大量**文号、专名、精确条款**（如「第 42 周」「SCP 二期」「5 万元以下」），dense 向量对这些精确匹配不敏感。消融 #1 给出直接证据：

| 指标 | 纯 Dense | Hybrid（默认） |
|---|---|---|
| Recall@5 | 0.836 | **0.909** |
| Recall@8 | 0.891 | **0.945** |
| MRR | 0.764 | **0.794** |

增益集中在精确匹配类题型：decision 0.75→0.88、term 0.75→0.83、time_filter 0.80→1.00。

### 4.2 双路检索 + RRF 融合

```text
QueryRewriter（零 LLM）
  ├─ 实体识别 → 实体聚焦改写
  ├─ 时间识别 → 年份过滤（仅聚合题启用）
  └─ 聚合意图 → 放宽检索预算（top_n 25）
        │
        ├─ Dense（BGE-M3，1024 维）
        └─ BM25（Qdrant 内置 qdrant/bm25 + jieba 预分词）
                │
                ▼
        FusionQuery：RRF 融合（k=60）→ 双路 prefetch
                │
                ▼
        BGE Reranker → 取 top-6 送合成
```

**RRF 为什么是 k=60**：平衡两路权重，避免 dense 或 BM25 任一路独大；这是消融实验后定稿的参数，不是拍脑袋。

### 4.3 改写器：零 LLM，规则先行

`retrieve/rewrite.py` 只做三件事：

1. **实体聚焦**：把「客服系统升级」这类宽泛问法改写成带实体约束的查询。
2. **时间过滤**：只有「明确时间限定的聚合题」才加 `doc_date` 过滤（覆盖率仅 19%，对普通事实题是毒药）。
3. **聚合意图**：命中「有哪些记录 / 都讨论了什么 / 哪些文档」等模式时，放宽检索预算到 top-25，按文档去重。

**收益**：改写零成本、零延迟，且行为完全可解释——这是延迟优化的第一道防线。

### 4.4 重排：榜首精度换上下文长度

无重排 vs 有重排（v2 口径 72 条）：MRR 0.804→0.842、Recall@5 0.891→0.922，但 **nDCG@8 0.808→0.810（≈持平）**。

**解释**：重排把 top-6 削给 LLM，nDCG 数到的第 7/8 位被裁掉——增益在榜首精度，不在列表级。所以重排的定位是「给合成层更干净的上下文」，不是「提升列表级指标」。

---

## 第 5 章 合成层：强制引用 + 拒答 + 分题型思考分流

### 5.1 上下文组装

重排后的 top-6 块按 `max_contexts` 上限组装，每块带完整前缀：

```text
（文档名 第p页）……正文……
```

**这个前缀是 Faithfulness 的生命线**：judge 必须拿到和合成模型**逐字一致**的上下文，否则会把「答案里提了文档名」误判为不忠实（见第 6 章的度量伪影复盘）。

### 5.2 prompt 契约

`generate/prompts.py` 的 SYSTEM_ANSWER 强制三件事：

1. **答案必须带 [n] 引用**，且每个引用能回落到 payload 的 `doc_id + page + block_type`。
2. **上下文没有就拒答**，不得编造；拒答语固定风格（「根据现有文档无法回答……」）。
3. **只依据原文**，不引入外部知识。

消融 #4（30 条同题对照）：引用存在率 **0.067 → 1.00**，且顺带把包含匹配 +6.7pt——要求标注来源会促使模型贴原文而非自由改写。

### 5.3 拒答判定

`eval/runner.py` 的 `_refusal_ok()` 判断答案是否包含拒答语；no_answer 题（黄金集 8 条）要求「正确拒答且未编造 must_contain」。拒答正确率实测 **1.00**。

### 5.4 分题型思考分流（P95 目标的由来）

端到端 p95 曾高达 32.8s，根因是 deepseek-flash 输出里 **91% 是看不见的 reasoning token**，聚合题跨 15~25 块归纳，思考量爆炸。

**唯一有效杠杆：聚合题关合成侧思考**（`DOC_RAG_LLM_REASONING_EFFORT_AGGREGATE=none`）：

| 题型 | 端到端均值 | 包含匹配 | Faithfulness |
|---|---|---|---|
| 聚合题（开思考） | 27.7s | 1.00 | 0.961 |
| 聚合题（关思考） | **2.7s** | **1.00** | **0.979** |
| 短答题（保留思考） | 2.2~5.3s | 质量优先 | — |

**拍板结论**：聚合题关思考、其余保思考（decision / open_discussion 最依赖思考，全关思考包含匹配 −10.9pt）。P95 ≤ 8s 目标改写为**分题型 SLO**：聚合 ≤5s ✓ / 短答 ≤8s。

---

## 第 6 章 评估闭环：这个项目真正的护城河

### 6.1 黄金集：7 题型 72 条（v2）

构造方式（`eval/goldgen.py`）：

| 题型 | 数量 | 构造方式 |
|---|---|---|
| fact | 12 | 分层采样文档 + LLM 生成（答案须逐字能在原文找到） |
| decision | 10 | 程序化构造（决议区提取，零 LLM 成本） |
| open_discussion | 10 | LLM 生成，强制「未形成决议」表述 |
| term | 12 | 专名 / 编号 / 制度名，测精确召回 |
| cross_doc | 8 | 程序化找跨 ≥3 篇文档复现的关键词 |
| time_filter | 12 | 年份 × 高频词组合（v1 仅 5 条是已知弱点） |
| no_answer | 8 | 验证全库不存在的域外主题，测拒答 |

**脱敏样例**：`data/eval/golden_sample.json`（虚构公司内容，10 条覆盖 7 题型）与 `data/sample_raw/` 的 10 篇合成会议纪要 doc_id 咬合，clone 即可复现。

### 6.2 双轨指标

- **exact-match 轨**（确定性、可复现）：Recall@5/8、MRR、nDCG@8、包含匹配、拒答正确率、引用有效率/存在率、文档覆盖率。
- **RAGAS 轨**（语义判分）：默认只开 `faithfulness`；`answer_relevancy` 实测 0.39 但对中文拒答/未决表述有结构性惩罚，**降级为诊断信号，不进质量门禁**。

### 6.3 统计口径：噪声地板与分题型

- **噪声地板 1.4pt**：同条件全量重跑两轮 Faithfulness 0.9563 / 0.9423，差异小于 1.4pt 不下结论。
- **分题型报 n**：聚合题 n 小（cross_doc 8、time_filter 12），p95 基本等于 max，必须同时报 n。
- **抽样均匀覆盖**：RAGAS 抽样改为均匀取各题型，不再取前 N 条（会整段漏掉末尾题型）。
- **结果文件自证**：`meta.llm_model` / `meta.answer_cache` / `meta.prompt_version` / `meta.prompt_fingerprint` 全部落盘，防止「数字没有出处」。

### 6.4 两次度量伪影复盘（诚实度拉满）

**伪影 1：Faithfulness 0.77 → 0.9563**
- 根因：judge 拿到的上下文缺了「（文档名 第p页）」前缀，而合成 prompt 要求标注来源文档名 → 答案里「《某文档》中…」被判不忠实。
- 修复：把 LLM 实际看到的上下文原样交给 judge，同一批答案 0.6375 → **0.9563**，消融 #2 的 18pt 假差距归零。

**伪影 2：「prompt 收紧 +14.8pt」被三组对照重跑推翻**
- 根因：结果文件与 prompt 版本对应关系断裂，且 judge 上下文缺文档名。
- 重跑（2026-09-18）：基线/收紧/收紧+重排 = 0.9483 / 0.9431 / 0.9437，配对差 −0.5pt（p>0.55）——**无差异**。
- 配套修复：`ANSWER_PROMPTS` 注册表 + `prompt_version` 开关 + 结果文件 `prompt_fingerprint`。

**伪影 3：延迟数字曾是笔糊涂账**
- 根因：仓库里没有任何延迟测量代码，文档里同时存在「≈6.3s 达标」和「1.3s ping」两个矛盾的数。
- 修复：`chat_timed()` + `Synthesizer.last_meta` + runner 分阶段计时 + `--fresh-answers`（缓存命中的毫秒数是本地查询耗时，不是模型延迟）。

---

## 第 7 章 消融实验与关键数字

| 实验 | 对照 | 结论 |
|---|---|---|
| #1 检索 | 纯 Dense vs Hybrid | 召回集中提升，精确匹配类收益最大 |
| #2 分块 | 固定 512 vs 结构感知 | 召回略亏，表格完整率 100% vs 52.5%；Faithfulness 无差异 |
| #3 重排 | 无 vs 有 | 榜首精度 +3.8pt，nDCG@8 ≈持平，增益在榜首 |
| #4 引用约束 | 无 vs 有 | 引用存在率 0.067→1.00，包含匹配 +6.7pt |
| #5 元数据过滤 | 时间限定聚合题 | 文档覆盖率 0.61→0.96；普通事实题加过滤会误伤 |

**回答质量（产品路径：改写 + 重排 + 收紧 prompt + deepseek-flash）**：

| 指标 | 数值 |
|---|---|
| 包含匹配准确率 | 0.875（v2 口径 72 条） |
| 拒答正确率 | 1.00 |
| 引用有效率 / 存在率 | 1.00 / 1.00 |
| RAGAS Faithfulness | 0.95（全量 55 条 · 修正口径） |
| 检索延迟 | p50 423ms · p95 509ms |
| 端到端延迟 | 聚合 ≤5s ✓ / 短答 ≤8s（分题型 SLO） |

---

## 第 8 章 复现指南与成本纪律

### 8.1 环境钉子（踩坑换来的，勿随意升级）

| 项 | 定稿 | 原因 |
|---|---|---|
| Python | 3.12 | 3.14 的 asyncio 会让 ragas 0.3.1 执行器静默全失败返回 NaN |
| langchain | 0.3.x | 1.x 移除了 ragas 依赖的 `chat_models.vertexai` |
| ragas | 0.3.x + pillow | ragas 0.3.x 未声明但 multi_modal_prompt 需要 pillow |
| Qdrant | ≥ 1.15 | 内置 `qdrant/bm25` 稀疏模型 |
| uv 源 | 清华镜像 | 国内直连 PyPI 超时 |

### 8.2 成本纪律

1. **先试算**：每个付费任务先跑 3 条，按实际 token 外推全量，超出预算即停。
2. **模型与任务匹配**：合成用 deepseek-flash；judge 关思考；embedding/rerank 用 SiliconFlow BGE-M3。
3. **缓存**：本地响应缓存使重复评估近乎零成本（实测命中 90%）；但**测延迟必须 `--fresh-answers` 关缓存**。
4. **参考成本**：judge 全量一轮 ≈¥0.5；合成一轮 63 条 ≈¥1.3；示例语料端到端 ≈¥0.02。

### 8.3 结果文件说明

- `data/eval/results_*.json`：逐条评估结果（含公司内容，gitignore）。
- `data/eval/gold.json`：黄金集 v2（72 条）。
- `data/eval/golden_sample.json`：脱敏样例（入库）。
- `.cache/llm_cache.sqlite`：LLM 响应缓存（本地，可随时删）。

---

## 第 9 章 常见坑与排障

### 9.1 401 / 429 / 404

- **401**：Key 复制不全（智谱 key 曾出现 62 字符、49 字符两种残缺）。
- **429**：OpenRouter 免费档限 50 请求/天（评估循环跑不完）；DeepSeek 错峰折扣时段（北京时间 00:30–08:30）跑批量。
- **404**：base_url 误填 `/api/v1/responses`；代码已归一化容忍，但建议直接写官方 base_url。

### 9.2 延迟测出来「特别低」

- 缓存命中的毫秒数是本地查询耗时，不是模型延迟。**测真实延迟必须 `--fresh-answers`**。
- `doc-rag check` 的 ping 是无上下文连通性探测，≠ 合成延迟；合成延迟看 `doc-rag eval` 的分阶段摘要。
- 延迟是双峰分布：报均值会同时低估短答题、藏起聚合题尾部，必须分题型报 p50/p95/max。

### 9.3 入库把全量语料误灌进目标 collection

**这是示例语料首跑真实踩到的坑**：`index_parsed` 按 parsed 目录**全量 upsert**，CLI 原本不能覆盖 parsed 目录——把 `data/parsed` 的 1131 篇公司语料中间 JSON 一起灌进了示例 collection（仅本机内污染，未出域）。

**修复**：`ingest --parsed-dir` 参数，隔离目录必须与 `--raw-dir` 同时指定：

```bash
uv run doc-rag ingest  --raw-dir data/sample_raw --parsed-dir data/sample_parsed \
                       --kb doc_rag_sample --recreate
```

### 9.4 评估数字「忽高忽低」

- judge 上下文与合成 prompt 是否逐字一致（缺文档名前缀 → Faithfulness 假低）。
- 抽样是否均匀覆盖题型（取前 N 条 → 整段漏掉末尾题型）。
- 重排 API 对近似并列项会抖动（63 条里有 3~14 条上下文变化），这是**非位级可复现**，两次重跑的块集合都会变，不是 bug。

### 9.5 字体与 CI

- 示例语料 PDF 用系统 CJK 字体（Windows simhei / Linux Noto CJK / macOS PingFang），脚本自动探测；字体子集化后 10 篇共 232KB。
- CI 只装核心依赖 + dev 组；依赖 `eval` extra（langchain）的 3 个测试用例用 `pytest.importorskip` 显式跳过，避免 CI 装 langchain 全家桶。

---

## 第 10 章 模块索引与扩展建议

### 10.1 文件到职责速查

| 文件 | 职责 |
|---|---|
| `src/doc_rag/ingest/pdf.py` | PDF 解析、碎化行重建、find_tables |
| `src/doc_rag/ingest/office.py` | docx 转换与解析 |
| `src/doc_rag/ingest/chunker.py` | 结构感知分块 |
| `src/doc_rag/ingest/indexer.py` | 分块 → Embed → Qdrant upsert |
| `src/doc_rag/retrieve/hybrid.py` | Dense + BM25 + RRF 融合 |
| `src/doc_rag/retrieve/rewrite.py` | 零 LLM 改写、实体/时间识别、聚合意图 |
| `src/doc_rag/retrieve/rerank.py` | BGE Reranker |
| `src/doc_rag/generate/prompts.py` | 合成 prompt 注册表与指纹 |
| `src/doc_rag/generate/llm.py` | OpenAI 兼容 client、缓存、token 统计 |
| `src/doc_rag/eval/goldgen.py` | 黄金集构造（LLM + 程序化） |
| `src/doc_rag/eval/runner.py` | exact-match 评估与指标计算 |
| `src/doc_rag/eval/ragas_runner.py` | RAGAS 判分 |
| `src/doc_rag/api/main.py` | FastAPI（/health、/query、/query/stream SSE） |
| `src/doc_rag/api/demo.py` | Gradio 演示页 |
| `scripts/make_sample_corpus.py` | 合成示例语料生成（含三重自检） |

### 10.2 Phase 3 可选方向

- 全量压测（ingest 吞吐 / 查询 P95）
- 图片 caption 入库（~15 份真扫描件走 MinerU 兜底）
- GraphRAG / 结构化索引（聚合题文档覆盖率受检索预算数学封顶，top_n / 答案集大小）
- 简单反馈回路（点赞入评估集）

---

## 结语：怎么把这个项目讲清楚

面试时不要只说「我做了个 RAG」，按这条线讲：

1. **问题**：公司会议文档异构、表格多、聚合题多、doc_date 稀疏。
2. **架构**：接入归一 → Hybrid 检索 → 强制引用合成 → 双轨评估。
3. **取舍**：为什么 Hybrid、为什么结构感知分块、为什么聚合题关思考。
4. **验证**：5 组消融 + 噪声地板口径 + prompt 版本指纹。
5. **诚实**：主动讲两次推翻自己结论的度量伪影，以及 `--parsed-dir` 误灌全量这个坑。

**仓库里每条结论都能复现**：命令见 `PLAN §6`，结果文件含公司内容不入库，commit 历史按决策粒度提交。
