# 现行系统设计（架构手册）

> 读者：本项目架构师/维护者本人。  
> 目的：改代码前先看清**现在系统怎么跑**，而不是从实验日志里拼。  
> 证据与历史推翻：见 [PLAN](../design/PLAN.md)。概览与流程图：见 [ARCHITECTURE](ARCHITECTURE.md) 与 [diagrams/](diagrams/)。  
> 口径日期：2026-09-19（以仓库源码与 `configs/default.yaml` 为准）。

---

## 0. 怎么用这份手册

| 你要做的事 | 先看 |
|------------|------|
| 搞清楚用户一次提问发生了什么 | §2 在线主流程 |
| 搞清楚文档怎么进库 | §3 离线主流程 |
| 改某块逻辑 | §4 模块地图 + §8 定位表 |
| 改配置/开关 | §7 开关语义 |
| 弄清数据结构 | §6 数据契约 |
| 评估、延迟、消融数字为什么那样 | 不在本手册展开 → PLAN |

---

## 1. 系统是什么

**一句话**：把公司会议纪要（飞书/Word 批出的 PDF / docx，手动放进 `data/raw`）做成可检索问答库：问一句 → 系统检索相关片段 → 用 LLM 生成**带编号引用**的答案；上下文不够时**拒答**。

**业务边界（刻意不做）**

| 边界 | 现状 |
|------|------|
| 语料来源 | 只认本地 `data/raw`，不对接飞书/OpenAPI |
| 权限模型 | 无多租户；API 只有单一 `DOC_RAG_API_TOKEN` |
| 多模态 | 扫描件/图片 caption 未入库 |
| 编排框架 | 自研薄编排，不引 LangChain/LlamaIndex 做主路径 |
| GraphRAG | 未做（Phase 3 方向） |
| 决议时效/版本 | **无任何时效建模**：检索无 recency 项，也没有「旧决议被后续会议推翻」这类题型——问「现在有效的是哪版」目前不在系统能力内 |
| 索引对账 | 只做「同 `doc_id` 删旧再插」；源文件**改名或删除**后旧点永久留在 collection，没有清理路径 |
| 并发 | 只有进程内 `rate_limit_rpm` + `/metrics`；延迟全部测于并发 = 1（Phase 3 全量压测未做） |

**四条交付入口（必须共用同一管线）**

1. CLI：`doc-rag query`
2. FastAPI：`POST /query`、`POST /query/stream`
3. Gradio：`doc-rag demo`
4. 评估：`doc-rag eval`

装配点**只有一个**：`src/doc_rag/orchestrator.py` 的 `Orchestrator`。禁止在入口层再手拼 rewrite→retrieve→rerank→synthesize。

---

## 2. 在线主流程（每题）

```text
用户问题
   │
   ▼
[入口] CLI / API(鉴权+限流) / Gradio / eval
   │
   ▼
Orchestrator.answer() 或 answer_stream()
   │
   ├─ 1. 改写 plan（可关；失败退化）
   ├─ 2. HybridRetriever.retrieve()  → retrieved（未截断清单）
   ├─ 3. Rerank（可关；失败退回融合顺序）→ 仍是一份完整排序清单
   ├─ 4. 上下文截断 + 生成 citations
   └─ 5. Synthesizer（用**原始问题** + contexts）
   │
   ▼
Result {answer, citations, plan, latency_ms, ...}
   │
   ├─ 非流式：JSON / CLI 文本
   └─ 流式 SSE：rewrite → delta* → citations → done
```

### 2.1 步骤明细

| 步 | 做什么 | 输入 | 输出 | 失败/旁路 | 代码 |
|----|--------|------|------|-----------|------|
| 1 改写 | LLM 判意图：是否聚合、是否年份、检索串是否实体聚焦 | `question` | `plan` | 关闭改写 / 超时 / JSON 坏 → `degraded=true` 的 noop plan | `retrieve/rewrite_llm.py` |
| 2 检索 | Dense + BM25 双路 prefetch → Qdrant RRF；聚合题按 doc_id 去重；过滤结果过少则回退 | `plan.rewritten`, filters, aggregate, top_n | `RetrievalOutcome.chunks` = **retrieved** | 过滤后 `< min(3, limit)` → 去掉过滤重试并记 `filter_fallback` | `retrieve/hybrid.py` |
| 3 重排 | 调 Rerank API，返回**全量重排**清单（不截断） | rewritten + retrieved | 重排后的完整列表 + `context_budget` | API 异常 → 保持融合顺序，`rerank_error` 必须可见 | `retrieve/rerank.py` |
| 4 截断 | 进 LLM 的块数 = `min(max_contexts, rerank.top_n 若开重排)` | 重排后清单 | `contexts` + `citations` | `retrieved` **不被截断覆盖**（指标分母） | `orchestrator.py` `_prepare` |
| 5 合成 | 强制 `[n]` 引用 + 无据拒答；聚合题可关思考 | **原始问题** + contexts | 答案文本 | 空上下文：CLI 可 `stop_on_empty`；eval 不 stop（拒答本身被测） | `generate/synthesizer.py`, `prompts.py` |

### 2.2 API 行为（改服务前必读）

文件：`src/doc_rag/api/main.py`

| 端点 | 鉴权 | 说明 |
|------|------|------|
| `GET /health` | 无 | 探针 |
| `GET /metrics` | 要令牌 | Prometheus 文本：请求量、分阶段 p50/p95、改写/重排失败计数、过滤回退计数、流式断连计数 |
| `POST /query` | 要令牌 | 完整管线同步返回 |
| `POST /query/stream` | 要令牌 | SSE 事件流 |

- `auth_token` 为空 → `/query*` **一律 503**（fail-closed，不是裸奔）。
- `kb` 只能在 `api.allowed_collections` 白名单内。
- `rate_limit_rpm` 进程内滑动窗口（默认 30）；慢推理可占满线程池，必须限流。
- **故意不开 CORS**：令牌接口；要 Web UI 走同源反代或 Gradio。

### 2.3 CLI 问答开关

```bash
uv run doc-rag query "问题" [--kb NAME] [--top-n N] [--no-rewrite] [--no-rerank] [--stream] [--timing]
```

- `--no-rerank` → `use_rerank=False`（覆盖配置）
- 不传 → `use_rerank=None` → 跟随 `rerank.enabled`
- `--timing`：分阶段延迟；**缓存命中时合成 ms 不是模型延迟**

---

## 3. 离线主流程（入库，一次）

```text
data/raw/*.pdf|*.docx|*.doc
        │
        ▼
pipeline.run()  按扩展名路由 + sha256 去重 + 逐文件容错
        │  doc_id = sha256[:16]
        ▼
data/parsed/<doc_id>.json   ← IntermediateDoc（统一 IR）
        │
        ▼
chunker：结构感知分块（标题层级；表格整块不切；约 300–600 字）
        │
        ▼
元数据：默认 base_meta（文件名启发式）；--llm-meta 才 LLM 抽取
        │
        ▼
Embedder：BGE-M3 dense 1024d
BM25 文本：jieba 预分词空格拼接
        │
        ▼
Indexer：按 doc_id 删旧再 upsert（幂等）
        │
        ▼
Qdrant collection
  vectors: dense(COSINE) + bm25(qdrant/bm25, IDF)
  payload indexes: doc_id, category, doc_group, block_type, doc_date
                  (+ topics/attendees 仅在 use_llm_meta 时建)
```

> **两路索引喂进去的不是同一段文本**：dense 向量 = `section_path + "\n" + 正文`
> （`ingest/indexer.py` `_embed_text`，为同质语料加区分度），BM25 稀疏向量 = **只有正文**
> （同文件 `build_bm25_text(chunk.text)`）。所以 RRF 融合的是两个不同口径的排名，
> 且改任一侧输入都会同时动两路——这个不对称至今没做过消融。

### 3.1 双路解析

| 扩展名 | 解析器 | 要点 |
|--------|--------|------|
| `.pdf` | `ingest/pdf.py` PyMuPDF | 碎化行重建；`find_tables` 重建带框表格；保留 page/bbox |
| `.docx` / `.doc` | `ingest/office.py` | LibreOffice headless → mammoth；标题层级进 IR |

下游（分块/入库/检索）**只认 IntermediateDoc**，不认来源。

### 3.2 入库命令

```bash
uv run doc-rag ingest [--raw-dir DIR] [--parsed-dir DIR] [--kb NAME] [--recreate] [--limit N] [--llm-meta] [--parse-only]
uv run doc-rag backfill   # 只改 payload 元数据，不重新向量化
uv run doc-rag profile    # Phase0 语料画像
```

**踩坑锁**：示例语料必须同时指定 `--raw-dir data/sample_raw --parsed-dir data/sample_parsed --kb doc_rag_sample`，否则会把公司全量 parsed JSON 灌进示例 collection。

### 3.3 默认入库时元数据的真实覆盖

| 字段 | 默认是否填 | 备注 |
|------|------------|------|
| `doc_id, title, text, page, section_path, block_type` | 是 | 来自解析/分块 |
| `doc_date, category, doc_group, meeting_type` | 是（启发式） | `doc_date` 全库约 **19%** 有值 |
| `topics, attendees` | 默认空 | 必须 `--llm-meta`；默认不建 payload index |
| LLM 元数据抽取 | **默认关** | `metadata_extraction.enabled: false`；成本 1 次/文档 |

因此：**时间过滤只在「明确时间限定的聚合题」上启用**——普通事实题加 `doc_date` 过滤会误伤大半语料。

---

## 4. 模块地图（文件索引）

```text
src/doc_rag/
├── orchestrator.py          # 在线管线唯一装配点
├── config.py                # 配置加载（yaml + env: 前缀）
├── net.py                   # 统一 HTTP 重试判定（429/5xx/超时才重试）
├── log.py / metrics.py      # 结构化日志 / 进程内 Prometheus 指标
├── cli.py                   # typer 入口
├── ingest/
│   ├── schema.py            # Block / IntermediateDoc / Chunk
│   ├── pipeline.py          # 解析路由 + sha256 去重
│   ├── pdf.py / office.py   # 双路解析
│   ├── chunker.py           # 结构感知分块
│   ├── metadata.py          # base_meta + 可选 LLM 抽取
│   ├── embedder.py          # BGE-M3 dense
│   ├── bm25.py              # jieba 预分词文本
│   ├── indexer.py           # 幂等 upsert 到 Qdrant
│   └── profile.py           # 语料画像
├── retrieve/
│   ├── rewrite_llm.py       # LLM 查询改写（意图/聚合/年份）
│   ├── hybrid.py            # Dense+BM25+RRF + 聚合 + 过滤回退
│   └── rerank.py            # BGE Rerank（全量排序 vs 上下文预算分离）
├── generate/
│   ├── prompts.py           # system/user prompt + 版本注册表 + 指纹
│   ├── llm.py               # OpenAI 兼容 client、缓存、chat_timed
│   └── synthesizer.py       # 引用合成 / 拒答 / 聚合思考分流
├── eval/
│   ├── schema.py            # 黄金集字段契约
│   ├── runner.py            # exact-match 评估 + RAGAS 第二轨（judge 都在这）
│   ├── compare.py           # 配对差 + 噪声地板
│   └── goldgen.py           # 黄金集构造
└── api/
    ├── main.py              # FastAPI
    └── demo.py              # Gradio
```

配置：`configs/default.yaml`（密钥走 `env:`）。  
评估结果：`data/eval/*.json`（含公司内容，默认 gitignore）。

---

## 5. 现行控制流：开关与退化

### 5.1 总表

| 能力 | 默认 | 谁控制 | 关闭/失败时 | 可观测标志 |
|------|------|--------|-------------|------------|
| 查询改写 | **开**（生产路径） | CLI `--no-rewrite`；eval `use_rewrite` | noop plan：`rewritten=原问题`, `degraded=true` | `plan.degraded`, `/metrics` rewrite 失败, `meta.rewrite_degraded` |
| 元数据过滤 | 按改写结果 | 仅 **aggregate 且有 year** 才加 `doc_date` | 过滤结果过少 → 去掉过滤重试 | `filter_applied`, `filter_fallback`, `n_before_fallback` |
| 聚合检索 | 改写判 `aggregate` | 强制可用 `force_aggregate` | 非聚合：limit=fusion_limit(12) | plan.aggregate；聚合池 pool=50，按 doc_id 去重 |
| 重排 | **开** `rerank.enabled=true` | CLI `--no-rerank`；eval 显式 bool | 失败：**不降级跳过，退回融合顺序** | `rerank_error`；结果文件不得自称 +rerank |
| 上下文截断 | **总是发生** | `max_contexts`(10) 与 `rerank.top_n`(6) | 进 LLM 块数 = min(两者) | `context_budget`, `Result.contexts` |
| 合成缓存 | **开** `llm.cache=true` | `DOC_RAG_LLM_CACHE=0` / `--fresh-answers` | 命中时 ms=本地查询，不是模型延迟 | `latency.synth_cached`, `meta.answer_cache` |
| 合成思考 | 全局默认**开**（空=不覆盖） | `DOC_RAG_LLM_REASONING_EFFORT` | 设 `none` 砍延迟，会改答案、作废基线 | `synth_meta` |
| 聚合题思考 | 推荐 `none` | `DOC_RAG_LLM_REASONING_EFFORT_AGGREGATE` | 未配则回落全局 | 同上 |
| prompt 版本 | `tightened` | `llm.prompt_version` | `baseline` 仅实验对照 | `prompt_fingerprint` |
| LLM 元数据抽取 | **关** | `--llm-meta` / `metadata_extraction.enabled` | 文件名启发式 | ingest 对账打印 |
| API 鉴权 | 令牌空=拒绝 | `DOC_RAG_API_TOKEN` | 503 | — |
| judge 思考 | **关** `none` | `eval.judge.reasoning_effort` | 开思考则贵一个数量级 | token 计量 |

### 5.2 改写 plan 契约

`rewrite_llm.rewrite()` 返回：

```python
{
    "rewritten": str,  # 送检索的查询串（可能是实体聚焦后的短串）
    "filters": dict
    | None,  # 如 {"doc_date": {"gte": "2026-01-01T00:00:00", "lt": "2027-01-01T00:00:00"}}
    "aggregate": bool,
    "top_n": int | None,  # 聚合题 → retrieval.aggregate_top_n（默认 25）；否则 None
    "reason": str,  # 人可读依据
    "degraded": bool,  # True = 未真正改写（失败/未配置）；机器标志，不要嗅文案
}
```

**模型只输出** `{rewritten, aggregate, year, reason}`；`filters` / `top_n` 由代码根据配置拼装。  
改写独立预算：`rewrite.timeout_s=5`, `max_attempts=2`, `reasoning_effort=none`；`rewrite.model/base_url/api_key` 留空=继承 `llm`。

### 5.3 检索参数（默认）

| 键 | 值 | 含义 |
|----|-----|------|
| `k_dense` / `k_bm25` | 20 / 20 | 双路 prefetch 候选 |
| `fusion_limit` | 12 | 非聚合题融合后块数 |
| `aggregate_pool` | 50 | 聚合检索候选池（再去重） |
| `aggregate_top_n` | 25 | 聚合题检索预算 |
| `max_contexts` | 10 | 进 LLM 上限 |
| `rerank.top_n` | 6 | **仅上下文预算**，不截断检索清单 |
| `retrieval.mode` | `hybrid` | `dense` 仅消融对照 |

**两条预算规则最容易读错**：

1. 生产路径不传 `top_n` → 聚合题听改写的建议，走 `aggregate_top_n`(25)；而 `eval` 默认
   显式传 `--top-n 8`，把它压掉（为的是各消融臂同预算可比）。所以**改 `aggregate_top_n`
   不会让任何默认 `eval` 数字动**；要量生产口径必须加 `--honor-rewrite-budget`。
2. 聚合题取回 25 块之后，进 LLM 的仍是 `min(max_contexts, rerank.top_n)` = **6 块**。
   放宽检索预算只改善「检回来多少」（实测文档覆盖率 cross_doc 0.2086→0.5018、
   time_filter 0.3020→0.6888），**不自动改善答案**；要兑换成质量得动 `rerank.top_n`，
   那会作废现有全部质量基线（未拍板）。

---

## 6. 数据契约

### 6.1 中间表示 IR — `ingest/schema.py`

```text
IntermediateDoc
  meta: SourceMeta {source_type, doc_id, title, owner?, ...}
  blocks: [Block {type, text, heading_level?, page?, bbox?}]
              type ∈ heading | paragraph | list_item | table | quote

Chunk
  chunk_id, doc_id, text
  section_path: [str]
  page?: int
  block_type: str
```

落盘：`data/parsed/<doc_id>.json`。

### 6.2 Qdrant payload（点）

```text
chunk_id, doc_id, title, text, section_path, page, block_type
doc_date, category, doc_group, meeting_type
attendees?, topics?          # 默认常空
```

向量名：`dense`（1024 COSINE）、`bm25`（Document model=`qdrant/bm25`）。  
点 id：`uuid5(NAMESPACE_URL, chunk_id)`。  
幂等：upsert 前按 `doc_id` delete。

### 6.3 检索块（retrieved 一项）

`RetrievalOutcome.chunks` 的元素（示意；`retrieve()` 返回的是 `RetrievalOutcome`，不是裸列表）：

```text
{chunk_id, doc_id, text, title, page, block_type, score, ...payload 字段}
```

### 6.4 进 LLM 的 contexts / citations — `orchestrator.py`

```text
contexts: [{no, text, doc, page}]           # 编号从 1 起；doc=title or doc_id
citations: [{no, doc, page, doc_id, block_type}]
```

上下文展示前缀：`（文档名 第p页）` + 正文（prompts.format_context）。  
**judge 必须看到与合成 LLM 逐字相同的上下文串**，否则 Faithfulness 会系统性假低。

### 6.5 Orchestrator.Result

```text
plan, retrieved, contexts, citations
answer, synth_meta
rerank_error, latency_ms
top_n_used, rewrite_top_n
filter_applied, filter_fallback, n_before_fallback
context_budget
```

### 6.6 两份清单纪律（评估/改指标时）

| 清单 | 内容 | 用途 |
|------|------|------|
| `retrieved` | 未截断（重排只重排不砍） | Hit@k（首命中位次）/ MRR / nDCG@8 / 文档覆盖率（逐条真 Recall 的均值，连带 `coverage_ceiling_*` 一起读） |
| `contexts` | 截断后 | 实际进 LLM；引用编号 |

**禁止合并这两份清单当检索结果**——会静默改掉指标分母。  
合成用的是**原始问题**，不是 `plan.rewritten`（改写只服务检索）。

### 6.7 流式 SSE 事件

```text
{"type":"rewrite","plan":{...}}
{"type":"delta","text":"..."}   # 0..n 次
{"type":"citations","citations":[...]}
{"type":"done","result":Result}
```

流式与非流式**同一缓存键**（`llm._build_request` 同源）。

### 6.8 评估结果 meta（自证）

结果 JSON `meta` 至少应能回答：`llm_model`, `answer_cache`, `prompt_version/fingerprint`,
`retrieval` 模式串, `rewrite_degraded`, `rewrite_model`, `rerank_failed`,
`filter_fallback_n` / `filters_applied_n`, `budget`（`fixed:<top_n>` 还是 `rewrite`，即
检索预算听谁）、`context_budget`（`max_contexts` 与 `rerank.top_n` 各自取值）, `collection`。
`summary` 侧对应新增：`hit_at_5 / hit_at_8 / hit_within_budget`、`coverage_ceiling_mean` /
`coverage_ceiling_by_type`、`list_len`（两臂清单是否等长的证据）、`over_refusal_gold_rate`。

---

## 7. 设计不变量（改代码前）

1. **Orchestrator 是唯一装配点**  
   入口层不得再内联管线。改动管线逻辑只改 `orchestrator.py`，并跑 `tests/test_orchestrator_parity.py`。

2. **检索块 ≠ 合成块**  
   指标算 `retrieved`；LLM 只看 `contexts`。重排失败可见；过滤回退可见。

3. **退化必须计数，不能静默**  
   改写失败、重排失败、过滤回退、缓存故障，都要能从 Result / metrics / 日志看到。

4. **配置键必须被代码读取**  
   `tests/test_config_keys_are_wired.py`：default.yaml 叶子键在 src 中必须有读取点。禁止「文档/配置宣称不存在的能力」（如已删除的 `rrf_k`）。

5. **成本相关默认值必须安全**  
   LLM 元数据抽取默认关；`--limit` 同时约束解析与入库；缓存键含 model+base_url+temperature+prompt。

6. **预算分离**  
   改写：延迟预算（5s×2，可退化）；合成：质量预算（可长思考）；judge：关思考省钱、不改身份。

7. **元数据过滤是语料属性约束，不是模型品味**  
   `doc_date` 稀疏 → 只给明确时间限定的聚合题加过滤。

8. **实验臂要能与生产路径等长对照**  
   例如重排不截断检索清单；eval 可用 `plan_override` 重放改写结果。

---

## 8. 「我想改 X」定位表

| 想改的行为 | 主文件 | 副作用/必看 |
|------------|--------|-------------|
| 入口请求字段 / 鉴权 / 限流 | `api/main.py` | 白名单 `allowed_collections`；fail-closed |
| CLI 参数与输出 | `cli.py` | 与 API/demo/eval 的 Orchestrator 调用方式保持一致 |
| 管线顺序、截断、Result 字段 | `orchestrator.py` | parity 测试；指标分母纪律 |
| PDF 解析 / 表格 / 碎化行 | `ingest/pdf.py` | IR 字段；页码/引用 |
| docx 解析 | `ingest/office.py` | LibreOffice 依赖 |
| 中间 JSON 字段 | `ingest/schema.py` | 下游 chunker/indexer/eval 全连坐 |
| 分块大小 / 表格是否切断 | `ingest/chunker.py` | 检索召回 vs 表格完整性（见 PLAN 消融） |
| 入库 payload / 索引 / 幂等 | `ingest/indexer.py` | `use_llm_meta` 默认读配置 |
| 文件名元数据规则 | `ingest/metadata.py` | `backfill` 同源 |
| 向量化模型/维度 | `ingest/embedder.py` + `configs` embedding | `dense_dim` 不匹配应在建库前硬失败 |
| 查询改写 prompt / 意图规则 | `retrieve/rewrite_llm.py` | 泛化门禁 `check-rewrite`；勿把 top_n/filter 交给模型 |
| 双路检索 / RRF / 聚合 / 过滤 | `retrieve/hybrid.py` | Qdrant 1.19.1 行为；filter_fallback |
| 重排模型 / top_n 语义 | `retrieve/rerank.py` | 全量排序 vs context_budget |
| 合成 prompt / 引用规则 | `generate/prompts.py` | 版本注册表 + 指纹；改 prompt 要升版本 |
| 合成调用 / 思考分流 | `generate/synthesizer.py` | 聚合档只在显式配置时覆盖 |
| 缓存 / 重试 / token 计量 | `generate/llm.py` + `net.py` | 缓存键、测延迟要关缓存 |
| 指标定义 / 评估循环 | `eval/runner.py` | retrieved vs contexts；拒答判定 |
| RAGAS / judge 口径 | `eval/runner.py` `_run_ragas` / `_judge_chat_kwargs` / `probe_judge` | judge 上下文与 LLM 逐字一致；`_judge_chat_kwargs` 是正式判分与探针的唯一同源构造点 |
| 配对差 / 噪声地板 | `eval/compare.py` | 小于地板不下结论 |
| 黄金集构造 | `eval/goldgen.py` + `eval/schema.py` | 真实 gold 不入库；样例 golden_sample 可入库 |
| 默认行为总开关 | `configs/default.yaml` | 新键必须有代码读取 |
| HTTP 状态码/重试策略 | `net.py` | 401/400 不重试 |
| 日志是否含语料 | `log.py` | **只落度量，不落问题原文/正文** |

---

## 9. 外部依赖（现行）

| 依赖 | 用途 | 钉版/说明 |
|------|------|-----------|
| Qdrant | 向量+BM25 存储 | compose 钉 `v1.19.1`；`check` 会对版本 |
| Embedding API | BGE-M3 dense 1024 | SiliconFlow 等；**API 不返回 learned sparse** |
| Rerank API | BGE-Reranker-v2-m3 | 默认与 Embedding 同源 endpoint |
| LLM API | 合成/改写/judge | 默认 DeepSeek `deepseek-flash`（OpenAI 兼容） |
| SQLite | LLM 响应缓存 | `.cache/llm_cache.sqlite`，本地可删 |
| LibreOffice | .doc/.docx 归一 | office 解析路径 |

---

## 10. 常用命令（现行）

```bash
# 环境
docker compose up -d
uv sync
uv run doc-rag check

# 入库
uv run doc-rag profile
uv run doc-rag ingest
uv run doc-rag ingest --raw-dir data/sample_raw --parsed-dir data/sample_parsed --kb doc_rag_sample --recreate

# 问答
uv run doc-rag query "问题" --timing
uv run doc-rag query "问题" --stream --no-rerank

# 服务
uv run doc-rag serve          # 需 DOC_RAG_API_TOKEN
uv run --extra demo doc-rag demo --port 7860

# 评估
uv run doc-rag eval --gold data/eval/gold.json --rewrite --rerank --fresh-answers
uv run doc-rag eval --gold data/eval/gold.json --rewrite --rerank --retrieval-only --honor-rewrite-budget  # 生产预算口径
uv run doc-rag check-rewrite
uv run doc-rag compare-ragas --group "A=a.json" --group "B=b.json"
uv run doc-rag probe-judge <results.json> <item_id>

# 缓存（键里带问题原文 + 检索到的正文；re-ingest 后要作废）
uv run doc-rag cache-stats
uv run doc-rag cache-clear --yes
```

---

## 11. 相关文档

| 文档 | 角色 |
|------|------|
| [README.md](../../README.md) | 项目入口、快速开始、评估结果摘要 |
| [docs/README.md](../README.md) | 文档索引 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 架构总览与数据流示意 |
| [diagrams/doc-rag-full-flow.drawio](diagrams/doc-rag-full-flow.drawio) | 完整流程图（可编辑） |
| [guides/how-it-works.md](../guides/how-it-works.md) | 教学向全链路拆解 |
| [design/PLAN.md](../design/PLAN.md) | 决策证据、消融、被推翻的结论（**事实源，不是操作手册**） |
