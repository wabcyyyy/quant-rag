# Doc-RAG 运行时架构图

## 系统概览

```mermaid
flowchart TD
    %% ============ External Systems ============
    subgraph External["外部系统"]
        LLM_API[("LLM API\nDeepSeek/OpenRouter/\nSiliconFlow")]
        EMB_API[("Embedding API\nBGE-M3 / SiliconFlow")]
        RERANK_API[("Rerank API\nBGE-Reranker-v2-m3 /\nSiliconFlow")]
        QDRANT[("Qdrant Vector DB\nlocalhost:6333")]
        FILE_SYSTEM[("文件系统\ndata/raw, data/parsed,\n.cache")]
    end

    %% ============ Ingestion Pipeline (Offline) ============
    subgraph Ingestion["离线摄入管线"]
        direction TB
        
        subgraph Parse["解析阶段"]
            RAW[("原始文档\nPDF / DOCX")]
            PARSE_PDF["PDF 解析器\nsrc/ingest/pdf.py"]
            PARSE_OFFICE["Office 解析器\nsrc/ingest/office.py"]
            INTERMEDIATE[("中间 JSON\nIntermediateDoc\nschema.py")]
        end

        subgraph Chunk["分块阶段"]
            CHUNKER["分块器\nsrc/ingest/chunker.py\nstructural / fixed"]
            CHUNKS[("Chunk 对象\nsection_path, page,\nblock_type")]
        end

        subgraph Meta["元数据抽取 (可选)"]
            LLM_META["LLM 元数据抽取\nsrc/ingest/metadata.py\n1次/文档, 成本高"]
            BASE_META["基础元数据\n文件名启发式\ncategory, doc_group,\ndoc_date, meeting_type"]
            META[("文档元数据\ndoc_date, category,\ndoc_group, topics,\nattendees")]
        end

        subgraph Embed["向量化"]
            EMBEDDER["Embedder\nsrc/ingest/embedder.py\nBGE-M3 dense 1024d\n批量 32, 重试 3次"]
            DENSE_VEC[("Dense 向量\n1024 维")]
        end

        subgraph Index["入库"]
            INDEXER["Indexer\nsrc/ingest/indexer.py\n幂等 upsert\nThreadPoolExecutor(8)"]
            BM25_TEXT["BM25 文本\njieba 分词\nsrc/ingest/bm25.py"]
            QDRANT_UPSERT[("Qdrant Collection\nnamed vectors:\n- dense (COSINE)\n- bm25 (IDF modifier)\npayload indexes:\ndoc_id, category,\ndoc_group, block_type,\ndoc_date, topics,\nattendees")]
        end
    end

    %% ============ Online Query Pipeline ============
    subgraph Online["在线查询管线"]
        direction TB
        
        subgraph API["API 层"]
            FASTAPI["FastAPI\nsrc/api/main.py\n:8000"]
            AUTH["鉴权 + 限流\nrequire_token()\n滑动窗口 RPM"]
            HEALTH["/health"]
            METRICS["/metrics\nPrometheus 格式"]
            QUERY_EP["POST /query"]
            STREAM_EP["POST /query/stream\nSSE"]
        end

        subgraph Orchestrator["Orchestrator 统一入口\nsrc/orchestrator.py"]
            ORCH["Orchestrator 类\n单例 (lru_cache)"]
            PREPARE["_prepare()\n编排核心逻辑"]
        end

        subgraph Rewrite["查询改写 (可选)"]
            REWRITER["LLMQueryRewriter\nsrc/retrieve/rewrite_llm.py\n意图分类 + 过滤器提取\nreasoning_effort=none\n5s timeout, 2 attempts"]
            REWRITE_PLAN[("改写计划\nrewritten, filters,\naggregate, top_n,\nreason, degraded")]
        end

        subgraph Retrieve["混合检索"]
            HYBRID["HybridRetriever\nsrc/retrieve/hybrid.py"]
            DENSE_QUERY["Dense 查询\nBGE-M3 向量化\nclient.query_points"]
            BM25_QUERY["BM25 查询\nQdrant 内置 bm25\njieba 预分词"]
            RRF["RRF 融合\nFusionQuery(RRF)"]
            AGGREGATE["聚合检索\npool=50, 按 doc_id 去重\n每文档最佳块"]
            FILTER_FALLBACK["过滤回退\n结果<3 时去掉过滤重试\n记 filter_fallback"]
            RETRIEVED[("检索结果\nretrieved (未截断)\nscore, chunk_id,\ndoc_id, text, page,\nblock_type, doc_date")]
        end

        subgraph Rerank["重排 (可选)"]
            RERANKER["Reranker\nsrc/retrieve/rerank.py\nBGE-Reranker-v2-m3\n全量重排, 不截断清单"]
            RERANKED[("重排结果\n或退回融合顺序\nrerank_error")]
        end

        subgraph Context["上下文构建"]
            TRUNCATE["上下文预算截断\nmin(max_contexts=10,\nrerank.top_n=6)"]
            CONTEXTS[("contexts (已截断)\nno, text, doc, page")]
            CITATIONS[("citations\nno, doc, page,\ndoc_id, block_type")]
        end

        subgraph Synthesize["合成层"]
            SYNTH["Synthesizer\nsrc/generate/synthesizer.py"]
            PROMPTS["Prompts\nsrc/generate/prompts.py\nsystem: tightened/baseline\nuser: 编号引用 + 无据拒答"]
            LLM_CLIENT["LLM Client\nsrc/generate/llm.py\nOpenAI SDK 兼容\n流式/非流式\n本地 SQLite 缓存"]
            ANSWER[("答案\n带 [n] 引用")]
        end
    end

    %% ============ Evaluation System ============
    subgraph Evaluation["评估体系"]
        direction TB
        
        GOLDEN["黄金集生成\nsrc/eval/goldgen.py\nLLM 生成 + 程序化"]
        EVAL_RUNNER["评估运行器\nsrc/eval/runner.py"]
        RETRIEVAL_METRICS["检索指标\nHit@5/8/清单末, MRR, nDCG@8,\n文档覆盖率 (+ 结构上限)"]
        ANSWER_METRICS["答案指标\n包含匹配, 拒答正确率,\n引用有效率/存在率"]
        RAGAS["RAGAS 第二轨\nfaithfulness\njudge 模型 (关思考)"]
        JUDGE["Judge 评估\nsrc/eval/schema.py\n陈述抽出 + 逐条判定"]
        COMPARE["对比分析\nsrc/eval/compare.py\n配对差 + 噪声地板"]
    end

    %% ============ Observability ============
    subgraph Observability["可观测性"]
        METRICS_REG["进程内指标注册表\nsrc/metrics.py\nCounter / Histogram"]
        LOGGING["结构化 JSON 日志\nsrc/log.py\nrequest_id 追踪"]
        LLM_CACHE["LLM 响应缓存\n.cache/llm_cache.sqlite\n命中率统计"]
        CACHE_STATS["缓存统计\ncache_stats()\nhit/miss, token用量,\n推理token单列"]
    end

    %% ============ Connections ============
    %% Ingestion flow
    RAW --> PARSE_PDF
    RAW --> PARSE_OFFICE
    PARSE_PDF --> INTERMEDIATE
    PARSE_OFFICE --> INTERMEDIATE
    INTERMEDIATE --> CHUNKER
    CHUNKER --> CHUNKS
    CHUNKS --> LLM_META
    CHUNKS --> BASE_META
    LLM_META --> META
    BASE_META --> META
    META --> INDEXER
    CHUNKS --> INDEXER
    INDEXER --> EMBEDDER
    EMBEDDER --> DENSE_VEC
    INDEXER --> BM25_TEXT
    DENSE_VEC --> QDRANT_UPSERT
    BM25_TEXT --> QDRANT_UPSERT
    META --> QDRANT_UPSERT
    INDEXER --> QDRANT_UPSERT

    %% Online flow
    QUERY_EP --> AUTH
    STREAM_EP --> AUTH
    AUTH --> ORCH
    ORCH --> PREPARE
    PREPARE --> REWRITER
    REWRITER --> REWRITE_PLAN
    PREPARE --> HYBRID
    HYBRID --> DENSE_QUERY
    HYBRID --> BM25_QUERY
    DENSE_QUERY --> EMB_API
    BM25_QUERY --> QDRANT
    DENSE_QUERY --> RRF
    BM25_QUERY --> RRF
    RRF --> AGGREGATE
    AGGREGATE --> FILTER_FALLBACK
    FILTER_FALLBACK --> RETRIEVED
    RETRIEVED --> RERANKER
    RERANKER --> RERANK_API
    RERANKER --> RERANKED
    RERANKED --> TRUNCATE
    TRUNCATE --> CONTEXTS
    CONTEXTS --> CITATIONS
    CONTEXTS --> SYNTH
    SYNTH --> PROMPTS
    SYNTH --> LLM_CLIENT
    LLM_CLIENT --> LLM_API
    LLM_CLIENT --> LLM_CACHE
    LLM_CLIENT --> ANSWER
    ANSWER --> QUERY_EP
    ANSWER --> STREAM_EP

    %% Evaluation connections
    GOLDEN --> EVAL_RUNNER
    ORCH --> EVAL_RUNNER
    RETRIEVED --> RETRIEVAL_METRICS
    ANSWER --> ANSWER_METRICS
    ANSWER --> RAGAS
    CONTEXTS --> RAGAS
    RAGAS --> JUDGE
    JUDGE --> LLM_API
    EVAL_RUNNER --> COMPARE

    %% Observability connections
    ORCH --> METRICS_REG
    QUERY_EP --> METRICS_REG
    STREAM_EP --> METRICS_REG
    ORCH --> LOGGING
    LLM_CLIENT --> CACHE_STATS
    LLM_CLIENT --> LOGGING

    %% Config
    CONFIG[("配置\nconfigs/default.yaml\n+ .env + 环境变量")]
    CONFIG -.-> Ingestion
    CONFIG -.-> Online
    CONFIG -.-> Evaluation
    CONFIG -.-> Observability

    %% Styles
    classDef external fill:#f5f5f5,stroke:#999,stroke-width:1px
    classDef ingestion fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef online fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef eval fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef obs fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef data fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 5 5

    class LLM_API,EMB_API,RERANK_API,QDRANT,FILE_SYSTEM external
    class RAW,PARSE_PDF,PARSE_OFFICE,INTERMEDIATE,CHUNKER,CHUNKS,LLM_META,BASE_META,META,EMBEDDER,DENSE_VEC,INDEXER,BM25_TEXT,QDRANT_UPSERT ingestion
    class FASTAPI,AUTH,HEALTH,METRICS,QUERY_EP,STREAM_EP,ORCH,PREPARE,REWRITER,REWRITE_PLAN,HYBRID,DENSE_QUERY,BM25_QUERY,RRF,AGGREGATE,FILTER_FALLBACK,RETRIEVED,RERANKER,RERANKED,TRUNCATE,CONTEXTS,CITATIONS,SYNTH,PROMPTS,LLM_CLIENT,ANSWER online
    class GOLDEN,EVAL_RUNNER,RETRIEVAL_METRICS,ANSWER_METRICS,RAGAS,JUDGE,COMPARE eval
    class METRICS_REG,LOGGING,LLM_CACHE,CACHE_STATS obs
    class CONFIG data
```

## 关键数据流

### 1. 离线摄入流 (Ingestion Pipeline)

```
data/raw/*.pdf,*.docx
       │
       ▼
┌──────────────────┐
│  pipeline.run()  │  ← 双路由：.pdf → extract_pdf, .docx/.doc → extract_office
└────────┬─────────┘
         │ IntermediateDoc (schema.py)
         ▼
┌──────────────────┐
│  chunk_by()      │  ← structural (标题层级) / fixed (固定大小)
└────────┬─────────┘
         │ List[Chunk]
         ▼
┌──────────────────┐     ┌──────────────────┐
│  extract_metadata() │  │  base_meta()     │  ← LLM 抽取 (可选, --llm-meta) / 文件名启发式
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
         ┌──────────────────┐
         │  embedder.embed()│  ← BGE-M3, 批量 32, 重试 3 次, 维度校验 1024
         └────────┬─────────┘
                  │ dense vectors
                  ▼
         ┌──────────────────┐
         │ index_parsed()   │  ← 幂等 upsert: 删旧块 → 批量 64 upsert
         │  - dense vector  │
         │  - bm25 Document │  ← jieba 分词文本 + qdrant/bm25 模型
         │  - payload indexes│
         └────────┬─────────┘
                  │
                  ▼
         Qdrant Collection: doc_rag_demo
         - vectors: dense (COSINE), bm25 (IDF)
         - payload indexes: doc_id, category, doc_group, block_type, doc_date, topics, attendees
```

### 2. 在线查询流 (Online Query Pipeline)

```
POST /query {question, kb?, top_n?}
       │
       ▼
┌──────────────────┐
│ require_token()  │  ← 鉴权 + 限流 (滑动窗口, 进程内)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Orchestrator.answer() │  ← 单例 (lru_cache), 无跨请求可变状态
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ _prepare()       │  ← 核心编排: rewrite → retrieve → rerank → truncate
└────────┬─────────┘
         │
    ┌────┴────┐
    ▼         ▼
[use_rewrite] [use_rewrite=False]
    │            │
    ▼            ▼
┌─────────┐  ┌─────────────────┐
│LLMQuery │  │ plan = {        │
│Rewriter │  │  rewritten: q,  │
│         │  │  filters: null, │
│reasoning│  │  aggregate: F,  │
│_effort= │  │  top_n: null,   │
│none     │  │  reason: "已关闭│
│5s/2att  │  │   改写",        │
└────┬────┘  │  degraded: F }  │
     │       └────────┬────────┘
     ▼                │
┌─────────┐           │
│ plan =  │           │
│ {rewrit-│           │
│ ten,    │           │
│ filters,│           │
│ aggregate│           │
│ }       │           │
└────┬────┘           │
     │                │
     └───────┬────────┘
             ▼
    ┌──────────────────┐
    │ HybridRetriever  │
    │ .retrieve()      │
    └────────┬─────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
  Dense路         BM25路
  (向量化后查询)   (jieba分词后查询)
      │             │
      └──────┬──────┘
             ▼
      ┌──────────────────┐
      │ RRF 融合         │  ← Qdrant FusionQuery(RRF), prefetch 双路
      └────────┬─────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
    [aggregate]    [!aggregate]
        │             │
        ▼             ▼
  pool=50, 按    limit=12
  doc_id 去重,      │
  返回每文档      ▼
  最佳块        结果列表
        │
        ▼
    ┌──────────────────┐
    │ 过滤回退保护     │  ← 结果 < min(3, limit) 时去掉过滤重试
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ [use_rerank]     │
    └────────┬─────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
  Reranker       退回融合顺序
  (BGE-Reranker)  rerank_error
  全量重排, 不砍清单
      │
      ▼
    ┌──────────────────┐
    │ 上下文预算截断    │  ← min(max_contexts=10, rerank.top_n=6)
    │ → contexts       │
    │ 生成 citations   │
    └────────┬─────────┘
             │
             ▼
    ┌──────────────────┐
    │ Synthesizer      │
    │ .answer() /      │
    │ .answer_stream() │  ← 同一缓存键, 流式与非流式互通
    └────────┬─────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
   非流式         流式 (SSE)
   返回完整       事件: rewrite/
   答案+引用      delta*/citations/done
      │
      ▼
   返回 Result
   {answer, citations,
    latency_ms, plan,
    rerank_error, ...}
```

### 3. 延迟测量口径 (关键设计)

```
latency_ms = {
  "rewrite":      改写耗时 (ms),
  "retrieve":     检索耗时 (ms),
  "rerank":       重排耗时 (ms),
  "retrieval_total": rewrite+retrieve+rerank 总和,
  "synthesize":   合成耗时 (ms) - 缓存命中时为本地查询耗时,
  "synth_cached": 是否缓存命中 (缓存命中时不是模型延迟!),
  "total":        端到端总耗时
}

关键约束:
- 测真实延迟必须: DOC_RAG_LLM_CACHE=0 或 --fresh-answers
- 缓存命中的 synthesize ms = 本地 SQLite 查询耗时 (毫秒级)
- reasoning_effort 显著影响合成延迟 (聚合题 24-32s → 2s)
```

## 核心组件交互矩阵

| 组件 | 职责 | 关键配置 | 依赖 |
|------|------|----------|------|
| **Orchestrator** | 管线统一编排 | cfg, retriever, synthesizer | HybridRetriever, LLMQueryRewriter, Synthesizer |
| **HybridRetriever** | Dense+BM25 RRF 检索 | k_dense=20, k_bm25=20, fusion_limit=12 | QdrantClient, Embedder |
| **LLMQueryRewriter** | 查询改写/意图分类 | reasoning_effort=none, timeout_s=5 | LLM API |
| **Reranker** | 语义重排 | top_n=6, model=bge-reranker-v2-m3 | Rerank API |
| **Synthesizer** | 带引用合成/拒答 | prompt_version=tightened, require_citation | LLM Client, Prompts |
| **LLM Client** | 统一调用+缓存+重试 | cache=true, max_attempts=4, timeout=180s | OpenAI SDK, SQLite |
| **Embedder** | BGE-M3 向量化 | dense_dim=1024, batch=32 | Embedding API |
| **Indexer** | 入库 (幂等) | recreate, use_llm_meta, chunk_strategy | QdrantClient, Embedder, Chunker |

## 部署拓扑

```mermaid
flowchart LR
    subgraph Client["客户端"]
        CLI[doc-rag CLI]
        CUR[curl / HTTP客户端]
        GRADIO[Gradio Demo\n:7860]
    end

    subgraph Server["服务端 (单进程)"]
        FASTAPI[FastAPI :8000]
        ORCH[Orchestrator 单例]
        CACHE[(LLM缓存\n.cache/llm_cache.sqlite)]
        METRICS[(指标注册表\n内存)]
    end

    subgraph Infra["基础设施"]
        QDRANT[Qdrant :6333\nCollection: doc_rag_demo]
        LLM[LLM API\nDeepSeek/OpenRouter]
        EMB[Embedding API\nBGE-M3]
        RERANK[Rerank API\nBGE-Reranker]
    end

    CLI --> FASTAPI
    CUR --> FASTAPI
    GRADIO --> FASTAPI
    FASTAPI --> ORCH
    ORCH --> CACHE
    ORCH --> METRICS
    ORCH --> QDRANT
    ORCH --> LLM
    ORCH --> EMB
    ORCH --> RERANK
```

## 配置管理

```yaml
# configs/default.yaml + .env (env:VAR 优先)

embedding:        # BGE-M3 dense 向量
  base_url, api_key, model, dense_dim: 1024

llm:              # 合成/判断/改写 (默认同源)
  base_url, api_key, model
  temperature: 0.0
  cache: true                    # 本地响应缓存
  reasoning_effort: ""           # 空=开思考, "none"=关思考
  reasoning_effort_by_type:      # 按**预测题型**查表；未列出的跟随全局
    cross_doc: none              # 可用键只有 single/cross_doc/time_filter
    time_filter: none            # 填 term/fact 会直接报错（服务侧预测不到）

qdrant:
  url: http://localhost:6333
  collection: doc_rag_demo

api:
  auth_token: env:DOC_RAG_API_TOKEN
  allowed_collections: [doc_rag_demo, doc_rag_sample]
  rate_limit_rpm: 30

retrieval:
  k_dense: 20, k_bm25: 20
  fusion_limit: 12
  aggregate_pool: 50
  max_contexts: 10

rewrite:          # 改写独立预算 (关键路径)
  reasoning_effort: none
  timeout_s: 5
  max_attempts: 2

rerank:
  enabled: true
  model: BAAI/bge-reranker-v2-m3
  top_n: 6

metadata_extraction:
  enabled: false  # 成本高, 需显式 --llm-meta

eval:
  gold_file: data/eval/gold.json
  ragas_sample: 15
  judge.reasoning_effort: none
```

## 关键设计决策记录

| 决策 | 理由 | 位置 |
|------|------|------|
| Orchestrator 单例 + 无可变状态 | 避免并发请求互串 collection | orchestrator.py:75-121 |
| 改写/合成/重排独立 LLM 配置 | 关键路径预算分离 (改写 5s×2 vs 合成 180s) | rewrite_llm.py:123-144 (endpoint), configs/default.yaml:63-91 |
| 本地 SQLite LLM 缓存 | 评估重跑零成本, 流式/非流式互通 | llm.py:33, 75-135 |
| Dense+BM25 RRF 融合 | Qdrant 服务端融合, 无需客户端合并 | hybrid.py:114-134 |
| 聚合检索 (doc_id 去重) | 跨文档题需要文档多样性 | hybrid.py:50-66, 136-144 |
| 过滤回退保护 | 元数据字段稀疏 (doc_date 仅 18.8% 覆盖) | hybrid.py:78-84 |
| 上下文预算 = min(max_contexts, rerank.top_n) | 控制输入 token；重排只重排序不砍清单，两臂才等长可比 | orchestrator.py:196-203, rerank.py |
| reasoning_effort_by_type | 聚合题关思考无损质量, 延迟 30s→2s；`low` 实测是上限不是中间档 | synthesizer.py:resolve_effort_cfg + configs/default.yaml |
| Prompt 版本管理 (tightened/baseline) | 消融实验变量隔离；结果文件按指纹自证 | prompts.py:49,60,93 + eval/runner.py meta.prompt_fingerprint |
| 缓存键包含 base_url | 防止同名模型不同供应商混用缓存 | llm.py:85-96 |
| 进程内指标 + Prometheus 格式 | /metrics 需要鉴权, 不对外裸奔 | main.py:111-140, metrics.py |