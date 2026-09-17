# doc-rag · 企业文档 RAG

面向公司会议记录类文档（**飞书批量导出 PDF / .doc(x)**）的检索问答系统。
设计决策、取舍理由与评估口径见 [PLAN.md](PLAN.md)。

## 快速开始

```bash
cp .env.example .env   # 填 Key：LLM=DeepSeek 官方 / Embedding=SiliconFlow
uv sync                # 安装依赖，首次生成 uv.lock（版本随之锁定）
uv run doc-rag check   # API 冒烟：LLM 连通 / Embedding 维度 / sparse 探测
docker compose up -d   # Qdrant :6333
uv run doc-rag profile # Phase 0：语料画像（data/raw 放入语料后执行）
uv run doc-rag ingest  # 双路接入 → data/parsed 统一中间 JSON
uv run pytest          # 测试
```

## 状态（对照 PLAN §7 路线图）

- [x] 骨架：配置 / 统一中间表示 / 接入（PDF + doc(x)，含碎化行重建与带框表格重建）/ 结构分块 / 画像 CLI
- [x] Phase 0：语料画像（1130 份）、表格解析抽查、黄金集 v1（64 条，人名实体构造）
- [x] Phase 1：入库 1121 篇 / 3856 块、Hybrid 检索（Dense+BM25+RRF）、引用问答、双轨评估闭环
- [~] Phase 2：消融 #1–#5 全部完成（Dense vs Hybrid / 分块 / Rerank / 引用约束 / 元数据过滤）；
  FastAPI 与演示页进行中
- [ ] Phase 3：全量压测、图片 caption 入库、GraphRAG 跨文档

## 评估结果（黄金集 v1 · 63 条 · 1121 篇语料）

消融 #1：纯 Dense vs Hybrid（Dense+BM25+RRF，top-8）

| 指标 | 纯 Dense | Hybrid（默认） |
|------|---------|---------------|
| Recall@5 | 0.836 | **0.909** |
| Recall@8 | 0.891 | **0.945** |
| MRR | 0.764 | **0.794** |

分题型（Recall@8，Hybrid）：fact 1.00 · open_discussion 1.00 · decision 0.88 ·
term 0.83 · time_filter 1.00 · cross_doc 1.00

**跨文档聚合题**（改用文档覆盖率度量，Recall 类指标对此已失去区分度）：

- 覆盖率上限 = top_n / 答案集大小；预算扫描：cross_doc 0.13（top-8）→ 0.35（top-30）
- 实体聚焦查询：周碧玉 0.06→0.22
- 消融 #5 元数据过滤（时间限定题）：覆盖率 **0.61 → 0.96**

回答质量（产品路径：改写 + 重排 + 收紧 prompt + deepseek-flash，黄金集 63 条）：

| 指标 | 数值 |
|------|------|
| 包含匹配准确率 | **0.8545** |
| 拒答正确率 | **1.00** |
| 引用有效率 / 存在率 | **1.00 / 1.00** |
| RAGAS Faithfulness | **0.95**（全量 55 条 · 修正口径 · 两轮 0.9563/0.9423） |
| 检索延迟（改写+嵌入+RRF+重排） | p50 **423ms** · p95 **509ms** |
| 端到端延迟 | p50 **4988ms** · **p95 32825ms（超 8s 目标）** |

延迟（全量 63 条实测 · deepseek-flash · 合成缓存关闭）：

| 题型 | 端到端 p50 | 答案均长 |
|------|-----------|---------|
| fact / term | 2.2～2.8s | 约 100 字 |
| decision | 5.3s | 184 字 |
| cross_doc | 24.6s | 743 字 |
| time_filter | 32.8s | 890 字 |

> **P95 ≤ 8s 目标不达标**：端到端 p95 32.8s，33%（21/63）的条目超 8s。延迟是**双峰**的
> ——短答题 2～3s，聚合题 24～33s，报单一均值会同时低估前者、藏起后者。根因是
> deepseek-flash 的输出里 **91% 是看不见的 reasoning token**，聚合题要跨 15～25 块归纳，
> 思考量随之爆炸。**压延迟的唯一有效杠杆是关掉合成侧思考**（`DOC_RAG_LLM_REASONING_EFFORT=none`）：
> 端到端 p95 32.8s → **2.6s（达标）**、成本 −94%，代价是包含匹配 −10.9pt（p=0.031），
> 损失集中在 decision/open_discussion 这类需要归纳的题。属产品取舍，默认保持质量优先。
> 复现：`uv run doc-rag eval --rewrite --rerank --fresh-answers`（`--fresh-answers` 必须加——
> 缓存命中的毫秒数是本地查询耗时，不是模型延迟）。

> **Faithfulness 的 0.77 是度量 bug，已修正**：judge 拿到的上下文缺了 `（文档名 第p页）` 前缀，
> 而合成 prompt 要求标注来源文档名 → 答案里「《某文档》中…」被判为不忠实
> （提及文档名的 18 条均值 0.475，不提的 37 条 0.716）。把 LLM 实际看到的上下文原样交给 judge 后，
> 同一批答案从 0.6375 升到 **0.9563**，消融 #2 的 18pt 假差距随之归零（修正后 0.5pt，
> 小于 1.4pt 的 judge 噪声地板）。复盘见 PLAN §5.3。
> 配套：RAGAS 抽样改为**均匀覆盖**（原先取前 N 条，整段漏掉排在末尾的 cross_doc/time_filter）；
> 逐条分数落盘；`doc-rag compare-ragas` 出配对差 + 95%CI + 符号检验 + 噪声地板；
> `doc-rag probe-judge` 打印单条的 judge 中间产物（口径 bug 就是靠它定位的）。

> **延迟数字也曾是笔糊涂账，已补测量能力**：PLAN 里同时存在「合成 5.3～7.4s / 端到端 ≈6.3s 达标」
> 与「deepseek-flash 1.3s」两个矛盾的数——前者测于 OpenRouter 时期的另一个模型且是估算，
> 后者是无上下文的连通性 ping。根因是**仓库里没有任何延迟测量代码**，数字只存在于文档文本。
> 现在 `eval` 落盘分阶段（改写/检索/重排/合成/端到端）+ 分题型分位数，
> `meta` 自证 `llm_model` 与 `answer_cache`，`latency.cache_contaminated` 标记缓存污染。

模型选型对比（同条件换模型，检索指标完全一致）：

| 指标 | deepseek-flash | GLM-4.5-Air |
|------|---------------|-------------|
| 包含匹配 | **0.836** | 0.745 |
| 引用存在率 | **1.00** | 0.921 |

> 本地响应缓存使重复评估近乎零成本（实测命中 90%）：同配置重跑不重复付费。

## 注意

- `data/raw`、`data/parsed` 已 gitignore——**公司文档严禁提交**（合规，见 PLAN §8）
- API Key 全部走环境变量（`configs/default.yaml` 中 `env:` 前缀）
- 评估依赖：`uv sync --extra eval`
