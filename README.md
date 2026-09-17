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
- [~] Phase 2：消融 #1（纯 Dense vs Hybrid）已完成；cross_doc 聚合检索、Rerank、Faithfulness 优化进行中
- [ ] Phase 3：全量压测、图片 caption 入库、GraphRAG 跨文档

## 评估结果（黄金集 v1 · 64 条 · 1121 篇语料）

消融 #1：纯 Dense vs Hybrid（Dense+BM25+RRF）

| 指标 | 纯 Dense | Hybrid（默认） |
|------|---------|---------------|
| Recall@5 | 0.667 | **0.702～0.719** |
| Recall@8 | 0.754 | **0.807** |
| MRR | 0.597 | **0.602～0.612** |

> 区间为两次独立运行结果：RRF 并列分数的打破顺序带来 ±1.7pt 波动，
> 因此所有结论只按相对变化读（PLAN §5.3 评估可信度口径）。

分题型（Recall@8，Hybrid）：fact 1.00 · open_discussion 1.00 · decision 0.88 ·
term 0.83 · time_filter 0.57 · **cross_doc 0.38（当前瓶颈，待做聚合检索）**

回答质量（Hybrid）：包含匹配 0.82 · 拒答正确率 1.00 · 引用有效率 1.00 · RAGAS Faithfulness 0.62

## 注意

- `data/raw`、`data/parsed` 已 gitignore——**公司文档严禁提交**（合规，见 PLAN §8）
- API Key 全部走环境变量（`configs/default.yaml` 中 `env:` 前缀）
- 评估依赖：`uv sync --extra eval`
