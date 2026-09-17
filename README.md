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

## 评估结果（黄金集 v1 · 62 条 · 1121 篇语料）

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

回答质量（Hybrid）：包含匹配 0.82 · 拒答正确率 1.00 · 引用有效率 1.00 · RAGAS Faithfulness 0.62

## 注意

- `data/raw`、`data/parsed` 已 gitignore——**公司文档严禁提交**（合规，见 PLAN §8）
- API Key 全部走环境变量（`configs/default.yaml` 中 `env:` 前缀）
- 评估依赖：`uv sync --extra eval`
