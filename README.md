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

- [x] 骨架：配置 / 统一中间表示 / 接入（PDF + doc(x)，含带框表格重建）/ 结构分块 / 画像 CLI
- [ ] Phase 0：语料画像、表格解析抽查、黄金集 v0
- [ ] Phase 1：Embed + Qdrant 入库、Hybrid 检索（Dense+Sparse+RRF）、引用问答、第一版 RAGAS
- [ ] Phase 2：四组消融、元数据过滤、Rerank、元数据全量字段
- [ ] Phase 3：全量压测、图片 caption 入库、GraphRAG 跨文档

## 注意

- `data/raw`、`data/parsed` 已 gitignore——**公司文档严禁提交**（合规，见 PLAN §8）
- API Key 全部走环境变量（`configs/default.yaml` 中 `env:` 前缀）
- 评估依赖：`uv sync --extra eval`
