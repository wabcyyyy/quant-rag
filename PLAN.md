# 企业文档 RAG 项目计划书（会议记录语料 · PDF/docx 双路）

> 版本：v0.6（采集路线定稿：批量导出 PDF 为主，飞书 API 条件触发）  
> 日期：2026-09  
> 已确认：公司文档 RAG 实习立项 · 自研薄编排（库当零件）· Dense+BM25 Hybrid（sparse 留 Phase 2 实验）· 云端 API · 批量导出 PDF/doc(x) 双路 · CLI/API 基线  
> 定位：**公司会议记录类文档（首批 ~300 份，可扩至几千）的检索问答系统，同时是一段能讲清工程取舍与质量闭环的实习经历**  
> 语料画像：首批约 300 份会议记录式短文档；来源：飞书云文档（批量导出 PDF）+ .doc/.docx；born-digital、无图文交叉复杂版面（Phase 0 画像复核）  
> 注：本文档为个人主计划（含 §1/§6 求职叙事）；提交 mentor 的立项版需剥离面试相关小节

---

## 1. 项目定位（求职叙事）

这不是「再做一个聊天机器人」，而是公司文档场景的真实立项：解决几百份分散会议记录的检索与复用痛点，同时成为一段 **能证明你懂 RAG 真实难点** 的实习经历：

| 面试官会问 | 你要能展示的 |
|------------|--------------|
| 这项目为什么存在？ | 公司文档检索痛点——需求来源找 mentor 要量化数字（检索耗时/复用率） |
| 为什么不直接 LangChain 一把梭？ | 重型轮子全用库（存储/解析/模型/评估），公司定制层自研；框架抽象反而碍事 |
| 这个语料难在哪？ | 同质短文档的区分与聚合（一个查询命中几十份相似记录）→ 元数据抽取 + Hybrid + 消融数字说话 |
| 为什么不用飞书 API，用批量导出的 PDF？ | 批量导出已解决采集；接入不是差异化点，工程预算花在元数据与评估。重评触发线：几千份且更新频繁 / 表格或标题解析质量不达标 |
| 「讨论过但没决定」怎么不答错？ | 决议/讨论区分的 prompt 约束 + 专项评估题 |
| 怎么证明变好了？ | 黄金集 + RAGAS，有 before/after 数字 |
| 幻觉怎么控？ | 强制引用、faithfulness 门禁、无据拒答 |
| 和网上 demo 差在哪？ | Hybrid + Rerank、结构元数据、可追溯引用 |

**电梯演讲（面试开场，30 秒）**：

> 我在公司实习时做了一套面向几百份会议记录的 RAG 问答系统。这类语料难在「同质」——一个查询动辄命中几十份相似记录，所以我在入库时用 LLM 抽取日期、参会人、议题元数据，检索走 Dense+BM25 混合+元数据过滤+重排，自建黄金集跑消融，Recall@10 从 __ 提到 __。文档来自飞书和 Word，批量导出后在接入层做双通道归一化——表格按版面重建而不是硬切文本。

**简历 bullet 模板（实习经历区；数字跑完评估后回填，禁止预填；「负责/主导/参与」按实际职责强度选词）**：

```text
XX公司 XX部门 · LLM 应用实习生                    2026.xx–2026.xx
• 负责企业文档 RAG 问答系统核心管线（Qdrant/PyMuPDF/BGE-M3）：双通道接入归一化
  （PDF/docx、带框表格重建）、结构感知分块、LLM 元数据抽取（日期/参会人/议题/决议）、
  Dense+BM25+RRF 混合检索 + 元数据过滤、Rerank、强制引用与无据拒答
• 自建 50+ 条中文黄金评估集（决议确认/跨文档聚合/时间限定/无答案题），RAGAS + 事实题
  exact-match 双轨评估，消融定位各模块贡献：Recall@10 __→__，Faithfulness __→__
• 沉淀失败案例分析（扫描表格解析、术语检索 miss 等），修复过程可复现
```

---

## 2. 技术决策（已确认）

| 项 | 选择 | 理由 |
|----|------|------|
| 路线 | 自研薄编排层：重型轮子全用库，公司定制层自己写 | 不重复造轮子；贡献集中在文档特有定制，讲得清 |
| 接入 | 飞书批量导出 PDF + doc(x)（采集已解决的路径） | 飞书 OpenAPI 降为可选：仅当增量同步/表格质量成为瓶颈时启用 |
| 文件解析 | doc/docx：LibreOffice headless + mammoth；PDF：PyMuPDF | born-digital 语料，轻依赖即可；MinerU 仅在画像发现扫描件时兜底 |
| 元数据抽取 | LLM 入库时抽取：日期、会议类型、参会人、议题、决议 | 会议记录的检索价值一半在元数据；300 份短文档抽取成本可忽略 |
| 编排 | 自研 Pipeline（不引编排框架，模块边界清晰） | 解析/分块/评估全是深度定制，框架抽象碍事；未来多轮/agent 场景可局部引入 LlamaIndex/LangGraph |
| 存储 | Qdrant（Dense + 全文 BM25 双路） | Query API 原生 RRF 融合；单存储，几千份规模无压力 |
| Embedding | BGE-M3 dense via SiliconFlow API | 中文文档正确默认；API 不吐 sparse（2026-09 查证），hybrid 第二路走 Qdrant full-text BM25 |
| Reranker | bge-reranker（本地小模型）或 API rerank | 性价比高 |
| LLM | DeepSeek 官方 API：deepseek-v4-pro（旗舰正式版） | 合成与 judge 质量优先；deepseek-flash 可作批量抽取的便宜备选；旧别名 deepseek-chat 已过渡期不用 |
| 评估 | RAGAS + exact-match 双轨；黄金集 v0 ≥30 → 终版 ≥50（每题型 ≥8） | LLM 打分与确定性指标互为校验 |
| 交付 | CLI + FastAPI；Gradio 演示页作加分项 | 基线优先，演示够用即可 |

**不做的事（刻意收窄，避免烂尾）**：多租户权限、生产 K8s、GraphRAG 全量、多模态 VLM 深度集成（可留作 roadmap）。

---

## 3. 总体架构

```text
┌─────────────────────────────────────────────────────────────────┐
│                     CLI  /  FastAPI  /  (Gradio Demo)            │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│              Orchestrator（Query 改写 · 检索 · 重排 · 合成）         │
└───────┬───────────────────────────────┬─────────────────────────┘
        │ 在线                          │ 离线
┌───────▼───────────────┐     ┌─────────▼─────────────────────────┐
│  Hybrid Retriever     │     │  Ingest Pipeline                  │
│  Dense+BM25+RRF      │     │  多源→归一化→分块→LLM元数据→Embed→入库│
│  → Rerank → top-n     │     │  飞书API · doc(x) · PDF(PyMuPDF)            │
└───────┬───────────────┘     └─────────┬─────────────────────────┘
        │                               │
┌───────▼───────────────────────────────▼─────────────────────────┐
│   Qdrant（向量+全文） · SQLite/JSON 元数据 · data/raw 原文件         │
└───────────────────────────────┬─────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│  OpenAI 兼容 LLM API · BGE-M3（Dense+Sparse）· bge-reranker（本地/API）│
└─────────────────────────────────────────────────────────────────┘
```

**五条设计原则（面试可背）**

1. **接入克制**：born-digital 简单版面快通道已够，不为想象中的难例引重依赖——MinerU 与飞书 API 都是条件触发。  
2. **检索块 ≠ 合成块**：小块/句级召回，父块/窗口进 LLM。  
3. **默认 Hybrid**：纯 Dense 搞不定条款号、文号、专名，Dense+BM25 双路是底线。  
4. **引用可追溯**：答案 → `[1][2]` → `doc/page/block`。  
5. **评估驱动**：改任何参数先跑黄金集，拒绝「感觉变好」。

---

## 4. 技术选型对照（开源最佳实践）

| 层 | 首选 | 备选 | 参考项目里的做法 |
|----|------|------|------------------|
| 接入 | **批量导出 PDF + doc(x)** | 飞书 OpenAPI（条件触发） | 采集不是瓶颈，不引审批依赖 |
| 文件解析 | **mammoth + LibreOffice headless（doc/docx）、PyMuPDF（PDF）** | MinerU（仅扫描件兜底） | born-digital 语料，轻依赖 |
| 编排 | **自研薄编排** | LlamaIndex（未来局部引入） | 模块边界对齐框架抽象，保留逃生门 |
| 向量库 | Qdrant（dense+sparse+RRF） | Milvus 2.4+（内置 BM25） | 多路召回 + 融合重排 |
| 评估 | RAGAS | DeepEval | Faithfulness / Context Precision |
| 参考架构 | Cognita、FlashRAG | LangGraph | 可组装流水线、可复现实验 |

**直接借鉴的项目**

- [RAGFlow](https://github.com/infiniflow/ragflow)：模板化分块、引用可视化、多路召回  
- [MinerU](https://github.com/opendatalab/MinerU)：版面/表格/公式/OCR、page-block locator  
- [Docling](https://github.com/docling-project/docling)：统一文档表示、本地可跑  
- [LlamaIndex Production RAG](https://docs.llamaindex.ai/)：解耦检索/合成、结构化检索、重排  
- [RAGAS](https://github.com/explodinggradients/ragas)：标准评估指标  

---

## 5. 核心实现要点

### 5.1 离线 Ingest

```text
PDF（飞书批量导出等）→ PyMuPDF：文本块 + find_tables 重建带框表格
doc/docx            → LibreOffice headless 归一 → mammoth（保留标题结构）
（可选）飞书 OpenAPI → docx blocks 原生结构（触发线见 §1 面试表）
    → 统一中间 JSON（blocks: type/heading_level/text；PDF 附 page/bbox）
    → 结构感知分块
    → 元数据（doc 级 LLM 抽取：date/meeting_type/attendees/topics/decisions +
      chunk 级：title, section_path, page, block_type）
    → Embedding
    → Qdrant upsert + 元数据落盘
```

元数据以 LLM 抽取为主；文件名日期（会议纪要命名惯例）用正则解析作第二信号交叉校验；增量靠重下 + sha256 去重（几千份内无压力）。

**分块默认**

| 规则 | 值 |
|------|-----|
| 正文块 | 300–600 汉字 / ~400–800 tokens，overlap 10–15% |
| 切分依据 | 会议记录按议题/发言人段落优先；通用走标题层级 → 段落 → 滑动窗口 |
| 表格/公式 | 整块，禁止硬切 |
| 必带元数据 | `doc_id, page, section_path, block_type` |

Qdrant 侧为 `doc_id / page / block_type` 建 payload index，支撑元数据过滤与规模化检索。

### 5.2 在线 Query

```text
问题 → (可选)改写/关键词 + 实体/时间识别
     →（可选）元数据过滤（date / attendee / topic）
     → Dense top-k + BM25 top-k（Qdrant full-text，jieba 预分词）
     → RRF 融合 (k=60)
     → Rerank 取 5–8 块
     → 扩父块 / 去重
     → LLM：仅依据上下文 + 编号引用 + 不足则拒答
```

> Phase 1 正选：BM25 为 hybrid 第二路（已查证：SiliconFlow API 不返回 sparse 权重，2026-09）。learned sparse（火山引擎 BGE-M3 / 本地 GPU FlagEmbedding）列为 Phase 2 消融实验。

> 跨文档聚合题（「关于 X 我们做过哪些决定」）：检索结果按 doc_id 聚合去重后再合成，引用按文档归组。

### 5.3 评估闭环（求职亮点）

黄金集 **终版 ≥ 50 条**（v1 现为 64 条：v0 从 30 起步，每题型 ≥ 6），类型覆盖：

- 单会事实题（某次会上说了/定了什么）  
- 决议确认题（讨论过但未决定——决议/讨论区分，faithfulness 专项）  
- 时间/参会人限定题（测元数据过滤增益）  
- 跨文档聚合题（「关于 X 我们做过哪些决定」）  
- 术语/文号/编号题（测 Sparse/BM25 价值）  
- 少量表格数值题（预算表等）  
- **无答案题**（未讨论过的议题，测拒答）  

| 指标 | 目标（示例） |
|------|----------------|
| Context Recall@10 | ≥ 0.80 |
| Faithfulness | ≥ 0.85 |
| Answer Relevancy | ≥ 0.80 |
| 事实题 exact-match（日期/数值/文号类） | ≥ 0.85 |
| 无答案拒答率 | ≥ 0.90 |
| P95 延迟 | ≤ 8s |

**必须能展示的对比实验（简历数字来源）**

1. 纯 Dense vs Dense+BM25 vs Dense+Sparse（Recall@10，三组）  
2. 固定 512 切 vs 结构感知分块（Faithfulness）  
3. 无 Rerank vs 有 Rerank（MRR / nDCG）  
4. 无引用约束 vs 强制引用（人审 Citation Accuracy）
5. 无元数据过滤 vs 有元数据过滤（时间/参会人限定题 Recall）

**已跑出的结果（黄金集 v1 · 62 条 · 1121 篇语料）**

消融 #1 纯 Dense vs Hybrid（Dense+BM25+RRF，top-8）：

| 指标 | 纯 Dense | Hybrid | Δ |
|------|---------|--------|---|
| Recall@5 | 0.836 | **0.909** | +7.3pt |
| Recall@8 | 0.891 | **0.945** | +5.5pt |
| MRR | 0.764 | **0.794** | +3.0pt |

分题型（Recall@8）：fact 1.00→1.00 · open_discussion 1.00→1.00 · decision 0.75→0.88 ·
term 0.75→0.83 · time_filter 0.80→1.00 · cross_doc 1.00→1.00
→ Hybrid 增益集中在精确匹配类（decision / term / time_filter）。

**跨文档聚合题：度量修正与预算规律**

- **度量陷阱**：来源集合曾只取 `@姓名：` 提及的文档（黄日航 8 篇 vs 全库 **205 篇**含名），
  把正确检索判为未命中。修正为「全部含名文档」后 Recall 类指标对聚合题失去区分度
  → 改用**文档覆盖率**（检索到的来源文档数 / 答案集大小）。
- **覆盖率上限 = top_n / 答案集大小**：康少云（56 篇）top-8 上限 0.14，实测 0.12 已达上限 84%。
- **预算扫描**（聚合检索，按文档去重）：cross_doc 0.13（top-8）→ 0.22（top-15）→ 0.35（top-30）；
  time_filter 0.37 → 0.68 → 0.82。
- **实体聚焦查询**（剥离「公司文档里出现过哪些讨论或安排」模板话术，只用人名检索）：
  周碧玉 0.06→0.22、何李健 0.10→0.21。

消融 #5 元数据过滤（时间限定题 · 实体查询 + top-15 聚合）：

| 指标 | 无过滤 | 年份过滤 | Δ |
|------|-------|---------|-----|
| 平均文档覆盖率 | 0.61 | **0.96** | +35pt |

→ 结论：**聚合题的正确形态 = 实体聚焦查询 + 宽检索预算 + 元数据过滤**，三者缺一不可；
这也解释了为什么单纯换融合策略（Dense↔Hybrid）对聚合题无效。

**评估可信度口径（应对「RAGAS 是 LLM 打分，可信吗」）**

- judge 模型与版本固定，temperature=0；每个指标跑两遍看方差  
- 只报告相对变化（before/after），不吹绝对分  
- 人工抽查 15–20% 校准 judge  
- 日期/数值/文号类事实题用 exact-match / 包含匹配，不依赖 judge  

**黄金集构造的两次迭代（失败案例分析素材）**

- v0：跨文档题用词频选词 → 选出「处理结果/本处/case」等泛词与导出残留 token，
  来源集合无意义（Recall 0.25）。**根因是题目质量问题，不是检索问题。**
- v1：改为**结构信号抽人名**（`部门@姓名：` / 提案者 / 主持），并用 `@` 模式做交叉验证提纯
  → 18 个零噪声真人名，构造出有区分度的跨文档/时间题。零 LLM 成本（`gen-gold --programmatic-only`）。
- 同类问题：解析层碎化 PDF（每字符一行，4.5%）导致 grounding 校验 30/44 违规——修复行重建后归零。

---

## 6. 交付物与「面试包」

### 代码仓库结构

```text
RAG/
├── PLAN.md
├── README.md                 # 架构图 + 快速开始 + 评估结果表
├── pyproject.toml            # uv 管理
├── configs/default.yaml
├── data/{raw,parsed,eval}/
├── src/doc_rag/
│   ├── ingest/{feishu,office,pdf,chunker,pipeline}.py
│   ├── retrieve/{hybrid,rerank,rewrite}.py
│   ├── generate/{prompts,synthesizer}.py
│   ├── eval/ragas_runner.py
│   ├── api/main.py
│   └── cli.py
├── notebooks/01_ingest_compare.ipynb  # docx/PDF 接入与表格重建对比
├── notebooks/02_ablation.ipynb        # 消融实验
└── tests/
```

### 面试可展示清单

- [ ] `README` 架构图 + 30 秒电梯演讲  
- [ ] 多源接入对比示例（飞书 blocks / docx / PDF 的解析结果）  
- [ ] 检索消融表（Hybrid / Chunk / Rerank）  
- [ ] RAGAS 基线 vs 优化后  
- [ ] 一次带引用的真实问答（点得到页码）  
- [ ] 已知不足与下一步（GraphRAG、VLM 图表、增量更新）
- [ ] 2–3 个失败案例分析（解析坏例 → 定位 → 修复 → 指标变化）
- [ ] 细粒度 commit 历史（应对「AI 代做」质疑的第一证据）

### CLI 验收命令

```bash
uv run doc-rag ingest data/raw --kb demo
uv run doc-rag query --kb demo "关于供应商预付款，我们做过哪些决定？"
uv run doc-rag eval --kb demo --gold data/eval/gold.json
uv run doc-rag serve   # FastAPI :8000
```

---

## 7. 实施路线图（按周）

### Phase 0 — 叙事对齐与样本（3–5 天）

- 语料：公司文档库抽样，选 1 个代表性文档域（别选百科式纯文本）  
- 从首批 ~300 份抽 20–40 份做画像：来源构成（飞书/doc/PDF）、模板异构度、表格/图片占比  
- 表格解析质量抽查：find_tables 带框表格重建效果（画像表格占比 + 抽 5 份人审）  
- 黄金集 v0：≥ 30 条  

**产出**：语料画像统计、表格解析抽查笔记、样本、评测集 v0

### Phase 1 — 可演示基线（1–2 周）

- 项目骨架（uv + 配置 + OpenAI 兼容 client）  
- Ingest：双路接入（PDF / doc(x)）+ 统一中间 JSON + 结构分块 + Qdrant；文件 sha256 去重 + 断点续跑；doc 级 LLM 元数据抽取基础版（date / topics）  
- Query：Hybrid + RRF + 简单 Rerank + LLM 引用回答  
- CLI 三件套：`ingest` / `query` / `eval`  
- 跑出 **第一版 RAGAS 数字**  

**产出**：能问答、能出分，故事有「before」

### Phase 2 — 优化与讲清（1–2 周）

- Query 改写 + 实体/时间识别、元数据过滤、父块扩展、BGE Reranker、元数据全量字段（参会人/决议）  
- 跑完消融（含 Dense vs +BM25 vs +Sparse 三组检索对比），图表进 README  
- FastAPI；可选 Gradio 上传+对话+引用  
- README 面试版定稿  

**产出**：可对外 demo，有 after 数字

### Phase 3 — 加分项（可选）

- 表格专用策略 / 图片 caption 入库  
- 全量扩到几千份 + 压测调优（ingest 吞吐 / 查询 P95）  
- LightRAG/GraphRAG 跨文档题  
- 简单反馈回路（点赞入评估集）

---

## 8. 风险与对策

| 风险 | 对策 |
|------|------|
| 导出 PDF 的表格/标题解析质量不达标 | find_tables + 画像抽样人审；不足率过高才触发飞书 OpenAPI 路线 |
| 范围膨胀烂尾 | 严格 Phase 1 验收；GraphRAG/UI 全放 Phase 3 |
| API 费用 | 小模型起步；ingest 批处理；query 缓存 |
| 评估集太少被质疑 | 每类型至少 8–10 条；公开构造过程 |
| 抄袭感（和教程太像） | 强调：分层解析路由、消融数字、拒答策略、可复现 eval |
| 被质疑「AI 代做」 | 细粒度 commit 历史；白板讲清架构与每个取舍；消融可复现 |
| 敏感批次出域审批 | 已确认走云端 API；OpenAI 兼容 adapter 保留切内部网关的能力 |
| BGE-M3 sparse 的 API 不暴露 | 已查证（2026-09）：SiliconFlow 只返回 dense。Phase 1 直接走 Dense+BM25(jieba)；sparse 列为 Phase 2 实验（火山引擎 BGE-M3 / 本地 GPU） |
| 元数据抽取质量不稳 | 日期用正则 + LLM 兜底；参会人/议题抽 10–15% 人审；抽取 prompt 纳入评估 |
| 同质语料区分度低 | 元数据过滤 + Rerank；跨文档聚合题专项评估 |
| .doc 老格式转换失真 | LibreOffice headless 归一后抽 5–10% 人审；异常件单独处理 |

---

## 9. 立即可做的 3 件事

1. 云端 API 已确认：锁定 LLM/Embedding Key，实测 BGE-M3 是否返回 sparse 权重  
2. 把批量导出的 PDF 放进 data/raw/，跑 `uv run doc-rag profile`；抽 5 份含表格的人审解析质量  
3. 画像合格即进 Phase 1：Embed + Qdrant 入库（骨架已就绪）  

---

## 10. 参考

- LlamaIndex — Production RAG（decouple chunks、structured retrieval、rerank）  
- RAGFlow — DeepDoc、模板分块、多路召回  
- MinerU / Docling — PDF → LLM-ready 结构化  
- RAGAS — RAG 评估指标  
- BGE-M3 / BGE Reranker — 中文友好多路表示与重排  

---

*画像与人审通过后，直接进 Phase 1 入库与检索（骨架已就绪）。*
