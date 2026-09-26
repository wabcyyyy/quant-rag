# 一键复现指南（`doc-rag repro`）

> 目标：**clone 之后两条命令跑出头条数字**，不需要私有语料、不需要读 PLAN。
> 适用范围：公开合成语料（320 篇虚构会议纪要）+ 公开黄金集核心集（74 条）。

## 1. 前置

| 项 | 要求 |
|---|---|
| Python | 3.12（`uv sync` 会按 `pyproject.toml` 装） |
| Qdrant | `docker compose up -d`（镜像钉 `v1.19.1`） |
| 密钥 | `.env` 里填 DeepSeek（合成/改写/judge）+ SiliconFlow（嵌入/重排）——见 `.env.example` |
| 联网 | 需要：嵌入与重排走 API。跑 `--answers` 才会用到合成模型 |
| OCR（可选但影响读数） | `uv sync --extra ocr`——不装则 **4 篇无文本层的扫描件不入库**（320→316 篇 / 348→344 块），
实测 Hit@5 从 0.9394 掉到 0.9242（≈1.5pt）。要完全对齐头条数字就装上 |

缺密钥时检索侧仍可跑通一半（入库要嵌入），**不要**把 key 写进任何被 git 跟踪的文件。

## 2. 两条命令

```bash
docker compose up -d         # ① 起 Qdrant
uv run doc-rag repro         # ② 入库公开语料 → 检索侧基线 → 打印读数表
```

`repro` 默认做三件事，且**默认值全部指向公开产物**：

1. 入库 `data/sample_parsed/s3`（320 篇公开合成语料）→ collection **`doc_rag_repro`**
   （与生产库 `doc_rag_sample` 隔离，不污染既有基线）；
2. 跑 `data/eval/gold_core.json` 的**检索侧**基线（`--retrieval-only --rerank`，零 LLM 调用）；
3. 打印读数表并落盘 `data/eval/repro_retrieval.json`。

加 `--answers` 会再跑一遍答案侧基线（`--rewrite --rerank`，**真实计费**，74 条约 ¥0.5）。

## 3. 预期读数（2026-09-26 冻结 `1d53ea329bf4` 实测）

**检索侧**（`--retrieval-only --rerank`，n=74）：

| 指标 | 值 |
|---|---|
| Hit@5 / @8 | 0.9545 / 0.9848 |
| Recall@清单（上限） | 0.9779（0.9942） |
| MRR | 0.7137 |
| nDCG@8 | 0.7811 |

**答案侧**（`--rewrite --rerank`，n=74；`--answers`）：

| 指标 | 值 |
|---|---|
| 严格关键词准确率 | 0.9394 |
| 要点召回（聚合题） | 0.6694（n=31） |
| 拒答正确率 / 引用有效率 | 1.0 / 1.0 |

> **预期读数是一个区间，不是一个字面值**：`repro` 走的是**从头入库的独立 collection**
> （collection 名进指纹，`index_fp` 必然与冻结库不同），两次独立入库的向量带 API 浮点
> 噪声（**解析产物逐字相同**，差异全在嵌入侧）。实测：主仓 0.9394 vs 干净 clone 0.9242，
> 逐条对账**只差 1 条**（c069 的 gold 篇 rank 2 → 掉出 top-8），落在检索地板
> （0~2 条 / ≤1.1pt）内。历史上同语料全量重嵌入测到过 14/72 条检回集合不同——
> 所以：**差 1~2 条属正常，差更多才值得查**。这也是「跨 `freeze_id` 禁止并排报数」
> 这条纪律的实体（同结构、不同向量，指纹当前看不见）。

## 4. 判读门槛（三条噪声地板，2026-09-26 公开底座重测）

| 指标 | 地板 |
|---|---|
| doc 级检索（Hit@5） | 0~2 条（≤1.1pt） |
| faithfulness（judge 轨） | 0.4pt |
| **keypoint_recall（聚合子集）** | **7.1pt** |

**差异小于地板的不要下结论**——尤其 keypoint：聚合题要点判据的 run-to-run 抖动
在新底座上是 7.1pt（旧公司语料的 3.10pt 不迁移）。

## 5. 常见坑

| 现象 | 原因 / 处置 |
|---|---|
| `Qdrant 不可达` | 没起容器。`docker compose up -d` |
| `示例语料中间件不存在` | 缺 `data/sample_parsed/s3`：`uv run doc-rag ingest --raw-dir data/sample_raw --parsed-dir data/sample_parsed/s3`（会解析 320 篇，¥0） |
| 入库数字不是 320 篇 | 两种原因：① 用同一 collection 跑过别的语料 → 加 `--recreate`；
② **没装 OCR extra** → 316 篇属预期，`repro` 会当场告警并给出对齐命令 |
| 首次运行比之后慢 | 干净 clone 没有解析产物（gitignored），`repro` 会自动补一次本地解析（零 API 成本）；之后复用那批 JSON |
| 读数比上表低 1~2 条 | 先在**同一状态**上重跑一遍（`--repeat 2`）拿极差；差在地板内不算差异 |
| 答案侧延迟远大于预期 | 缓存命中会伪装低延迟；测真实延迟加 `--fresh-answers`（`repro` 的答案侧默认吃缓存） |
| `synth_fp / index_fp 不匹配`告警 | 你改过仓库或换过 collection——`doc-rag freeze --kb doc_rag_repro` 会落新的冻结记录；**跨 `freeze_id` 禁止并排报数** |

## 6. 要复现**完整**电池（消融 / A4 / RGB）

`repro` 只复现头条两条。完整口径见：

- `docs/guides/metrics.md` —— 判读协议（地板、分母纪律、`freeze_id` 纪律）；
- `docs/design/PLAN.md` §5.7 —— 阶段读数与 C 电池读数表（含每条的命令）；
- `docs/guides/benchmark-rgb.md` —— 外部基准（需先 `bench-rgb-fetch` 拉数据）。