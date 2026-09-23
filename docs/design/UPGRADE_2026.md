# 2026 升级落地方案（执行级实施方案）

> 定位：ADR-0001/0002/0003 拍板后的**执行方案**——每步的命令、产物、验收、成本、停止条件。
> 决策与理由在 ADR；本文只管「怎么做、怎么判、动了什么要重发什么」。
> 结论落地时回写 PLAN.md（新增 §5.6 或并入既有节），本文转为过程材料——不与 PLAN 争夺事实源地位。
> 版本：v1（2026-09-23，经代码与数据自审后细化；v0 为会话内草案，未落盘）

---

## 0. 自审结论（v0 → v1 修订了什么）

对 v0 草案做了代码级（Explore 全量核对）与数据级（本地结果文件重读）自审，**推翻或修正五处**：

| # | v0 的说法 | 事实核验 | v1 修订 |
|---|----------|---------|--------|
| 1 | 「term 是唯一 Hit@5 <0.9 的题型」（引 README 原文） | **错**。逐条数据：Hit@5 未中的可答题共 **5 条**——decision q017、term q019/q020、**time_filter q063/q064**（也是 0.8333）。README 那句「唯一一个连碰到都没碰到」与它自己的分题型表矛盾，是个文档 bug | 诊断范围扩到 5 条；README 措辞列入 Phase 3 文档修正 |
| 2 | 执行顺序：M1 prune → … → M3 两段式 | **顺序错了**。任何库变更（prune --yes / 周次回填）都会让两段式臂与 A6/C25/S12 历史臂不可比——要么违反「新旧不混用」，要么被迫重跑三条臂多花 ¥2–3 | **L2 验收先于库变更**（冻结库窗口）；见 §2 阶段顺序。注意：仅报告的 prune 不变更库（幂等 upsert + 报告），可提前 |
| 3 | 「backfill 只动 payload，基线安全」 | **半对**。set_payload 确实原地合并不重向量（cli.py:1339-1349）；但 doc_date 覆盖率变化会改 time_filter 题的检索结果——那正是目的，但意味着 time_filter 相关读数要重发，不是「安全」 | 基线重发对照表（§4）明确列出：周次回填后 time_filter 全族读数重跑重发 |
| 4 | 隐含假设：`ingest --prune` 是轻量操作 | pipeline 的 sha256 去重**只在批内**（pipeline.py:47-61），全量 re-ingest 会重新解析+重嵌入 3856 块（幂等、≈¥1、约 30 分钟） | Phase 0 先 `--limit 20` 试跑核实行为，再决定全量；成本计入预算 |
| 5 | 两段式引用/判分语义未细化 | 现行 `citation_valid = 1 ≤ n ≤ len(contexts)`（runner.py:378-380），分母是 LLM 所见清单 | 设计定案：**两段式下 item.contexts 仍存原始块**，微摘要存独立字段，reduce 的 `[n]` 编号指原始块——现行引用检查零改动即可用（§3.4） |

自审同时**核实成立**的三条关键假设：聚合路由用预测题型（`predict_type`，与 agent 门控同键）；重排只重排序不截断（parity 测试钉死）；backfill 不重新向量化。另核出一个**潜伏 bug**：`date_from_filename` 的正则会把不带「第」的「2026年42周」误解析成 **2026-04-02**（metadata.py:14 的回溯匹配）——周次解析必须连这个一起修。

## 1. 证据基础（本轮核出的关键事实）

数据侧（`data/eval/results_20260921_200212.json` 逐条）：

- **q063/q064（time_filter 未中）**：`filter_applied=True`、`filter_fallback=False`、年份过滤 2025、`doc_coverage=0.0`、`first_hit_rank=None`，改写后查询为人名（周碧玉/贝佳淼）。机制推断：gold 文档是周次命名会议档案、**没有 doc_date 字段**，被 `doc_date ∈ 2025` 过滤整族排除；过滤后剩余 ≥3 条所以不触发回退。→ **周次回填有了具体靶子和可检验预言：回填后这两条应翻绿（或至少 coverage 离开 0）**。验证法：拿 gold 的 `source_doc_ids` 对 `data/parsed` 标题核对周次格式（Phase 2 第一步）。
- **q019**（term 未中）：问题本身含「2025年第36周」——**周次字符串出现在查询里**，若 gold 块正文不含周次（只在文件名/payload title），BM25 两路都摸不到。修法候选（Phase 2 诊断后定）：标题并入 bm25 分词文本，或查询侧归一。**这是检索侧变更，走配对臂，不拍脑袋改**。
- **q020**（term 未中）：「动议区提到的文件名」——纯词面题，dense 语义不沾边，gold 完全不在 top-8。
- **q017**（decision 未中）：日期限定单会题，同样 gold 不在 top-8。
- gold.json schema：`{meta, items}`；聚合 20 题 = q045–q064（cross_doc 8 + time_filter 12）。

臂文件映射（对照用，均已确认在库）：

| 臂 | 文件 | 口径 |
|----|------|------|
| A6（现基线） | `results_20260920_203312_kc.json` | n=72 · 清单 8 / 上下文 6 · keypoint 重判版 |
| C25（直喂） | `results_20260920_210500_kc.json` | n=72 · 清单 25 / 上下文 25 |
| S12 | `results_20260921_000300.json` | n=20 · 12 块 |
| AGENT | n=20 且 items 带 trace 的 09-21 文件（执行时按 `trace` 非空确认） | policy 层 |
| 答案侧基准（−3.1pt 配对用） | `results_20260919_135225.json` | n=72 · rewrite+rerank · 严格关键词 0.8438 |

代码侧（file:line 见 Explore 核对，要点）：

- 两段式挂点：`orchestrator.py` `answer/answer_stream` 按 `predicted ∈ {cross_doc, time_filter}` 分流（与 agent 门控同键，agent.py:98-108）；map 段的上下文预算要照 agent 先例在 orchestrator.py:238-247 开自己的 budgets 分支——否则被 `min(max_contexts=10, rerank.top_n=6)` 截在 6 块。
- `--fresh-answers` 关的是**进程内全部 LLM 缓存**（含改写/judge），不只答案（cli.py:929-935 → llm.py:66-72）——延迟测量用它是对的，judge 重判要单独 `--fresh-judge` 控制。
- 并行微摘要无现成机制，照 `ingest/indexer.py:223` 的 `ThreadPoolExecutor(8)` 模式套 `chat_timed`（缓存层 RLock 可重入）。
- `latency_ms` 键集合被 parity 测试钉死（test_orchestrator_parity.py:501-509 恰 7 键）——加 map 段键必须同步改该测试。
- doc_date：payload 字段名 `doc_date`、值 "YYYY-MM-DD"、**DATETIME 索引已建**（indexer.py:58）；周次文件名 ingest 侧零解析。
- backfill：`cli.py:1311-1368`，规则重算 `base_meta`（无 LLM）→ `set_payload` 按 doc_id 合并——**扩展 metadata.py 的解析规则后 backfill 即生效，无需新命令**。

## 2. 阶段计划

> 排序原则：**先冻结库窗口做完所有「与历史臂配对」的测量，再做库变更，最后重发基线**。
> 每阶段末尾跑离线门禁四件套：`pytest -q` / `ruff check` / `ruff format --check` / `mypy src`。

### Phase 0 · 只读诊断 + 地基（≈¥0~1，不动库）

| 步 | 动作 | 产物 | 验收 |
|----|------|------|------|
| 0.1 | 5 条未中逐条归因：dump q017/q019/q020/q063/q064 的 `retrieved_doc_ids` vs gold `source_doc_ids`；对 gold 文档核对①标题是否周次格式②gold 块正文是否含查询关键词（BM25 可行性）③dense 相似度位次 | 诊断笔记落 PLAN 新节（§5.6 起） | 每条有归因结论：过滤排除 / 词面不匹配 / 语义不沾边 |
| 0.2 | 构造聚合 20 题子集：`gold.json` 按 type 过滤写 `data/eval/gold_agg20.json`（gitignored，保留 meta 并注明派生） | 子集文件 | id 集合 = q045–q064 |
| 0.3 | 幽灵块清点：先 `ingest --raw-dir data/raw --limit 20` 核实 re-ingest 行为（是否重嵌入），确认成本后全量 `ingest --prune`（**只报告**） | 幽灵篇数/块数落 PLAN §5.1 那段 ⚠ 的替换文本 | 「当前基线库里语料之外的点数」从未知变已知；**不执行 `--yes`** |
| 0.4 | LICENSE（默认 MIT，一行决断） | LICENSE 文件 | — |

**停止条件**：0.3 若 re-ingest 行为异常（解析产物大面积读不通等），停，先修再继续——三道闸是既定纪律。

### Phase 1 · L2 两段式合成（冻结库窗口，≈¥2~4）

| 步 | 动作 | 产物 | 验收 |
|----|------|------|------|
| 1.1 | 机制开发（细节见 §3）：路由分支 + map 并行微摘要 + reduce 合成 + 落盘 + parity `synthesis_route` 维度 + mock 测试 | 代码 + 全离线测试绿 | 四件套门禁绿；五入口 parity 含新维度 |
| 1.2 | 示例库冒烟：`doc_rag_sample` 上构造 2 条聚合题跑通 map→reduce→引用 | 冒烟记录 | 引用能点回原始块；map 失败退化路径走通 |
| 1.3 | 真库验收：20 题子集 × 两段式臂（检索开关与 C25 **逐字同款**：`--top-n 25 --max-contexts 25` + 两段式路由开关——聚合题 prefetch 50 由 `aggregate_pool` 自动生效），`--fresh-answers`；随后 `rescore_keypoints.py` + `compare-retrieval --metric keypoint_recall` 对 A6/C25 配对（A6/C25 为 n=72，按题号交集配对，P2 先例） | 两段式结果文件 + 配对判读 | **ADR-0002 门槛**（≥+10pt 且聚合 p95 ≤8s 双过才转正；见 §3.6 抖动规则） |
| 1.4 | 若点估计落在门槛 ±3.1pt（地板）带内：同配置重跑一遍取极差，再判 | 重跑文件 | 报结论带两次读数 |
| 1.5 | −3.1pt 归因臂（同库窗口顺手）：`eval --rerank`（**无 --rewrite**）72 条 → 与已发布 with-rewrite 基准 `data/eval/results_20260919_135225.json`（0.8438 那轮）配对 `answered_ok` | 对照文件 | 配对前先核基准文件 `summary` 的严格关键词准确率确为 0.8438（防拿错文件）；−3.1pt 归因为「改写 / 噪声」二选一，落 README 悬数字段 |

**停止条件**：1.3 质量不过门槛 → 记负结果收档（ADR-0002 既定分支），**不迭代 prompt 挽救**——单轮结论优先，挽救留给显式重开。

### Phase 2 · 库变更 + L1 弱项修复（≈¥1~2）

前置：Phase 1 的所有配对测量已落盘（此后库才允许变更）。

| 步 | 动作 | 产物 | 验收 |
|----|------|------|------|
| 2.1 | 若 0.3 报告有幽灵块：`ingest --prune --yes`（过三道闸） | 清理记录 | 删点数 = 报告数 |
| 2.2 | 周次解析：`metadata.py` 加「(20\d{2})年第?(\d{1,2})周」→ ISO 周一日期（±1 周精度，年份过滤场景无损）；**修「2026年42周」→2026-04-02 的误解析 bug**；单测锁两例 | 规则 + 测试 | 回归测试绿 |
| 2.3 | `doc-rag backfill` 全量 → 重报 doc_date 覆盖率（18.8% → 预期 ~43%：211+270/1121） | 覆盖率数字 | 增量文档的标题确为周次格式（0.1 已核） |
| 2.4 | 重跑检索读数：`eval --retrieval-only --rewrite --rerank` 72 条全量 → 与已发布四项对照；重点看 q063/q064 是否翻绿、time_filter 全族变化 | 新检索基线文件 | 任何变动数字进入 §4 重发清单；q063/q064 的预言被证实或证伪都如实记录 |
| 2.5 | term 修复（**仅当 0.1 诊断出可修模式**）：按诊断实施（如标题并入 BM25 文本），修复臂 vs 基线臂等长配对 | 配对判读 | n=12 功效声明随行；Hit@5 不倒退、q019/q020 翻绿才保留 |

**停止条件**：2.5 若修复引入其他题型回退（配对输 >2 条），回滚变更、只留诊断记录。

### Phase 3 · 基线重发 + 文档收口（≈¥0.5~1）

| 步 | 动作 | 产物 |
|----|------|------|
| 3.1 | 若库已变更（2.1/2.3 任一执行）：全量 72 条 `--fresh-answers` 重跑答案侧头条；若两段式转正，聚合题口径换新并重判 faithfulness（judge ≈¥0.5） | 新头条数字 |
| 3.2 | 文档对齐：README（含「唯一题型」措辞 bug 修正、悬数字 −3.1pt 结案、time_filter 表更新）、PLAN（§5.6 升级记录 + §7 路线图勾销）、glossary 去 🆕、三条 ADR 状态收口 | 全部文档一致 |
| 3.3 | 按决策粒度提交（LICENSE / 诊断 / 两段式机制 / 验收数据 / 库变更 / 基线重发 分开） | commit 序列 |

## 3. 两段式合成实施细节（L2）

### 3.1 路由与预算

- 开关：`configs/default.yaml` 新节 `synthesis.two_stage`（默认 `enabled: false`，转正才翻），键域对齐 `agent.types`：`types: [cross_doc, time_filter]`。路由信号 = `predict_type(plan)`，与 agent/思考档同键（真题型只做事后核对 `type_matches_gold`）。
- 检索预算：两段式臂的检索开关与 C25 **逐字同款**（`--top-n 25 --max-contexts 25` 固定预算；聚合题 prefetch 50 由 `aggregate_pool` 按 aggregate 标志自动生效）。**变量隔离：与 C25 唯一差异是合成路由**——检索确定性（doc 级重跑地板 = 0），故候选集逐字相同。
- 上下文预算：map 段吃 **retrieved 去重后的文档清单**（照 agent 先例在 orchestrator.py:238-247 加 budgets 分支，不受 `min(max_contexts, rerank.top_n)` 截断）；每篇文档的全部入选块拼为一篇输入。

### 3.2 map 段（逐篇微摘要）

- 输入：该篇全部入选块（含表格块），前缀带 doc 标识；输出 ~100–200 tok：「该篇与问题相关的决定/事实（保留人名、数值、文号原词）；无关则明确输出『无关』」。
- 执行：`ThreadPoolExecutor(8)` 并行 `chat_timed`（照 indexer.py:223 模式）；`reasoning_effort: none`；走响应缓存（键含 messages，天然按（问题×文档×块集）去重）。
- 退化：单篇失败 → 该篇降级用原文首块；失败率 > 30% → 整题退回单发口径并计数（`synthesis_degraded` 机器标志 → `/metrics` + `meta`，全退化中止——沿用 rewrite degraded 先例）。
- 落盘：`items[i].summaries = [{doc_id, text, source_context_idx[], degraded}]`——**重放纪律：reduce 输入可从落盘摘要逐字重建，缺记录值拒绝重放**（照 `rewritten`/trace 先例）。

### 3.3 reduce 段（聚合合成）

- 输入：全部微摘要（含「无关」标记的排除或降权）+ 原始问题；prompt 在现有合成 prompt 基础上改「上下文为逐篇摘要」段落，**引用编号 [n] 指原始块的序号**（摘要携带 source_context_idx）。
- 合成规则 3 不放宽：逐篇枚举陈述，禁止跨篇合并——prompt 显式重申 + `audit-refusals` 式抽查。

### 3.4 判分与引用语义（关键设计）

- **`item.contexts` 仍存原始块**（不是摘要）→ 现行 `citation_valid = 1 ≤ n ≤ len(contexts)` 语义零改动即用；`n_contexts` 口径与历史臂可比。
- 摘要存独立字段（3.2），faithfulness 判分用「LLM 实际所见」= 摘要集 + 原始块编号前缀的重构串——judge 重放只读落盘值。
- keypoint 判据零改动（纯答案字符串匹配，`rescore_keypoints.py` 直接可用）。

### 3.5 parity 与测试清单

- parity 四处：`_cfg()` 加 `synthesis.two_stage` 节；`prompt_spy` 加微摘要 system_prompt 分流；新增照 `test_agent_mode_traces_match_across_all_entries` 的断言（每入口 map 调用数、reduce prompt、degraded 标志逐项相等）；**同步改 501-509 的 latency 键集合断言**（新 `map_ms` 键）。
- 新单测：路由只认预测题型（域外键报错，照 `reasoning_effort_by_type` 先例）；map 退化三级路径；落盘摘要缺失时拒绝重放；`summaries` 不进 `contexts`（防两份清单合并的老错复发）。

### 3.6 延迟门槛的测量口径（防上游抖动误判）

- 主门槛：聚合 20 题端到端 p95 ≤ 8s（`--fresh-answers`）。
- **抖动规则**：单条 `retrieve` 分项 > 1s 视为测量污染（PLAN 已证 embedder 抖动 max 68s 与链路无关）——污染条目重跑，或在报告中双栏呈现（含/不含上游抖动），**门槛判读用干净栏**。
- 分位数按题型报（双峰纪律），n=20 时 p95 ≈ max，报告如实标注。

## 4. 基线重发对照表（哪步动什么，什么作废）

| 变更 | 影响范围 | 动作 |
|------|---------|------|
| 0.3 全量 re-ingest（幂等 upsert） | 无（向量与 payload 逐字不变） | 不重跑 |
| 2.1 prune --yes（若有幽灵） | 检索读数可能微动（幽灵挤出 gold 的情形） | 2.4 重跑检索四项 + K=5 族，重发 |
| 2.2+2.3 周次解析 + backfill | **time_filter 全族**（过滤集变化）；其余题型不动 | 2.4 重跑；README time_filter 行 + 消融 #5 口径更新；doc_date 覆盖率数字更新 |
| 2.5 term 修复（若做） | term 题型 + 全量均值 | 修复臂配对判读后重发 term 行；全量四项重跑 |
| 1.3 两段式转正（若双过门槛） | 聚合题答案侧基线 + 聚合 SLO | README「回答质量」「延迟按题型」两表的聚合行重发；单发其余题型不动 |
| Phase 3.1 头条重跑 | 答案侧全量表 | 逐表列明「重发于库变更后」 |

## 5. 风险与停止条件（汇总）

1. **微摘要信息损失**（最大技术风险）：keypoint 低于 C25 → 负结果收档，不抢救。
2. **map 段延迟超预期**：并行 8 路 + 关思考下单篇 ~1–2s，理论上 wall-clock ≈ 2 轮；若实测 p95 超门槛且质量已过 → 「默认关 + 留机制」（ADR-0002 既定分支），排查是否值得缩 top-25 → top-15（**这是新变量，须另开一轮对照，不在本次内顺手动**）。
3. **预算总超支**：各阶段成本见 §2；累计 > ¥8 即停盘对账（成本纪律：先算账、token 落盘对账）。
4. **离线门禁红**：任何阶段四件套不绿不进下一步。

## 6. 命令速查

```bash
# Phase 0
uv run doc-rag ingest --raw-dir data/raw --limit 20      # 核实 re-ingest 行为
uv run doc-rag ingest --raw-dir data/raw --prune          # 幽灵块清点（只报告）
# 0.2 聚合 20 题子集（写完自查输出 = 20）
uv run --no-sync python -c "
import json
g=json.load(open('data/eval/gold.json',encoding='utf-8'))
sub={**g,'items':[q for q in g['items'] if q['type'] in ('cross_doc','time_filter')]}
sub['meta']={**g.get('meta',{}),'derived_from':'gold.json v2 · agg20 subset for L2 two-stage acceptance'}
json.dump(sub,open('data/eval/gold_agg20.json','w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('items:',len(sub['items']))"
# Phase 1
uv run doc-rag eval --gold data/eval/gold_agg20.json --kb doc_rag_demo \
  --rewrite --rerank --top-n 25 --max-contexts 25 --two-stage --fresh-answers  # 两段式臂（--two-stage 开关名以实现为准）
uv run python scripts/rescore_keypoints.py <两段式结果文件>
uv run doc-rag compare-retrieval --group A6=data/eval/results_20260920_203312_kc.json \
  --group C25=data/eval/results_20260920_210500_kc.json \
  --group 2STAGE=<两段式结果文件>_kc.json --metric keypoint_recall
uv run doc-rag eval --kb doc_rag_demo --gold data/eval/gold.json --rerank --fresh-answers  # −3.1pt 归因臂（无 --rewrite；全量 gold）
# Phase 2
uv run doc-rag ingest --raw-dir data/raw --prune --yes                    # 仅当 0.3 报告有幽灵
uv run doc-rag backfill                                                   # 周次回填（规则改完后）
uv run doc-rag eval --retrieval-only --rewrite --rerank                   # 检索基线重跑
```
