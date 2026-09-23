# 评估口径参考（metrics）

这份文档只回答两件事：**每个数是怎么算出来的**、**什么条件下它才准被读成一个结论**。
具体数字（哪个臂是多少、什么被推翻过、复现命令）一律在 [design/PLAN.md](../design/PLAN.md)——
那里是唯一事实源，本文件不复制任何结果值，只复制**规则**。

代码入口：指标计算在 `src/doc_rag/eval/runner.py`，配对判读在 `src/doc_rag/eval/compare.py`，
判据归一化在 `src/doc_rag/eval/schema.py`。

---

## 0. 先约定五个词

| 词 | 含义 | 为什么必须先约定 |
|---|---|---|
| **臂（arm）** | 一次 `eval` 运行 | 臂之间的差只有**同题配对**才准读；不配对的均值差会把题目难度分布算进效应里 |
| **轨（track）** | 检索级 / 答案级 / RAGAS 级 | 三条轨分母不同，既不互相替代，也不跨轨借显著性 |
| **macro / micro** | macro = 逐条比率取均值；micro = 分子分母各自池化 | 同一批数据两种算法能差 3pt 量级，**报数必须标是哪一个** |
| **地板（floor）** | 同配置重跑时该指标自己会抖多少 | 小于地板的差没有结论资格（见 §8） |
| **门禁族** | `compare-retrieval` 默认一次跑齐、一起进 Holm 校正的那组指标 | 不在族内的指标是展示量，不能拿来放行 |

---

## 1. 指标分层一览

指标名一律用 IR 的标准读法（Hit / Recall / Precision / MRR / nDCG / MAP）。旧名仍被接受，见 §1.5。

| 层 | 指标 | 方向 | 进门禁族 |
|---|---|---|---|
| 检索 | `hit_at_5` `hit_at_8` `hit_at_list` `recall_at_5` `precision_at_5` `map_at_5` `ndcg_at_5` `ndcg_at_8` `recall_at_list` `recall_vs_ceiling` `mrr` | ↑ | 是 |
| 答案 | `answered_ok` `keypoint_recall` | ↑ | 是（后者的 n 必须一起报） |
| 答案（对照） | `strict_keyword_accuracy` `subseq_keyword_accuracy` `answer_grade_counts` | ↑ | 否，只作对照/展示 |
| 拒答与引用 | `refusal_acc` `over_refusal_rate` `over_refusal_gold_rate` `citation_valid_rate` `citation_presence_rate` | ↑ / ↓ | 否（单独读） |
| RAGAS | `faithfulness` `answer_relevancy` | ↑ / 诊断 | 否（judge 轨，走 `compare-ragas`） |
| 延迟与成本 | 分项 ms、分题型 p50/p95、`usage` token | ↓ | 否（SLO 判定，见 §6） |

---

## 1.5 命名：旧名 ↔ 标准名（以及一次真实的改名史）

`--metric` 接受两侧名字，结果文件的 `summary` 里**两个名字都在且同值**（旧键由标准键
派生写出于一处，不可能各自漂移）。派生表在 `eval/schema.py:LEGACY_SUMMARY_KEYS`。

| 旧名（历史结果文件、旧命令） | 标准名 | 它一直是同一个计算吗？ |
|---|---|---|
| `mean_doc_coverage` | `recall_at_list_macro` | 是——整条清单上的文档召回，macro |
| `coverage_ceiling_mean` | `recall_ceiling_macro` | 是——同长度清单下召回的天花板 |
| `coverage_by_type` / `coverage_ceiling_by_type` | `recall_by_type` / `recall_ceiling_by_type` | 是 |
| `hit_within_budget` | `hit_at_list` | 是 |
| `contains_acc` / `contains_acc_subseq` | `strict_keyword_accuracy` / `subseq_keyword_accuracy` | 是 |
| `keypoint_hit_ratio` / `keypoint_hit_mean` | `keypoint_recall` / `keypoint_recall_macro` | 是 |
| `coverage_vs_ceiling`（compare 侧） | `recall_vs_ceiling` | 是，但**名字必须说它是分段的**（见 §2） |

**一次真实的改名**：这个仓库以前有个指标叫 `recall_at_k`，报的其实是「首命中在前 k 位」
= Hit@k。因为逐条只存了首命中位次，**当时根本算不出 Recall@k**（gold 有 56 篇时，
在 8 个槽位上报 Recall 是自欺），于是它被改名成 `hit_at_k`，`recall` 这个名字空出来等着。
护栏写在 `tests/test_eval_metric_definitions.py`。2026-09-21 起逐条落盘 `retrieved_doc_ids`
（去重保序的 ranked 清单），真正的 `recall_at_5` / `precision_at_5` / `map_at_5` 才第一次可算——
所以今天两类名字都在，而且同截断下**必须给出不同的数**（相等就说明有一个是假的，这条已锁进测试）。

---

## 2. 检索层

**分母**：有 gold 来源文档的条目。`no_answer` 那类无来源题**不计入**（它们由 `refusal_acc` 单独评）；
未命中的条目按 0 计入，**不剔除**——剔除会把「找不到」从指标里抹掉。
记号：`#gold` = 该题 gold 文档篇数，`#清单` = 本次检索清单条数（表格里不写 `|x|`，竖线在 GFM 表里是列分隔符）。

**统一截断 K=5**：标准族只有一个 k。选 5 的理由是 LLM 实际只读 6 块（`rerank.top_n`），
@8 里有两三格模型根本没看过；而 @10 在 8 格清单上是**假数据点**（`recall_at_10` 在
64/64 条上逐字等于整条清单的召回）。**k 超过清单长度的读数一律不出。**
@8 那两列（`hit_at_8`、`ndcg_at_8`）保留，因为它们是已发布基线的一部分，不重算也不冒充。

| 指标 | 定义 | 读它的时候要注意 |
|---|---|---|
| `hit_at_5` / `hit_at_8` / `hit_at_list` | 首个命中 gold 的位次 ≤ k（`_list` = 整条清单内任意位） | 只回答「碰到没有」，**不回答「捞全没有」**；清单放宽 `hit_at_list` 必然涨，那是长度不是能力 |
| `recall_at_5` | 前 5 个**去重后文档**里命中的 gold 篇数 ÷ `#gold` | 真 Recall。与 Hit@5 的差就是「碰到 vs 捞全」 |
| `precision_at_5` | 前 5 个去重文档里命中的篇数 ÷ **5** | 分母是 k 不是块数。**单文档题的满分就是 0.2**，那不是质量差是定义 |
| `map_at_5` | 每个命中位的 precision 求平均，分母 `min(#gold, 5)` | 唯一同时惩罚「靠后」和「漏捞」 |
| `ndcg_at_5` / `ndcg_at_8` | 二值相关；doc 去重取首位次，IDCG 截到 k | 与 `first_hit_rank` 同源 |
| `recall_at_list` | 整条清单上的文档召回，逐条取均值（macro） | 预算相对读数；**跨臂比较用固定 k 的列**，它只回答「这份清单用满了没」 |
| `recall_ceiling` | `min(#gold, #清单) ÷ #gold` | 天花板不是成绩：清单 8 格、gold 56 篇时 0.143 就是满分 |
| `recall_vs_ceiling` | 逐条「召回 ÷ 天花板」再取均值 | **分段指标**：`#清单 ≤ #gold` 时数值上等于 Precision@清单，`#gold < #清单` 时退化成召回。另注意它是逐条归一，**不是两个均值相除** |
| `mrr` | 命中条目记 `1/位次`，未命中记 0 | 只看首个命中 |
| `list_len` | 清单/上下文的块数 min/max | 两臂可比性的自证字段，不是质量指标 |

> 实测（全量 72 条 · 生产路径 · `results_20260921_200212.json`）：
> `Hit@5 0.9219` / `Recall@5 0.6968` / `Precision@5 0.3969` / `nDCG@5 0.8422` /
> `MAP@5 0.8131`。分题型才有信息量：fact 的 Precision@5 = 0.2（定义如此），
> cross_doc 的 Precision@5 = **0.95** 而 Recall@5 = **0.1598**（槽位几乎没浪费，
> 低召回是 gold 中位 37 篇造成的），term 的 Hit@5 只有 **0.8333**（唯一连碰到都做不全的题型）。

---

## 3. 答案层

判据全部是**字符串匹配**，零 LLM，所以判据本身确定；抖动只来自答案文本（→ §8 的生成地板）。

| 指标 | 定义 | 限制（这条最重要） |
|---|---|---|
| `answered_ok` | 可答题：`must_contain` **全部**逐字命中；可拒答题：命中拒答措辞 | **跨题型语义不同**——在可拒答题上它量的是「有没有正确拒」，不是内容对不对。别把两类题的它平均在一起 |
| `strict_keyword_accuracy`（旧 `contains_acc`） | = `answered_ok` 在可答题上的比率（分母 = 非拒答且有判据的条目） | 聚合题上**饱和**：1 个关键词对 10~56 篇答案集，「答出 2 篇」与「答出 18 篇」同分 |
| `subseq_keyword_accuracy`（旧 `contains_acc_subseq`） | 字符子序列容忍版 | 假阳性**无上界**（「通过决议」能在无关句里命中）。留着只为证伪「严格口径在惩罚改写」，不是独立信号 |
| `keypoint_recall`（旧 `keypoint_hit_ratio`） | 逐篇要点命中 `hits / K`，**macro**；`K=0` 判 **None 不进分母** | 判据是逐字命中，所以它是「有没有把这条抄进清单」，不是「有没有理解」。绝对值只配横向比臂，不配当「答对了多少」 |
| `answer_grade_counts` | `full`：`hits ≥ ceil(0.8·K)`；`half`：`hits > 0`；`zero`：0 | 阈值未校准，**只展示，不进门禁** |

要点归一化（出题侧与判分侧必须同一个函数）：去掉所有空白，去掉 `@` `_` `\` `*` 四类装饰字符。
它救回过一批假阴性（源文档把 markdown 转义带进正文），也遗留三类已知假阴性——
列表编号、主谓间插入动词、人名与其主张被并列项目符号隔开——量级见 PLAN §5.5 的 ①。

---

## 4. 拒答与引用层

| 指标 | 定义 | 为什么是两条而不是一条 |
|---|---|---|
| `refusal_acc` | 可拒答题上答案命中拒答措辞的比例 | **只查措辞**，看不见内容——一句编造的「无法回答」也算对 |
| `over_refusal_rate` | 可答题上：判分未过 + 命中拒答措辞 + **上下文含 `must_contain` 原话** | 严格口径：漏掉「检回了正确文档、但那块没含那句原话」这一类真过度拒答 |
| `over_refusal_gold_rate` | 同上，但条件换成 **gold 文档已进上下文** | 宽口径：与上一条一起报才看得全（实测两者会一个报 0、一个抓出条目） |
| `citation_presence_rate` | 答案里出现 `[n]` 编号引用的比例 | 有没有标 |
| `citation_valid_rate` | 出现引用时，**全部**编号落在 `1..#上下文块数` 的比例 | 标得对不对；分母只含有引用的条目，所以它和 presence 一起读才有意义 |

---

## 5. RAGAS 轨（第二轨，独立命令 `compare-ragas`）

| 指标 | 口径 | 已知失效模式 |
|---|---|---|
| `faithfulness` | 答案里可归因到上下文的句子占比 | 枚举拆句、正确拒答的套话、prompt 要求写的「文档没记载…」hedge 这三种形态会被系统性扣错分。**单题分值不可复现**，只有配对差可读；judge 必须看到与 LLM **完全相同**的上下文串（含 `[n]（文档名 第p页）` 前缀），否则归属陈述会被判成编造 |
| `answer_relevancy` | 答案反向生成问题与原问题的嵌入相似度 | 中文下把正确拒答与忠实的「未决」表述判成 noncommittal 记 0 分 → **仅诊断，不进质量门禁** |

可拒答题按定义排除在 faithfulness 之外——所以「拒答里有没有编造」这件事这条轨看不见，
要用 `audit-refusals` 单独审。

---

## 6. 延迟与成本

- 分项：`rewrite` / `retrieve` / `rerank` / `retrieval_total`（不含 LLM）/ `synthesize` / `total`，
  外加 `synth_cached` 标记。汇总给 p50 / p95 / max，并**分题型**各算一份。
- SLO：全量端到端 `target_p95_ms = 8000`；现行口径是分题型——聚合 ≤5s、短答 ≤8s。
- **三条纪律**：
  1. 延迟必须 `--fresh-answers` 关缓存测。缓存命中的合成是毫秒级，不是延迟。
  2. 端到端读数**必须先对 `retrieve` 分项**：该分项实测有 62~127s 量级的端点停顿
     （`embedder.py` 的 `timeout=60.0` × `attempts=3`），不先扣除就会把它读成链路复杂度。
  3. 慢条目要能用 `usage`（prompt / completion / **reasoning** token）解释；
     「28s 但答案仅 308 字」这类现象只有配上 reasoning token 才算归因完成。

---

## 7. 判读规则（差值怎样才能成为结论）

1. **同题配对**：只取两臂都有值的条目逐条相减。
2. **95%CI**：对配对差做 bootstrap，`n = 5000`、`seed = 42`（固定种子，可复现）。
3. **精确符号检验**：赢/输/平计数，独立于 CI 给一个不依赖正态假设的读数。
4. **Holm–Bonferroni**：α = 0.05，**家族 = 本次运行的全部配对比较**；分多次运行不会互相校正——
   所以「跑两次各挑一个显著」在这套规则下不成立。
5. **`unscored`**：某臂没有任何判据条目时标 `unscored` 并**排除出家族**，不让它稀释校正。
6. **等长护栏**（分轨，措辞不同因为后果不同）：检索级两臂清单不等长 → 差值部分是长度的函数；
   答案级块数不等是**实验变量**，只有两臂 `K` 不同（分母变了）才告警；RAGAS 轨块数不等 →
   单独提示「faithfulness 随上下文变长单调走高，这个差偏向块数多的一臂」。
7. **显著 ≠ 值得**：过完统计这一关，还要过地板（§8）与成本（token / 延迟 / 是否作废基线）。

---

## 8. 三条噪声地板

| 指标 | 地板 | 怎么量出来的 | 实测日期 |
|---|---|---|---|
| doc 级检索指标 | **0**（前提：库未重嵌入） | 同配置连跑两遍，逐条配对 0 赢 / 0 输 / 64 平（块集合会变，doc 级读数不变）。**2026-09-23 全量 re-ingest 重嵌入后前提失效**：backfill 后同状态背靠背两遍，14/72 条检回集合不同、Hit@5 在 0.9375↔0.9688（±2 条 = ±3.1pt）间摆动——近并列文档的边界抖动开始跨越 top-8 集合边界（机制与数字见 PLAN §5.6 Phase 2.4） | 2026-09-19（修订 2026-09-23） |
| `faithfulness` | **≥ 1.9pt** | 同一批答案、同一 judge、temperature=0，全量 n=64 重判一次 | 2026-09-19 |
| `keypoint_hit_ratio` | **3.10pt** | 同配置 `--fresh-answers` 关缓存两遍，输入 token 逐字相同（抖动只来自生成） | 2026-09-21 |

三条**不可互换**：judge 地板不适用于答案轨，答案轨地板不适用于检索轨。
另注意「重跑地板 = 0」只说检索指标是确定性的，它同时意味着——**答案侧的 run-to-run 差异不会有任何检索指标先报警**；而重嵌入之后连检索指标自己都会摆 ±2 条，所以**任何「检索基线重发」必须带同状态重跑极差，单遍读数不再单独具有结论资格**。

---

## 9. 每个结果文件必须能自证什么

`meta` 现有字段：`timestamp` `top_n` `budget`（`fixed:N` 或 `rewrite`）`context_budget`
`collection` `agent`（开关来源/题型/步数/上限）`retrieval`（哪几路 + 消融位）
`rerank_failed` `rewrite_degraded` `rewrite_model` `filter_fallback_n` `filters_applied_n`
`with_answers` `llm_model` `answer_cache` `prompt_version` `prompt_fingerprint`。
RAGAS 侧另记 `judge_model` `judge_base_url` `judge_cross_vendor` `judge_reasoning_effort`
`judge_temperature` `judge_cache` 与 `source_results`。

**逐条 `items[]` 的 schema 版本**：2026-09-21 起多了 `retrieved_doc_ids`（去重保序的 ranked
清单）与 `recall_at_5`、`precision_at_5`、`ndcg_at_5`、`ap_at_5`。**之前的文件没有这些字段**，
`compare-retrieval` 会把它们在旧文件上判成 `unscored`（整臂无值时排除出 Holm 家族，
不稀释校正）——也就是「没测」，不是「测了且没差异」。旧键名与新键名在 `summary` 里同值并存，
读哪个都对，只是别拿旧文件的 `mean_doc_coverage` 去对一个新文件的 `recall_at_5`。

**两个已知洞**（引用旧文件时要自己补证）：

- 不记**黄金集指纹** → 「这条数字跑在 gold v1 还是 v2（有没有 `key_points`）」文件自证不了。
- 不记**合成侧生效的思考档** → 「聚合题关思考」这个前提对任何旧文件都不可证（judge 侧反倒记了）。

---

## 10. 八种常见误读

1. 把 `hit_at_k` 当 Recall 读（它只回答「碰到没有」）。聚合题上二者差得很远：
   实测同一批 20 条 `hit_at_list = 1.0` 而 `recall_at_list = 0.1609`。
2. 拿两个**清单长度不同**的 `recall_at_list` 直接比 → 该读 `recall_vs_ceiling`。
3. 把聚合题的 `strict_keyword_accuracy = 1.0` 读成「答全了」→ 它在 1 个关键词对 56 篇时恒饱和。
4. macro 与 micro 混报（同一批数两种算法可差 3pt 量级）。
5. 把 `refusal_acc` 读成「答得对」→ 它只查措辞；内容要 `audit-refusals`。
6. 用带缓存命中的运行读延迟 → 那是毫秒，不是延迟。
7. 跨题型平均 `answered_ok` → 它在可拒答题上量的是另一件事。
8. 用小于地板的差下结论，或把**分多次运行**的比较当成同一个 Holm 家族。

---

## 11. 相关文档

- [design/PLAN.md](../design/PLAN.md) §5.3（判分与噪声）、§5.5（agentic 期的门槛与实测）——所有结果值与放行判定在这里
- [architecture/system-design.md](../architecture/system-design.md) —— 这些指标背后的链路怎么跑
- [guides/how-it-works.md](how-it-works.md) —— 教学向全链路拆解
