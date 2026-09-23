# 词汇表（glossary）

> 定位：**索引式词汇表**——每条一句话定义 + 权威出处。规则与数字不在本文件展开：
> 指标规则在 [metrics.md](metrics.md)，数字与放行判定在 [design/PLAN.md](../design/PLAN.md)。
> 新术语随升级落进来（标 🆕），定义稳定后去掉标记（2026-09-23：升级 2026 的四条已转正）。

## 检索与指标

| 术语 | 一句话定义 | 权威出处 |
|---|---|---|
| Hit@k | gold 来源文档的**首个命中**是否出现在前 k 位——「碰到没有」，不是 Recall | [metrics.md §1](metrics.md) |
| Recall@k（真） | 前k格里捞回的 gold 文档**比例**——「捞全多少」；与 Hit 差 22.5pt 是常态 | [metrics.md §1](metrics.md) |
| retrieved vs contexts | 指标在**未截断**清单上算，LLM 只看截断后那份；两份清单不许合并 | PLAN §3 设计原则 |
| rerank.top_n | **上下文预算**（进 LLM 的块数），不是检索预算；重排本身只排序不截断 | PLAN §5.3 消融 #3 |
| 捕获率（recall_vs_ceiling） | 捞到的 gold / 该清单**本可**捞到的 gold；分母随清单长度变，跨臂必带上限一起读 | [metrics.md](metrics.md) |
| keypoint_recall | 聚合题答案级判据：逐篇要点命中数 / K；重跑噪声地板 ≈3.1pt | PLAN §5.5 E2 |
| 等长护栏 | 两臂清单长度 / 上限不等时，长度敏感指标直接标伪影，不进结论 | `compare.py` |
| 噪声地板 | 同条件重跑的极差；小于地板的差异**没有结论资格**（Faithfulness ≥1.9pt、keypoint ≈3.1pt、doc 级检索指标在库未重嵌入时 =0，重嵌入后实测 ±2 条） | [metrics.md](metrics.md) |
| 正对照 | 往已知坏样本里注入缺陷，验证检查器**不是恒真**（audit-refusals 的方法论） | PLAN §5.3 |
| 配对判读 | 同题配对差 + 自助法 95%CI + 符号检验 + Holm 校正；先看赢/输/平几题 | [metrics.md](metrics.md) |

## 系统与运维

| 术语 | 一句话定义 | 权威出处 |
|---|---|---|
| 幽灵块 | 源文件已删/已改，旧 doc_id 的点仍留在库里且**能被检索命中** | PLAN §5.1 幂等与对账 |
| 分题型 SLO | 聚合题 ≤5s、短答题 ≤8s；双峰延迟下单一 P95 会同时骗两端 | PLAN §5.3 |
| 思考分流 | 按预测题型给 reasoning_effort（聚合关思考 27.7s→2.7s）；档位表在 YAML 不在 .env | PLAN §5.3 |
| 编排唯一装配点 | 五条入口共用 `Orchestrator`，parity 测试锁 prompt 逐字一致 | PLAN §5.4 W1 |
| fail-closed | 未配置 API token → `/query` 一律 503；`/health` 免鉴权供探针 | README「注意」 |

## 本次升级引入

| 术语 | 一句话定义 | 权威出处 |
|---|---|---|
| 两段式合成 | 聚合题路由：top-25 检索 → **逐篇微摘要** → 聚合合成只吃摘要；单发口径不动。**实测双门槛未过，负结果收档不转正**（`enabled: false` 留机制） | [ADR-0002](../design/adr/ADR-0002-聚合题两段式合成.md) · PLAN §5.6 |
| 微摘要（map 段） | 每篇 gold 候选的 ~200 tok 摘要；并行、关思考、走响应缓存、逐篇落盘供重放 | [ADR-0002](../design/adr/ADR-0002-聚合题两段式合成.md) |
| 还债优先 | 本次升级主线：只还文档里已量出的债，不开 speculative 线 | [ADR-0001](../design/adr/ADR-0001-升级范围拍板-还债优先.md) |
| 周次回填 | 「2025年第44周」式文件名/正文日期 → ISO 日期区间，纯代码 backfill payload。**已执行**：doc_date 覆盖率 18.8%→62.5%，q063/q064 翻绿 | [ADR-0001](../design/adr/ADR-0001-升级范围拍板-还债优先.md) · PLAN §5.6 |
