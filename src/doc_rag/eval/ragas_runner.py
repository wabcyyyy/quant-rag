"""评估 runner（PLAN §5.3）：RAGAS + 事实题 exact-match 双轨，四～五组消融。

状态：桩（Phase 1 实装）。

可信度口径（应对「RAGAS 是 LLM 打分，可信吗」）：
- judge 模型与版本固定，temperature=0；每个指标跑 config.eval.judge.repeat 遍看方差
- 只报告相对变化（before/after），不吹绝对分
- 人工抽查 15–20% 校准 judge
- 日期/数值/文号类事实题用 exact-match / 包含匹配，不依赖 judge

消融清单（简历数字来源）：
1. 纯 Dense vs Dense+BM25 vs Dense+Sparse（Recall@10，三组）
2. 固定 512 切 vs 结构感知分块（Faithfulness）
3. 无 Rerank vs 有 Rerank（MRR / nDCG）
4. 无引用约束 vs 强制引用（人审 Citation Accuracy）
5. 无元数据过滤 vs 有（时间/参会人限定题 Recall）
"""

from __future__ import annotations

RAGAS_METRICS = ["context_recall", "faithfulness", "answer_relevancy"]


def run(gold_file: str, repeat: int = 2) -> dict:
    raise NotImplementedError(
        "Phase 1 实装：读 gold.json → 构造 Dataset → ragas.evaluate + exact-match"
    )
