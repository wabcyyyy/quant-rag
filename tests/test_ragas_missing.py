"""N6：RAGAS 轨必须报判分缺失——「n=请求条数、均值=幸存者均值」是假读数。

实锤：`data/eval/ragas_qwen_trial6.json` 的 `summary.n=6` 而 `faithfulness=1.0`
是 5 条的均值（q054 超时为 null，且 by_type 里 time_filter 整类消失）。当年靠
人工核对题型分布才发现（configs/default.yaml:207-210 的教训），代码里没有任何
机制阻止它再次发生——本文件把三件事钉成断言：
1. summary 必须带 n_scored / n_missing（按题型分层）；
2. 缺失率超阈值 → 硬拒出结论（沿 judge 缓存初始化失败硬拒的先例）；
3. 缺失集中在单一题型（整体率达标、题型层超阈）也必须拒——那正是当年的实际形态。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag.eval.runner import RagasMissingError, _ragas_score_audit

FAITH = SimpleNamespace(name="faithfulness")


def _per_item(spec):
    return [
        {"id": i, "type": t, "faithfulness": v}  # v=None = judge 未判出
        for i, t, v in spec
    ]


def test_reports_scored_and_missing_trial6_shape():
    """夹具 = ragas_qwen_trial6.json 的形状：n=6、1 条 null → n_scored=5、n_missing=1。"""
    per_item = _per_item(
        [
            ("q001", "fact", 1.0),
            ("q002", "fact", 1.0),
            ("q003", "fact", 1.0),
            ("q004", "decision", 1.0),
            ("q005", "decision", 1.0),
            ("q054", "time_filter", None),
        ]
    )
    audit = _ragas_score_audit(per_item, [FAITH], max_missing_rate=0.2)
    assert audit["n_scored"] == {"faithfulness": 5}
    assert audit["n_missing"] == {"faithfulness": 1}
    assert audit["missing_rate"] == {"faithfulness": round(1 / 6, 4)}
    # 缺失的题型去向必须可见：trial6 的 by_type 里 time_filter 整类消失就是这个盲区
    assert audit["n_missing_by_type"] == {"faithfulness:time_filter": 1}


def test_refuses_when_overall_missing_rate_exceeds_threshold():
    """2/6 = 33% > 阈值 20%：不许拿 4 条幸存者的均值冒充 6 条的结论。"""
    per_item = _per_item(
        [
            ("q001", "fact", 1.0),
            ("q002", "fact", 1.0),
            ("q003", "fact", 1.0),
            ("q004", "fact", 1.0),
            ("q005", "decision", None),
            ("q006", "decision", None),
        ]
    )
    with pytest.raises(RagasMissingError, match="faithfulness"):
        _ragas_score_audit(per_item, [FAITH], max_missing_rate=0.2)


def test_refuses_when_missing_concentrates_in_one_type():
    """整体 9.5% 达标、time_filter 层 100% 缺失 → 必须拒。

    configs/default.yaml:207-210 记录的真实事故形态：60s 超时掐掉的全是最长上下文
    那批条目，缺失集中在同一题型，均值被幸存者拉高——只看整体率拦不住它。
    """
    spec = [(f"q{i:03d}", "fact", 1.0) for i in range(1, 58)]
    spec += [(f"t{i:03d}", "time_filter", None) for i in range(1, 7)]
    per_item = _per_item(spec)
    assert len(per_item) == 63
    with pytest.raises(RagasMissingError, match="time_filter"):
        _ragas_score_audit(per_item, [FAITH], max_missing_rate=0.2)


def test_small_stratum_does_not_trigger_stratum_refusal():
    """题型层检查只对 n≥5 的层生效：2 条里缺 1 条按整体率判，不为小样本加戏。"""
    per_item = _per_item(
        [(f"q{i:03d}", "fact", 1.0) for i in range(1, 21)]
        + [("t001", "time_filter", 1.0), ("t002", "time_filter", None)]
    )
    audit = _ragas_score_audit(per_item, [FAITH], max_missing_rate=0.2)
    assert audit["n_missing"] == {"faithfulness": 1}  # 1/22 = 4.5% ≤ 20%，只报不拒
