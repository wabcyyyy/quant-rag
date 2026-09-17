"""nDCG@8（T6）回归测试——手算小样例，全离线。"""

from __future__ import annotations

import json
import math
from unittest.mock import Mock

import pytest

from doc_rag.eval import runner
from doc_rag.eval.runner import _ndcg_at_k


def test_perfect_ranking_scores_one():
    assert _ndcg_at_k(["A", "B", "C"], {"A"}) == 1.0


def test_reversed_ranking_discounted():
    # 唯一相关文档排第 3 位：1/log2(4) = 0.5
    assert _ndcg_at_k(["X", "Y", "A"], {"A"}) == 0.5


def test_multiple_relevant_all_on_top_is_one():
    expected = (1 / math.log2(2) + 1 / math.log2(3)) / (
        1 / math.log2(2) + 1 / math.log2(3)
    )
    assert _ndcg_at_k(["A", "B", "C"], {"A", "B"}) == expected


def test_duplicate_docs_counted_once_at_first_hit():
    # 同文档多块只按首个命中位次计一次：A 在第 2 位 → 1/log2(3)
    got = ["X", "A", "A", "A"]
    assert _ndcg_at_k(got, {"A"}) == 1 / math.log2(3)


def test_ideal_is_capped_at_k():
    # 相关文档 60 篇、只检回 8 篇且全在前 8 位 → 完美排序 = 1.0（IDCG 同步截断）
    got = [f"d{i}" for i in range(8)]
    relevant = {f"d{i}" for i in range(60)}
    assert _ndcg_at_k(got, relevant, k=8) == pytest.approx(1.0)


def test_no_relevant_returns_none():
    assert _ndcg_at_k(["X", "Y"], set()) is None


def test_runner_summary_has_ndcg(tmp_path, monkeypatch):
    """runner 接线：summary 增 ndcg_at_8，与 recall/mrr 同分母（排除无来源题）。"""
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"items": [
        {
            "id": "q001", "type": "fact", "question": "费用？",
            "expected_answer": "67元", "source_doc_ids": ["d1"], "must_contain": ["67元"],
        },
        {
            "id": "q002", "type": "no_answer", "question": "未讨论议题？",
            "expected_answer": "应拒答", "source_doc_ids": [], "must_contain": [],
            "refusable": True,
        },
    ]}), encoding="utf-8")
    retriever = Mock(collection="c", cfg={})
    retriever.retrieve.return_value = [
        {"doc_id": "d1", "title": "报价", "page": 1, "text": "费用67元", "block_type": "text"}
    ]
    synthesizer = Mock()
    synthesizer.answer.return_value = "根据现有文档无法回答"
    monkeypatch.setattr(
        runner, "_build_retriever", Mock(return_value=(retriever, synthesizer))
    )
    results = runner.evaluate(gold, cfg={"retrieval": {}, "llm": {"model": "m"}})
    assert results["summary"]["ndcg_at_8"] == 1.0
    assert results["items"][0]["ndcg_at_8"] == 1.0
    assert results["items"][1]["ndcg_at_8"] is None  # no_answer 不进分母
