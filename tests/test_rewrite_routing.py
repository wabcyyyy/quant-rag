"""N15：误路由率必须随 meta 自证（`rewrite_routing`）。

`items[].rewrite_aggregate`（改写器的预测）与真题型 `items[].type` 都早已落盘，
算匹配率是零 LLM 成本，但没有任何字段在算——A4 七臂的读数无法自证「这些题真的
走了聚合路」（RGB zh_int 的头条数字就是建立在 12/100 路由臂上的，见审查 N4/N8）。
字段：`meta.rewrite_routing = {n, agg_predicted, agg_gold, matches, recall,
false_trigger, by_type}`。改写退化与失败条没有「预测」可言，不进分母（n 自证）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag.eval import runner as eval_runner
from doc_rag.orchestrator import Result


class _RoutedOrch:
    """plan.aggregate 按题面脚本化；degraded 集合里的题改写退化（无预测）。"""

    def __init__(
        self, agg_by_question: dict[str, bool], degraded: set[str] | None = None
    ):
        self.agg = agg_by_question
        self.degraded = degraded or set()
        self.retriever = type("R", (), {"collection": "t", "cfg": {"mode": "hybrid"}})()

    def answer(self, question, **kw):
        result = Result(
            plan={
                "rewritten": question,
                "filters": None,
                "aggregate": self.agg[question],
                "top_n": None,
                "reason": "t",
                "degraded": question in self.degraded,
            },
            retrieved=[
                {
                    "chunk_id": "d1:1",
                    "doc_id": "d1",
                    "title": "文档",
                    "text": "预算 42 万元",
                    "section_path": [],
                    "page": 1,
                    "block_type": "paragraph",
                }
            ],
            contexts=[
                {
                    "no": 1,
                    "text": "预算 42 万元",
                    "doc": "文档",
                    "page": 1,
                    "doc_id": "d1",
                }
            ],
            citations=[{"no": 1, "doc": "文档", "page": 1, "doc_id": "d1"}],
            answer="预算 42 万元 [1]",
            retrieval_empty=False,
        )
        result.latency_ms = {
            "rewrite": 1.0,
            "retrieve": 1.0,
            "rerank": 1.0,
            "retrieval_total": 3.0,
            "synthesize": None,
            "synth_cached": None,
            "total": 3.0,
        }
        result.synth_meta = None
        return result


def _gold(tmp_path):
    p = tmp_path / "gold.json"
    items = [
        ("q1", "fact", "单点1"),
        ("q2", "fact", "单点2"),
        ("q3", "cross_doc", "聚合1"),
        ("q4", "cross_doc", "聚合2"),
        ("q5", "fact", "退化条"),
    ]
    p.write_text(
        json.dumps(
            {
                "meta": {"gold_version": "t"},
                "items": [
                    {
                        "id": i,
                        "type": t,
                        "question": q,
                        "expected_answer": "答案",
                        "must_contain": ["42"],
                        "source_doc_ids": ["d1"],
                        "refusable": False,
                    }
                    for i, t, q in items
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return p


def test_rewrite_routing_self_reported_in_meta(tmp_path, monkeypatch):
    """预测 vs 真题型的混淆矩阵：recall / false_trigger 必须可算、按题型分层。"""
    orch = _RoutedOrch(
        agg_by_question={
            "单点1": False,  # fact 未触发 → match
            "单点2": True,  # fact 误触发 → false_trigger
            "聚合1": True,  # cross_doc 触发 → match
            "聚合2": True,  # cross_doc 触发 → match
            "退化条": False,  # degraded：无预测，不进分母
        },
        degraded={"退化条"},
    )
    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: orch)
    results = eval_runner.evaluate(
        _gold(tmp_path), cfg={"retrieval": {}, "llm": {"model": "m"}}, use_rewrite=True
    )
    rr = results["meta"]["rewrite_routing"]
    assert rr["n"] == 4  # 5 条里 1 条退化不进分母
    assert rr["agg_predicted"] == 3
    assert rr["agg_gold"] == 2
    assert rr["matches"] == 3
    assert rr["recall"] == 1.0  # 该走聚合路的都触发了
    assert rr["false_trigger"] == 0.5  # 2 条单点题误触发 1 条
    assert rr["by_type"]["fact"] == {
        "n": 2,
        "agg_predicted": 1,
        "agg_gold": 0,
        "matches": 1,
        "recall": None,
        "false_trigger": 0.5,
    }
    assert rr["by_type"]["cross_doc"]["recall"] == 1.0
    assert rr["by_type"]["cross_doc"]["false_trigger"] is None


def test_rewrite_routing_absent_without_rewrite(tmp_path, monkeypatch):
    """不开 rewrite 就没有「预测」：字段必须缺席（None），不能报空统计冒充。"""
    orch = _RoutedOrch(agg_by_question={"单点1": False})
    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: orch)
    results = eval_runner.evaluate(
        _gold(tmp_path), cfg={"retrieval": {}, "llm": {"model": "m"}}
    )
    assert results["meta"]["rewrite_routing"] is None
