"""B2：eval 逐条容错 + --resume 的四条验收。

1. 单条异常不终止整轮；2. 失败条目不进分母（也不冒充 0）；
3. resume 只补未完成条目（成功的原样复用）；4. 未命中仍按 0 计入分母。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


from doc_rag.eval import runner as eval_runner
from doc_rag.orchestrator import Result


def _gold(tmp_path, n=3):
    p = tmp_path / "gold.json"
    p.write_text(
        json.dumps(
            {
                "meta": {"gold_version": "t"},
                "items": [
                    {
                        "id": f"q{i}",
                        "type": "fact",
                        "question": f"问题{i}",
                        "expected_answer": "答案",
                        "must_contain": ["42"],
                        "source_doc_ids": ["d1"],
                        "refusable": False,
                    }
                    for i in range(1, n + 1)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return p


class _FakeOrch:
    """可编程替身：questions_to_fail 里的题面触发异常；其余返回固定命中。"""

    def __init__(self, fail_on: set[str]):
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.retriever = type("R", (), {"collection": "t", "cfg": {"mode": "hybrid"}})()

    def answer(self, question, **kw):
        self.calls.append(question)
        if question in self.fail_on:
            raise RuntimeError("embed endpoint timeout (simulated)")
        result = Result(
            plan={
                "rewritten": question,
                "filters": None,
                "aggregate": False,
                "top_n": None,
                "reason": "t",
                "degraded": False,
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


def _patch(monkeypatch, orch: _FakeOrch):
    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: orch)


def _cfg():
    return {"retrieval": {"max_contexts": 3}, "llm": {"model": "m"}}


def test_single_failure_does_not_abort(tmp_path, monkeypatch):
    orch = _FakeOrch(fail_on={"问题2"})
    _patch(monkeypatch, orch)
    results = eval_runner.evaluate(_gold(tmp_path), cfg=_cfg())
    assert results["summary"]["n_errors"] == 1
    assert results["summary"]["n_items"] == 3
    failed = [it for it in results["items"] if it.get("error")]
    assert len(failed) == 1 and failed[0]["id"] == "q2"
    assert failed[0]["answered_ok"] is None  # 没测出 ≠ 0


def test_failed_item_excluded_from_denominator(tmp_path, monkeypatch):
    orch = _FakeOrch(fail_on={"问题2"})
    _patch(monkeypatch, orch)
    results = eval_runner.evaluate(_gold(tmp_path), cfg=_cfg())
    # 严格关键词分母 = 真跑成的 2 条，失败条不进也不冒充 0
    assert results["summary"]["n_items"] == 3
    assert results["summary"]["strict_keyword_accuracy"] == 1.0


def test_failed_item_excluded_from_hit_and_mrr_denominator(tmp_path, monkeypatch):
    """N2：失败条不得进 hit_at_5/mrr 的分母（`_error_row` docstring 承诺的全剔语义）。

    3 条题 1 条抛异常：q1/q3 都命中 → hit 与 mrr 都应是 1.0（分母 2）。
    旧实现只按拒答题过滤、不剔 error 行 → 失败条 first_hit_rank=None 被当 miss，
    hit_at_5 = mrr = 0.6667，与同文件里覆盖率/nDCG 的分母（剔除失败）自相矛盾。
    """
    orch = _FakeOrch(fail_on={"问题2"})
    _patch(monkeypatch, orch)
    results = eval_runner.evaluate(_gold(tmp_path), cfg=_cfg())
    assert results["summary"]["n_errors"] == 1
    assert results["summary"]["hit_at_5"] == 1.0
    assert results["summary"]["mrr"] == 1.0
    # 与派生侧（compare.py `_retrieval_value` 丢 error 行）重新一致：两侧同分母


def test_resume_only_reruns_incomplete(tmp_path, monkeypatch):
    out = tmp_path / "run.json"
    orch1 = _FakeOrch(fail_on={"问题2"})
    _patch(monkeypatch, orch1)
    eval_runner.evaluate(_gold(tmp_path), cfg=_cfg(), progress_file=out)
    assert out.exists()

    orch2 = _FakeOrch(fail_on=set())
    _patch(monkeypatch, orch2)
    results = eval_runner.evaluate(
        _gold(tmp_path), cfg=_cfg(), resume_from=out, progress_file=out
    )
    # 只重跑失败那一条；成功两条原样复用
    assert len(orch2.calls) == 1
    assert results["summary"]["n_errors"] == 0
    assert results["summary"]["n_resumed"] == 2
    assert results["meta"]["n_resumed"] == 2


def test_miss_still_counts_as_zero(tmp_path, monkeypatch):
    """未命中不是失败：first_hit_rank=None 但条目正常，分母照进、值按 0。"""
    orch = _FakeOrch(fail_on=set())
    # gold 指向 d2，替身只检回 d1 → 未命中
    p = tmp_path / "gold2.json"
    p.write_text(
        json.dumps(
            {
                "meta": {"gold_version": "t"},
                "items": [
                    {
                        "id": "q1",
                        "type": "fact",
                        "question": "问题1",
                        "expected_answer": "答案",
                        "must_contain": ["42"],
                        "source_doc_ids": ["d2"],
                        "refusable": False,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _patch(monkeypatch, orch)
    results = eval_runner.evaluate(p, cfg=_cfg())
    assert results["summary"]["n_errors"] == 0
    assert results["summary"]["hit_at_5"] == 0.0  # 计入分母、按 0 计
