"""N13：报告路径必须拒绝被 kill 的中间态；resume 路径必须仍能读它。

`_flush_progress` 逐条落盘的中间态带 `meta.partial=True`，但全仓库没有任何消费方
检查它——一份被 kill 的运行留下的 items+summary 与完整结果同形，读的人看不出它
没跑完。闸门加在**报告路径**（compare.load_scores：compare-ragas /
compare-retrieval / compare 全走它）；`resume_from` 路径**不查**该字段——它必须
能读中间态补跑（B2），别把续跑一起锁死。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_load_scores_refuses_partial(tmp_path):
    from doc_rag.eval.compare import load_scores

    p = tmp_path / "results_partial.json"
    p.write_text(
        json.dumps(
            {
                "meta": {"partial": True, "resume_of": "results_full.json"},
                "summary": {"n_items_done": 1, "hit_at_5": 1.0},
                "items": [
                    {
                        "id": "q1",
                        "type": "fact",
                        "first_hit_rank": 1,
                        "doc_coverage": 1.0,
                        "n_retrieved": 8,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="partial"):
        load_scores(p, metric="hit_at_5")


def test_load_scores_accepts_complete(tmp_path):
    from doc_rag.eval.compare import load_scores

    p = tmp_path / "results_full.json"
    p.write_text(
        json.dumps(
            {
                "meta": {"collection": "c"},
                "summary": {"hit_at_5": 1.0},
                "items": [
                    {
                        "id": "q1",
                        "type": "fact",
                        "first_hit_rank": 1,
                        "doc_coverage": 1.0,
                        "n_retrieved": 8,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    loaded = load_scores(p, metric="hit_at_5")
    assert loaded["items"] == {"q1": 1.0}


def test_resume_still_reads_partial_state(tmp_path, monkeypatch):
    """B2 不被锁死：meta.partial=True 的中间态必须能被 --resume 补跑。"""
    from doc_rag.eval import runner as eval_runner

    class _FakeOrch:
        def __init__(self):
            self.calls: list[str] = []
            self.retriever = type(
                "R", (), {"collection": "t", "cfg": {"mode": "hybrid"}}
            )()

        def answer(self, question, **kw):
            self.calls.append(question)
            from doc_rag.orchestrator import Result

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

    gold = tmp_path / "gold.json"
    gold.write_text(
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
                    for i in range(1, 4)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cfg = {"retrieval": {"max_contexts": 3}, "llm": {"model": "m"}}

    # 第一轮：q1 完成、q2 抛异常 → 逐条 flush 已在盘上；再手工把它标成未跑完的中间态
    out = tmp_path / "run.json"
    orch1 = _FakeOrch()
    monkeypatch.setattr(
        eval_runner,
        "_build_orchestrator",
        lambda *a, **kw: _FlakyOnce(orch1, {"问题2"}),
    )
    eval_runner.evaluate(gold, cfg=cfg, progress_file=out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["summary"]["n_errors"] == 1
    data["meta"]["partial"] = True
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # 第二轮：resume 读这份 partial 中间态 → 只补失败那条，不许被闸门拒绝
    orch2 = _FakeOrch()
    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: orch2)
    results = eval_runner.evaluate(gold, cfg=cfg, resume_from=out)
    assert len(orch2.calls) == 1
    assert results["summary"]["n_resumed"] == 2
    assert results["summary"]["n_errors"] == 0


class _FlakyOnce:
    """包装替身：questions_to_fail 里的题面抛异常（模拟端点抖动）。"""

    def __init__(self, inner, fail_on: set[str]):
        self._inner = inner
        self._fail_on = fail_on
        self.retriever = inner.retriever

    def answer(self, question, **kw):
        if question in self._fail_on:
            raise RuntimeError("embed endpoint timeout (simulated)")
        return self._inner.answer(question, **kw)
