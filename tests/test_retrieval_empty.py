"""B3：拒答归因——「索引挂了」与「真没记载」必须可区分。

改前，检索为空时合成器照样产出「无法回答」，与黄金集 no_answer 的正确拒答
在结果文件里长得一模一样：前者是事故（索引没建/语料没进来），后者是被测行为。
现在三处可见：`Result.retrieval_empty`、`/query` 响应与 SSE done、
`/metrics` 的 `doc_rag_empty_retrieval_total`（eval 逐条 + summary 计数同源）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag import metrics
from doc_rag.api.main import _record
from doc_rag.eval import runner as eval_runner
from doc_rag.orchestrator import Orchestrator
from doc_rag.retrieve.hybrid import RetrievalOutcome


class _FixedRetriever:
    def __init__(self, chunks: list[dict], collection: str = "t"):
        self._chunks = chunks
        self.collection = collection
        self.cfg = {"mode": "hybrid"}

    def retrieve(self, question, **kw):
        return RetrievalOutcome(chunks=list(self._chunks))


def _cfg() -> dict:
    return {"retrieval": {"max_contexts": 3}, "llm": {"model": "m"}}


def _chunk(doc_id: str = "d1") -> dict:
    return {
        "chunk_id": f"{doc_id}:1",
        "doc_id": doc_id,
        "title": "文档",
        "text": "正文",
        "section_path": [],
        "page": 1,
        "block_type": "paragraph",
    }


def test_empty_retrieval_flagged_on_result():
    orch = Orchestrator(_cfg(), retriever=_FixedRetriever([]), synthesizer=None)
    result = orch.answer("问题", use_rewrite=False, with_answer=False)
    assert result.retrieval_empty is True

    orch2 = Orchestrator(
        _cfg(), retriever=_FixedRetriever([_chunk()]), synthesizer=None
    )
    result2 = orch2.answer("问题", use_rewrite=False, with_answer=False)
    assert result2.retrieval_empty is False


def test_api_metrics_count_empty_retrieval():
    metrics.reset()
    orch = Orchestrator(_cfg(), retriever=_FixedRetriever([]), synthesizer=None)
    result = orch.answer("问题", use_rewrite=False, with_answer=False)
    _record(result, endpoint="query")
    assert "doc_rag_empty_retrieval_total 1" in metrics.render()
    _record(result, endpoint="query")
    assert "doc_rag_empty_retrieval_total 2" in metrics.render()


def test_eval_records_retrieval_empty(tmp_path, monkeypatch):
    """eval 逐条 + summary 计数；「空检索的拒答」与「有产出的拒答」两条记录可区分。"""
    gold = tmp_path / "gold.json"
    gold.write_text(
        json.dumps(
            {
                "meta": {"gold_version": "t"},
                "items": [
                    {
                        "id": "q1",
                        "type": "no_answer",
                        "question": "问题",
                        "expected_answer": "应拒答",
                        "must_contain": [],
                        "source_doc_ids": [],
                        "refusable": True,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class _EmptyOrch:
        def __init__(self, *a, **kw):
            self._orch = Orchestrator(
                _cfg(), retriever=_FixedRetriever([]), synthesizer=None
            )
            self.retriever = self._orch.retriever

        def answer(self, *a, **kw):
            kw["with_answer"] = False
            return self._orch.answer(*a, **kw)

    monkeypatch.setattr(
        eval_runner, "_build_orchestrator", lambda *a, **kw: _EmptyOrch()
    )
    results = eval_runner.evaluate(gold, with_answers=False)
    item = results["items"][0]
    assert item["retrieval_empty"] is True
    assert results["summary"]["empty_retrieval_n"] == 1
