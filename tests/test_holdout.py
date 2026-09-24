"""B9：外部 holdout 脚手架——3 行假输入产出合法 gold + 评分卷；eval 进 meta。"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from doc_rag.eval.schema import GoldItem

_spec = importlib.util.spec_from_file_location(
    "make_holdout", ROOT / "scripts" / "make_holdout.py"
)
mod = importlib.util.module_from_spec(_spec)
sys.modules["make_holdout"] = mod
_spec.loader.exec_module(mod)


def test_three_fake_lines_make_valid_gold_and_worksheet(tmp_path):
    qfile = tmp_path / "questions.txt"
    qfile.write_text(
        "新仓库建设目前进行到哪一步了？\n"
        "\n"  # 空行应被剔除
        "客服系统的代号是什么？\n"
        "有哪些差旅报销标准？\n",
        encoding="utf-8",
    )
    out = tmp_path / "gold_holdout.json"
    ws = tmp_path / "worksheet.md"
    import sys as _sys

    argv = _sys.argv
    _sys.argv = [
        "make_holdout.py",
        str(qfile),
        "--out",
        str(out),
        "--worksheet",
        str(ws),
        "--source-note",
        "外部人（假想），2026-09-25",
    ]
    try:
        mod.main()
    finally:
        _sys.argv = argv

    payload = json.loads(out.read_text("utf-8"))
    assert payload["meta"]["holdout"] is True
    assert payload["meta"]["count"] == 3
    for it in payload["items"]:
        item = GoldItem.model_validate(it)  # eval --gold 直接可消费
        assert item.origin == "external"
        assert item.must_contain == [] and item.source_doc_ids == []
    sheet = ws.read_text("utf-8")
    assert "| 3 | h003 |" in sheet
    assert "G 可用性" in sheet and "F 编造" in sheet


def test_eval_flags_holdout_in_meta(tmp_path, monkeypatch):
    from doc_rag.eval import runner as eval_runner

    gold = tmp_path / "g.json"
    gold.write_text(
        json.dumps(
            {
                "meta": {"gold_version": "holdout", "holdout": True},
                "items": [
                    {
                        "id": "h001",
                        "type": "holdout",
                        "question": "问题",
                        "expected_answer": "",
                        "must_contain": [],
                        "source_doc_ids": [],
                        "refusable": False,
                        "origin": "external",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class _Retriever:
        collection = "t"
        cfg = {"mode": "hybrid"}

    class _Orch:
        def __init__(self, *a, **kw):
            self.retriever = _Retriever()

        def answer(self, question, **kw):
            kw["with_answer"] = False
            from doc_rag.orchestrator import Result

            r = Result(
                plan={
                    "rewritten": question,
                    "filters": None,
                    "aggregate": False,
                    "top_n": None,
                    "reason": "t",
                    "degraded": False,
                },
                retrieved=[],
                contexts=[],
                citations=[],
                retrieval_empty=True,
            )
            r.latency_ms = {
                "rewrite": 0,
                "retrieve": 0,
                "rerank": 0,
                "retrieval_total": 0,
                "synthesize": None,
                "synth_cached": None,
                "total": 0,
            }
            return r

    monkeypatch.setattr(eval_runner, "_build_orchestrator", lambda *a, **kw: _Orch())
    results = eval_runner.evaluate(gold, with_answers=False, holdout=True)
    assert results["meta"]["holdout"] is True
    # 无程序化判据 → 全部自动指标为空是设计；empty_retrieval_n 记录的是另一回事
    assert results["summary"]["empty_retrieval_n"] == 1
