"""RAGAS 上下文口径测试：judge 必须看到 LLM 实际看到的那份上下文。"""

from __future__ import annotations

from doc_rag.eval.runner import (
    _contexts,
    _judge_contexts,
    _legacy_contexts,
    _sample_rows,
)


def _chunk(text: str, title: str | None = "某文档", page: int | None = 3) -> dict:
    return {"text": text, "title": title, "doc_id": "d1", "page": page}


def test_judge_contexts_carry_doc_title_and_number():
    """回归：只传正文时，答案里「《文档》中…」的归属陈述无法被验证（实测 24pt 低估）。"""
    ctx = _contexts([_chunk("最小规格：67元/月")])
    judged = _judge_contexts(ctx)
    assert len(judged) == 1
    assert judged[0].startswith("[1] （某文档 第3页）")
    assert "最小规格：67元/月" in judged[0]


def test_judge_contexts_fall_back_to_doc_id_without_title():
    ctx = _contexts([_chunk("正文", title=None, page=None)])
    assert _judge_contexts(ctx)[0].startswith("[1] （d1）")


def test_judge_contexts_keep_numbering_consistent_with_llm():
    """编号必须与送 LLM 的一致，否则答案里的 [n] 对不上上下文。"""
    ctx = _contexts([_chunk("甲"), _chunk("乙")])
    judged = _judge_contexts(ctx)
    assert judged[1].startswith("[2]")
    assert [c["no"] for c in ctx] == [1, 2]


def test_contexts_respect_max_contexts():
    ctx = _contexts([_chunk(f"块{i}") for i in range(12)], max_n=10)
    assert len(ctx) == 10


def test_legacy_contexts_detects_old_text_only_format():
    assert _legacy_contexts(None) is True
    assert _legacy_contexts([]) is True
    assert _legacy_contexts(["最小规格：67元/月"]) is True  # 旧格式：无编号前缀
    assert _legacy_contexts(["[1] （某文档 第3页）\n最小规格：67元/月"]) is False


def test_sample_rows_spreads_across_tail_types():
    """均匀抽样：黄金集按题型分块排序，取前 N 条会整段漏掉末尾题型。"""
    rows = [{"id": f"q{i:02d}", "type": "fact" if i < 40 else "cross_doc"} for i in range(55)]
    picked = _sample_rows(rows, 15)
    assert len(picked) == 15
    assert picked[0]["id"] == "q00"
    assert any(r["type"] == "cross_doc" for r in picked)  # 旧口径 rows[:15] 全是 fact
    assert len({r["id"] for r in picked}) == 15


def test_sample_rows_no_op_when_sample_exceeds_size():
    rows = [{"id": f"q{i}"} for i in range(5)]
    assert _sample_rows(rows, 0) == rows
    assert _sample_rows(rows, 99) == rows
