"""RAGAS 上下文口径测试：judge 必须看到 LLM 实际看到的那份上下文。

这里的 `_contexts` 走 Orchestrator 的真实构造路径（上下文拼装已从 eval 收进
Orchestrator，测试不能再测一份复制品）。
"""

from __future__ import annotations

import json

from doc_rag.eval.runner import (
    _judge_contexts,
    _legacy_contexts,
    _sample_rows,
)
from doc_rag.orchestrator import Orchestrator
from doc_rag.retrieve.hybrid import RetrievalOutcome


def _chunk(text: str, title: str | None = "某文档", page: int | None = 3) -> dict:
    return {"text": text, "title": title, "doc_id": "d1", "page": page}


class _Retriever:
    def __init__(self, results: list[dict]) -> None:
        self._results = results

    def retrieve(self, question, **kw):
        return RetrievalOutcome(chunks=list(self._results))


def _contexts(retrieved: list[dict], max_n: int | None = None) -> list[dict]:
    cfg = {"retrieval": {"max_contexts": max_n or 0}, "llm": {"model": "m"}}
    orch = Orchestrator(cfg, retriever=_Retriever(retrieved), synthesizer=None)
    return orch.answer("问题", use_rewrite=False, with_answer=False).contexts


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
    rows = [
        {"id": f"q{i:02d}", "type": "fact" if i < 40 else "cross_doc"}
        for i in range(55)
    ]
    picked = _sample_rows(rows, 15)
    assert len(picked) == 15
    assert picked[0]["id"] == "q00"
    assert any(r["type"] == "cross_doc" for r in picked)  # 旧口径 rows[:15] 全是 fact
    assert len({r["id"] for r in picked}) == 15


def test_sample_rows_no_op_when_sample_exceeds_size():
    rows = [{"id": f"q{i}"} for i in range(5)]
    assert _sample_rows(rows, 0) == rows
    assert _sample_rows(rows, 99) == rows


# ── E2 的杠杆：`eval --max-contexts` ────────────────────────────────────


def _capture_cfg(monkeypatch, tmp_path, args):
    """跑一次 CLI，停在 `runner.evaluate` 门口，把传进去的 cfg 抓回来。

    不真跑评估：E2 的两臂只要证明「预算确实被改了」。min(max_contexts, rerank.top_n)
    的生效口径在 test_orchestrator_parity 里已经钉着，这里不重复钉一遍。
    """
    import typer
    from typer.testing import CliRunner

    from doc_rag import cli
    from doc_rag.eval import runner

    seen: dict = {}

    def _stop(*a, **kw):
        seen.update(kw)
        raise typer.Exit(0)

    # eval 入口先校验黄金集存在，所以给一个真的空文件——不是为了跑，是为了别半路退出
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"items": []}), encoding="utf-8")
    cfg = {
        "retrieval": {"max_contexts": 10},
        "rerank": {"enabled": True, "top_n": 6},
        "eval": {"gold_file": str(gold), "ragas_sample": 0},
        "paths": {"eval": str(tmp_path)},
    }
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(runner, "evaluate", _stop)
    res = CliRunner().invoke(cli.app, args)
    assert res.exit_code == 0, res.output
    return cfg, seen


def test_max_contexts_flag_presses_both_knobs(monkeypatch, tmp_path):
    """只动 retrieval.max_contexts 的杠杆是假杠杆：块数是两个键的 min。"""
    cfg, seen = _capture_cfg(monkeypatch, tmp_path, ["eval", "--max-contexts", "25"])
    assert seen["cfg"]["retrieval"]["max_contexts"] == 25
    assert seen["cfg"]["rerank"]["top_n"] == 25
    # 改的是 load_config 返回的那一份；configs/default.yaml 不许被写
    assert cfg["retrieval"]["max_contexts"] == 25


def test_no_flag_leaves_the_budget_alone(monkeypatch, tmp_path):
    """不传时一个键都不动——否则 E2 的对照臂不知道自己在跟谁比。"""
    _, seen = _capture_cfg(monkeypatch, tmp_path, ["eval"])
    assert seen["cfg"]["retrieval"]["max_contexts"] == 10
    assert seen["cfg"]["rerank"]["top_n"] == 6
