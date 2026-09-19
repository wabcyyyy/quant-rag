"""P2 可观测：一次问答要留下一行可串起来的结构化记录，且不得带走语料内容。

改造前 `src/` 里没有任何日志——一个自称有延迟 SLO 的系统，出问题时无从查起；
而重排失败被静默吞掉正是因为没有任何痕迹可查。
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from doc_rag.api import main as api_main
from doc_rag.orchestrator import Orchestrator
from doc_rag.retrieve.hybrid import RetrievalOutcome


class _Synth:
    def __init__(self, llm_cfg):
        self.last_meta = {"ms": 12.0, "cached": False, "model": "deepseek-flash"}

    def answer(self, q, ctx, **kw):
        return "答案 [1]"


def _orch(rerank_enabled=False, **api):
    return Orchestrator(
        {
            "llm": {"model": "deepseek-flash"},
            "retrieval": {"max_contexts": 10},
            "qdrant": {"url": "http://x", "collection": "kb_default"},
            # 默认关重排：开着它又没桩掉 Reranker 会真去连 http://r 并走满退避重试
            "rerank": {
                "enabled": rerank_enabled,
                "base_url": "http://r",
                "api_key": "k",
            },
            "api": dict(api),
        },
        retriever=SimpleNamespace(
            retrieve=lambda q, **kw: RetrievalOutcome(
                chunks=[
                    {
                        "doc_id": "d1",
                        "title": "机密文档标题",
                        "page": 1,
                        "text": "正文",
                        "block_type": "p",
                    }
                ]
            ),
            cfg={},
        ),
        synthesizer=_Synth({}),
    )


@pytest.fixture
def logs(caplog):
    caplog.set_level(logging.INFO, logger="doc_rag")
    return caplog


def test_one_query_emits_one_structured_line(monkeypatch, logs):
    monkeypatch.setattr(api_main, "_orchestrator", lambda: _orch(auth_token="t"))
    TestClient(api_main.app, headers={"Authorization": "Bearer t"}).post(
        "/query", json={"question": "员工健身房预算是多少？"}
    )

    records = [r for r in logs.records if r.name == "doc_rag.api"]
    assert len(records) == 1
    assert records[0].getMessage() == "query"
    assert records[0].stage == "query"
    assert records[0].n_contexts == 1
    assert records[0].n_retrieved == 1
    assert records[0].model == "deepseek-flash"
    assert records[0].cached is False
    assert records[0].ms is not None


def test_logs_never_carry_question_or_chunk_text(monkeypatch, logs):
    """合规红线：语料是公司内容，日志外传就是第二条泄露路径。"""
    monkeypatch.setattr(api_main, "_orchestrator", lambda: _orch(auth_token="t"))
    TestClient(api_main.app, headers={"Authorization": "Bearer t"}).post(
        "/query", json={"question": "员工健身房预算是多少？"}
    )
    dumped = " ".join(
        f"{r.getMessage()} {getattr(r, '__dict__', {})}" for r in logs.records
    )
    assert "员工健身房" not in dumped
    assert "机密文档标题" not in dumped


def test_formatter_emits_a_per_request_id_and_nothing_else_leaks():
    """request_id 由 formatter 注入（不是 record 属性），所以要直接测 formatter。"""
    from doc_rag.log import JsonFormatter, new_request_id, request_id

    def _line() -> dict:
        rec = logging.LogRecord(
            "doc_rag.api", logging.INFO, "f", 1, "query", None, None
        )
        rec.question_text = (
            "不该出现在日志里的问题"  # 只有 _FIELDS 白名单里的键会被带上
        )
        tok = new_request_id()
        try:
            return json.loads(JsonFormatter().format(rec))
        finally:
            request_id.reset(tok)

    a, b = _line(), _line()
    assert a["request_id"] != b["request_id"] != "-"
    assert a["msg"] == "query" and a["level"] == "INFO"
    assert "question_text" not in a and "不该出现在日志里" not in json.dumps(
        a, ensure_ascii=False
    )


def test_rerank_failure_is_logged_as_warning(monkeypatch, logs):
    import doc_rag.retrieve.rerank as rerank_mod

    class Broken:
        def __init__(self, cfg):
            pass

        def rerank(self, q, chunks, top_n=None):
            raise RuntimeError("429 too many requests")

    monkeypatch.setattr(rerank_mod, "Reranker", Broken)
    orch = _orch(rerank_enabled=True, auth_token="t")
    result = orch.answer("问题？")
    assert result.rerank_error
    warnings = [r for r in logs.records if r.levelno == logging.WARNING]
    assert any("重排失败" in r.getMessage() for r in warnings)
    assert any("429" in getattr(r, "rerank_error", "") for r in warnings)


def test_configure_is_idempotent(capsys):
    from doc_rag import log as log_mod

    log_mod.configure()
    log_mod.configure()
    logger = log_mod.get_logger("probe")
    logger.info("only once")
    captured = capsys.readouterr()
    assert captured.err.count("only once") == 1
