"""`/metrics` 出口：SLO 与失败率要能持续读，不能只靠人工跑 eval 复算。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from doc_rag import metrics
from doc_rag.api import main as api_main
from doc_rag.orchestrator import Orchestrator
from doc_rag.retrieve.hybrid import RetrievalOutcome


class _Synth:
    def __init__(self, llm_cfg):
        self.last_meta = {"ms": 5.0, "cached": False, "model": "m"}

    def answer(self, q, ctx, **kw):
        return "答案 [1]"


def _orch(**api):
    return Orchestrator(
        {
            "llm": {"model": "m"},
            "retrieval": {"max_contexts": 10},
            "qdrant": {"url": "http://x", "collection": "kb_default"},
            "api": dict(api),
        },
        retriever=SimpleNamespace(
            retrieve=lambda q, **kw: RetrievalOutcome(
                chunks=[
                    {
                        "doc_id": "d",
                        "title": "t",
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


@pytest.fixture(autouse=True)
def clean_registry():
    metrics.reset()
    yield
    metrics.reset()


def test_counter_labels_are_stable_and_sorted():
    metrics.inc("doc_rag_requests_total", endpoint="query", kb="a")
    metrics.inc("doc_rag_requests_total", kb="a", endpoint="query")
    out = metrics.render()
    assert 'doc_rag_requests_total{endpoint="query",kb="a"} 2' in out


def test_observe_ms_records_count_sum_and_percentiles():
    for ms in (100.0, 300.0, 200.0):
        metrics.observe_ms("doc_rag_e2e_ms", ms)
    out = metrics.render()
    assert "doc_rag_e2e_ms_count 3" in out
    assert "doc_rag_e2e_ms_sum 600" in out
    assert "doc_rag_e2e_ms_p50 200.0" in out
    assert "doc_rag_e2e_ms_p95 300.0" in out


def test_none_latency_is_skipped_not_counted_as_zero():
    """None 是「这一段没测到」，记成 0 会凭空造出一个完美的低延迟样本。"""
    metrics.observe_ms("doc_rag_rewrite_ms", None)
    assert "doc_rag_rewrite_ms_count" not in metrics.render()


def test_sample_window_is_bounded():
    for i in range(metrics._MAX_SAMPLES + 400):
        metrics.observe_ms("doc_rag_e2e_ms", float(i))
    assert len(metrics._SAMPLES["doc_rag_e2e_ms"]) == metrics._MAX_SAMPLES


def test_metrics_endpoint_requires_token(monkeypatch):
    monkeypatch.setattr(api_main, "_orchestrator", lambda: _orch(auth_token="t"))
    client = TestClient(api_main.app)
    assert client.get("/metrics").status_code == 401
    ok = client.get("/metrics", headers={"Authorization": "Bearer t"})
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("text/plain")


def test_a_query_shows_up_in_metrics(monkeypatch):
    monkeypatch.setattr(api_main, "_orchestrator", lambda: _orch(auth_token="t"))
    client = TestClient(api_main.app, headers={"Authorization": "Bearer t"})
    client.post("/query", json={"question": "预算？"})
    out = client.get("/metrics").text
    assert 'doc_rag_requests_total{endpoint="query"} 1' in out
    assert "doc_rag_e2e_ms_count 1" in out
    assert "doc_rag_rerank_failures_total" not in out  # 没失败就不该冒出来


class _StreamSynth:
    def __init__(self, llm_cfg):
        self.last_meta = {"ms": 5.0, "cached": False, "model": "m"}

    def answer_stream(self, q, ctx, **kw):
        yield "第一段"
        yield "第二段 [1]"


def _stream_orch(**api):
    return Orchestrator(
        {
            "llm": {"model": "m"},
            "retrieval": {"max_contexts": 10},
            "qdrant": {"url": "http://x", "collection": "kb_default"},
            "api": dict(api),
        },
        retriever=SimpleNamespace(
            retrieve=lambda q, **kw: RetrievalOutcome(chunks=[dict(_CHUNK)]), cfg={}
        ),
        synthesizer=_StreamSynth({}),
    )


_CHUNK = {
    "doc_id": "d",
    "title": "t",
    "page": 1,
    "text": "正文",
    "block_type": "p",
}


def test_aborted_stream_is_counted_instead_of_vanishing(monkeypatch):
    """客户端中途断开：`done` 永不到达，这条请求不能从度量里整条消失。

    只记 `done` 的话，「成功率」是拿自己没记的那部分请求当分母算出来的。

    必须直接拽 `body_iterator` 才能造出「读一个事件就断开」——走 TestClient 的话
    portal 会把整条流跑到完成，断不开。而拽它要落在**有主人收摊的事件循环**里：
    `StreamingResponse` 的同步生成器由 starlette 丢进 anyio 线程池跑，线程只在宿主
    loop 正常结束时回收；裸 `asyncio.run()` 关掉 loop 时 root_task 的 done-callback
    没机会触发，那个非 daemon 的 worker 线程就永久悬着——252 项测试打印完全过、
    pytest 进程却不退出（2026-09-20 实测，泄漏点正是本条测试的前一版实现）。
    """
    from anyio.from_thread import start_blocking_portal
    from fastapi.responses import StreamingResponse

    monkeypatch.setattr(api_main, "_orchestrator", lambda: _stream_orch(auth_token="t"))
    resp = api_main.query_stream(api_main.QueryIn(question="预算？"))
    assert isinstance(resp, StreamingResponse)

    async def _abort() -> str:
        it = resp.body_iterator  # starlette 已把同步生成器包成异步迭代器
        first = await it.__anext__()
        await it.aclose()  # 模拟客户端在收到第一个事件后断开
        return first

    with start_blocking_portal() as portal:
        first = portal.call(_abort)

    assert "event: rewrite" in first
    out = metrics.render()
    assert "doc_rag_stream_incomplete_total 1" in out
    assert (
        'doc_rag_requests_total{endpoint="query_stream"}' not in out
    )  # 未完成不计成功
