"""P1 鉴权与租户边界：/query 与 /query/stream 不再裸奔。

改造前三条路由零 Depends、零鉴权、零限流，且 `body.kb` 未校验即可指向同一
Qdrant 实例上的任意 collection。这里把「未配置令牌就拒绝服务」和「kb 白名单」
两条边界钉住。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from doc_rag.api import main as api_main
from doc_rag.orchestrator import Orchestrator
from doc_rag.retrieve.hybrid import RetrievalOutcome

QUESTION = "新仓库什么时候投用？"
_CHUNKS = [
    {
        "doc_id": f"d{i}",
        "title": f"文档{i}",
        "page": 1,
        "text": f"正文{i}",
        "block_type": "paragraph",
    }
    for i in range(3)
]


def _cfg(**api):
    return {
        "llm": {"model": "m"},
        "retrieval": {"max_contexts": 10},
        "qdrant": {"url": "http://x", "collection": "kb_default"},
        "api": dict(api),
    }


class _Synth:
    def __init__(self, llm_cfg):
        self.last_meta = {"ms": 1.0, "cached": False, "model": "m"}

    def answer(self, q, ctx, **kw):
        return "答案 [1]"

    def answer_stream(self, q, ctx, **kw):
        yield "答案 "
        yield "[1]"


@pytest.fixture
def app_with_auth(monkeypatch):
    monkeypatch.setattr(
        api_main,
        "_orchestrator",
        lambda: Orchestrator(
            _cfg(
                auth_token="sekret",
                allowed_collections=["kb_default", "doc_rag_sample"],
            ),
            retriever=SimpleNamespace(
                retrieve=lambda q, **kw: RetrievalOutcome(chunks=list(_CHUNKS)), cfg={}
            ),
            synthesizer=_Synth({}),
        ),
    )
    return TestClient(api_main.app)


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "sekret-x"}]
)
@pytest.mark.parametrize("path", ["/query", "/query/stream"])
def test_missing_or_wrong_token_is_401(app_with_auth, headers, path):
    assert (
        app_with_auth.post(
            path, json={"question": QUESTION}, headers=headers
        ).status_code
        == 401
    )


@pytest.mark.parametrize("value", ["Bearer sekret", "sekret"])
def test_both_header_forms_are_accepted(app_with_auth, value):
    r = app_with_auth.post(
        "/query", json={"question": QUESTION}, headers={"Authorization": value}
    )
    assert r.status_code == 200
    assert r.json()["answer"] == "答案 [1]"


def test_health_stays_open(app_with_auth):
    """/health 不鉴权：容器探针要能在没有令牌时跑通。"""
    assert app_with_auth.get("/health").status_code == 200


@pytest.mark.parametrize("path", ["/query", "/query/stream"])
def test_kb_outside_allowlist_is_403(app_with_auth, path):
    r = app_with_auth.post(
        path,
        json={"question": QUESTION, "kb": "someone_elses_kb"},
        headers={"Authorization": "Bearer sekret"},
    )
    assert r.status_code == 403


def test_allowed_kb_is_accepted(app_with_auth):
    r = app_with_auth.post(
        "/query",
        json={"question": QUESTION, "kb": "doc_rag_sample"},
        headers={"Authorization": "Bearer sekret"},
    )
    assert r.status_code == 200


def test_unconfigured_token_refuses_service(monkeypatch):
    """令牌没配 = 503 拒绝服务，而不是悄悄开着让人打。"""
    monkeypatch.setattr(
        api_main,
        "_orchestrator",
        lambda: Orchestrator(
            _cfg(auth_token="", allowed_collections=["kb_default"]),
            retriever=SimpleNamespace(
                retrieve=lambda q, **kw: RetrievalOutcome(chunks=list(_CHUNKS)), cfg={}
            ),
            synthesizer=_Synth({}),
        ),
    )
    client = TestClient(api_main.app)
    r = client.post(
        "/query", json={"question": QUESTION}, headers={"Authorization": "Bearer x"}
    )
    assert r.status_code == 503
    assert "DOC_RAG_API_TOKEN" in r.text


def test_rate_limit_per_token(monkeypatch):
    """限流按令牌算：慢请求能占住线程池到 180s，没有闸门就是自我 DoS。"""
    from types import SimpleNamespace

    from doc_rag.api import main as main_mod

    main_mod._RATE_WINDOW.clear()
    monkeypatch.setattr(
        api_main,
        "_orchestrator",
        lambda: Orchestrator(
            {
                "llm": {"model": "m"},
                "retrieval": {"max_contexts": 10},
                "qdrant": {"url": "http://x", "collection": "kb_default"},
                "api": {
                    "auth_token": "t",
                    "allowed_collections": [],
                    "rate_limit_rpm": 2,
                },
            },
            retriever=SimpleNamespace(
                retrieve=lambda q, **kw: RetrievalOutcome(chunks=[]), cfg={}
            ),
            synthesizer=_Synth({}),
        ),
    )
    client = TestClient(api_main.app, headers={"Authorization": "Bearer t"})
    codes = [
        client.post("/query", json={"question": "问？"}).status_code for _ in range(3)
    ]
    assert codes == [200, 200, 429]
    main_mod._RATE_WINDOW.clear()


def test_rate_limit_disabled_when_zero(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        api_main,
        "_orchestrator",
        lambda: Orchestrator(
            {
                "llm": {"model": "m"},
                "retrieval": {"max_contexts": 10},
                "qdrant": {"url": "http://x", "collection": "kb_default"},
                "api": {
                    "auth_token": "t",
                    "allowed_collections": [],
                    "rate_limit_rpm": 0,
                },
            },
            retriever=SimpleNamespace(
                retrieve=lambda q, **kw: RetrievalOutcome(chunks=[]), cfg={}
            ),
            synthesizer=_Synth({}),
        ),
    )
    client = TestClient(api_main.app, headers={"Authorization": "Bearer t"})
    assert all(
        client.post("/query", json={"question": "问？"}).status_code == 200
        for _ in range(5)
    )
