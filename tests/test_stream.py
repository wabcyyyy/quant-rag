"""流式输出（T7）回归测试——全离线 mock，不打真实 API。

核心契约：
1. 流式与非流式**同一缓存键**——流式写下的缓存非流式必须能命中，反之亦然；
2. 缓存命中时整段单 chunk 返回且无本次用量；
3. 生成器耗尽后 meta / last_meta 才算就绪（含总耗时与用量）；
4. SSE 端点事件序列 = rewrite / delta* / citations / done。
"""

from __future__ import annotations

from types import SimpleNamespace

from doc_rag.generate import llm as llm_mod
from doc_rag.generate import prompts
from doc_rag.retrieve.hybrid import RetrievalOutcome

_CFG = {"model": "m", "base_url": "https://api.x.com", "api_key": "k"}


def _chunk(content):
    return SimpleNamespace(
        usage=None, choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
    )


def _usage_chunk(prompt=10, completion=5, reasoning=3):
    return SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        ),
        choices=[],
    )


def test_chat_stream_cache_hit_yields_whole_text(monkeypatch):
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: True)
    monkeypatch.setattr(llm_mod, "_cache_get", lambda key: "缓存的整段答案")
    stream, meta = llm_mod.chat_stream(_CFG, "问题", system_prompt="系统提示")
    pieces = list(stream)
    assert pieces == ["缓存的整段答案"]
    assert meta["cached"] is True
    assert meta["prompt_tokens"] is None  # 缓存命中无本次用量


def test_chat_stream_assembles_and_records_usage(monkeypatch):
    seen_kwargs: dict = {}
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: True)
    monkeypatch.setattr(llm_mod, "_cache_get", lambda key: None)

    class _FakeClient:
        def __init__(self, **kwargs):
            def _create(**kw):
                seen_kwargs.update(kw)
                return iter([_chunk("你"), _chunk("好"), _usage_chunk()])

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    cfg = {**_CFG, "reasoning_effort": "none"}
    stream, meta = llm_mod.chat_stream(cfg, "问题", system_prompt="s")
    assert "".join(stream) == "你好"
    assert meta["cached"] is False
    assert meta["prompt_tokens"] == 10
    assert meta["completion_tokens"] == 5
    assert meta["reasoning_tokens"] == 3
    assert meta["ms"] >= 0
    # 流式同样受 reasoning_effort 配置影响：必须真的进了请求参数
    assert seen_kwargs.get("reasoning_effort") == "none"
    assert seen_kwargs.get("stream") is True


def test_stream_and_nontream_share_cache_key(monkeypatch):
    """T7 的核心契约：流式写入的缓存键 == 非流式查询的键。"""
    store: dict = {}
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: True)
    monkeypatch.setattr(llm_mod, "_cache_get", lambda key: store.get(key))

    def _fake_put(key, model, response):
        store[key] = response

    monkeypatch.setattr(llm_mod, "_cache_put", _fake_put)

    class _FakeClient:
        def __init__(self, **kwargs):
            def _create(**kw):
                return iter([_chunk("答案")])

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    stream, _ = llm_mod.chat_stream(_CFG, "问题", system_prompt=prompts.SYSTEM_ANSWER)
    assert "".join(stream) == "答案"  # 耗尽 → 写缓存
    # 非流式调用必须直接命中流式写入的缓存（若键不一致会走到 FakeClient 并报错）
    text, meta = llm_mod.chat_timed(_CFG, "问题", system_prompt=prompts.SYSTEM_ANSWER)
    assert text == "答案"
    assert meta["cached"] is True


def test_answer_stream_sets_last_meta_after_exhaustion(monkeypatch):
    from doc_rag.generate.synthesizer import Synthesizer

    calls: list[dict] = []

    def _fake_chat_stream(cfg, user_prompt, system_prompt=None, temperature=None):
        calls.append({"cfg": dict(cfg), "system": system_prompt})
        meta = {
            "ms": 42.0,
            "cached": False,
            "model": "m",
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "reasoning_tokens": 0,
        }
        return iter(["段1", "段2"]), meta

    monkeypatch.setattr(llm_mod, "chat_stream", _fake_chat_stream)
    syn = Synthesizer({"model": "m"})
    assert syn.last_meta is None
    out = "".join(
        syn.answer_stream("q", [{"no": 1, "text": "t", "doc": "d", "page": 1}])
    )
    assert out == "段1段2"
    assert syn.last_meta["ms"] == 42.0
    assert (
        calls[0]["system"] == prompts.SYSTEM_ANSWER
    )  # 默认 tightened，与 answer() 一致


def test_answer_stream_aggregate_override(monkeypatch):
    from doc_rag.generate.synthesizer import Synthesizer

    calls: list[dict] = []

    def _fake_chat_stream(cfg, user_prompt, system_prompt=None, temperature=None):
        calls.append(dict(cfg))
        return iter(["x"]), {"ms": 1.0, "cached": False, "model": "m"}

    monkeypatch.setattr(llm_mod, "chat_stream", _fake_chat_stream)
    syn = Synthesizer({"model": "m", "reasoning_effort_by_type": {"cross_doc": "none"}})
    list(
        syn.answer_stream(
            "聚合题",
            [{"no": 1, "text": "t", "doc": "d", "page": 1}],
            question_type="cross_doc",
        )
    )
    assert calls[0]["reasoning_effort"] == "none"


# ---------------------------------------------------------------- SSE 端点


def test_query_stream_endpoint_event_sequence(monkeypatch):
    from fastapi.testclient import TestClient

    from doc_rag.api import main as api_main
    from doc_rag.orchestrator import Orchestrator

    cfg = {"llm": {"model": "m"}, "retrieval": {"max_contexts": 0}}
    # /query 与 /query/stream 要求鉴权；这里给一个配好的令牌
    cfg["api"] = {"auth_token": "t", "allowed_collections": []}
    retriever = SimpleNamespace(
        retrieve=lambda *a, **k: RetrievalOutcome(
            chunks=[
                {
                    "doc_id": "d1",
                    "title": "文档",
                    "page": 1,
                    "text": "正文",
                    "block_type": "p",
                }
            ]
        ),
    )

    class _FakeSyn:
        def __init__(self):
            self.last_meta = None

        def answer_stream(self, q, ctx, question_type=None, require_citation=True):
            def _gen():
                yield "第一段"
                yield "第二段"
                self.last_meta = {"ms": 9.0, "cached": False}

            return _gen()

    syn = _FakeSyn()
    monkeypatch.setattr(
        api_main,
        "_orchestrator",
        lambda: Orchestrator(cfg, retriever=retriever, synthesizer=syn),
    )

    client = TestClient(api_main.app, headers={"Authorization": "Bearer t"})
    resp = client.post("/query/stream", json={"question": "测试问题"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = [
        line.removeprefix("event: ")
        for line in resp.text.splitlines()
        if line.startswith("event: ")
    ]
    assert events == ["rewrite", "delta", "delta", "citations", "done"]
    assert '"text": "第一段"' in resp.text or '"text":"第一段"' in resp.text
    assert "latency_ms" in resp.text


def test_health_endpoint_unchanged():
    """旧端点行为不变（T7 验收项）。"""
    from fastapi.testclient import TestClient

    from doc_rag.api import main as api_main

    assert TestClient(api_main.app).get("/health").json() == {"status": "ok"}
