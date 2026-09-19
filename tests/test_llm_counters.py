"""`_STATS` 计数：线程安全，且不自锁死。

回归动机（本轮真实踩到）：把 hit/miss 增量收进内部持锁的 `_bump` 之后，调用点
残留的 `with _LOCK` 形成非重入锁嵌套，`chat_timed` 永久挂住——表现是测试套件
停在第一个用例上、没有任何报错。所以这里并发打一轮，同时锁住两件事：
不死锁、计数不重复也不丢失。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from doc_rag.generate import llm as llm_mod

CFG = {
    "model": "m",
    "base_url": "https://api.x.com",
    "api_key": "k",
    "cache": True,
}

_THREADS = 8


def _ok(text="答案"):
    resp = Mock()
    resp.choices = [Mock(message=Mock(content=text))]
    resp.usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )
    return resp


class _FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: _ok())
        )


def test_concurrent_calls_do_not_deadlock_or_miscount(monkeypatch):
    monkeypatch.setattr(llm_mod, "OpenAI", _FakeClient)
    monkeypatch.setattr(llm_mod, "_cache_get", lambda key: None)
    stored: list[str] = []
    monkeypatch.setattr(llm_mod, "_cache_put", lambda k, m, r: stored.append(r))

    before = llm_mod.cache_stats()  # 顺带验证读侧也不持锁嵌套
    with ThreadPoolExecutor(max_workers=_THREADS) as pool:
        answers = list(
            pool.map(lambda i: llm_mod.chat(CFG, f"问题{i}"), range(_THREADS))
        )

    after = llm_mod.cache_stats()
    assert answers == ["答案"] * _THREADS
    assert len(stored) == _THREADS
    assert after["miss"] - before["miss"] == _THREADS
    assert after["hit"] == before["hit"]
    assert after["prompt_tokens"] - before["prompt_tokens"] == 10 * _THREADS
    assert after["reasoning_tokens"] - before["reasoning_tokens"] == 3 * _THREADS
    assert after["cache_read_errors"] == before["cache_read_errors"]


def test_cache_hit_path_bumps_hit_without_deadlock(monkeypatch):
    monkeypatch.setattr(llm_mod, "_cache_get", lambda key: "缓存答案")
    monkeypatch.setattr(llm_mod, "cache_enabled", lambda cfg=None: True)

    def _boom(*a, **k):
        raise AssertionError("命中缓存就不该建客户端")

    monkeypatch.setattr(llm_mod, "OpenAI", _boom)
    before = llm_mod.cache_stats()
    assert llm_mod.chat(CFG, "问题") == "缓存答案"
    assert llm_mod.cache_stats()["hit"] == before["hit"] + 1
