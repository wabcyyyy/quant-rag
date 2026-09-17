"""LLM 响应缓存测试（不发真实请求：用假 cfg + monkeypatch）。"""

from doc_rag.generate import llm


def _cfg(model="model-a", base_url="https://api.a.com"):
    return {"model": model, "base_url": base_url}


def test_cache_key_differs_by_model_prompt_and_endpoint():
    msgs = [{"role": "user", "content": "同一个问题"}]
    k1 = llm._cache_key(_cfg(), msgs, temperature=0)
    k2 = llm._cache_key(_cfg(model="model-b"), msgs, temperature=0)
    k3 = llm._cache_key(_cfg(), [{"role": "user", "content": "另一个问题"}], temperature=0)
    k4 = llm._cache_key(_cfg(), msgs, temperature=0.7)
    assert len({k1, k2, k3, k4}) == 4  # 模型/prompt/参数任一变化都产生新键


def test_cache_key_differs_by_endpoint_even_for_same_model():
    """同名模型换供应商（OpenRouter ↔ 官方）答案不同，缓存键必须分开。"""
    msgs = [{"role": "user", "content": "同一个问题"}]
    assert llm._cache_key(_cfg(base_url="https://api.a.com"), msgs,
                          temperature=0) != llm._cache_key(
        _cfg(base_url="https://api.b.com"), msgs, temperature=0)


def test_cache_roundtrip_and_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_CACHE_PATH", tmp_path / "c.sqlite")
    monkeypatch.setenv("DOC_RAG_LLM_CACHE", "1")
    key = llm._cache_key(_cfg(model="m"), [{"role": "user", "content": "q"}], temperature=0)
    assert llm._cache_get(key) is None
    llm._cache_put(key, "m", "答案")
    assert llm._cache_get(key) == "答案"
    st = llm.cache_stats()
    assert st["cached_total"] == 1


def test_cache_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("DOC_RAG_LLM_CACHE", "0")
    assert llm.cache_enabled({"cache": True}) is False
    monkeypatch.setenv("DOC_RAG_LLM_CACHE", "1")
    assert llm.cache_enabled({"cache": False}) is True  # 环境变量优先
    monkeypatch.delenv("DOC_RAG_LLM_CACHE")
    assert llm.cache_enabled({"cache": False}) is False
    assert llm.cache_enabled({"cache": True}) is True
    assert llm.cache_enabled({}) is True  # 默认开
