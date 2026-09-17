"""LLM 响应缓存测试（不发真实请求：用假 cfg + monkeypatch）。"""

from doc_rag.generate import llm


def test_cache_key_differs_by_model_and_prompt():
    msgs = [{"role": "user", "content": "同一个问题"}]
    k1 = llm._cache_key("model-a", msgs, temperature=0)
    k2 = llm._cache_key("model-b", msgs, temperature=0)
    k3 = llm._cache_key("model-a", [{"role": "user", "content": "另一个问题"}], temperature=0)
    k4 = llm._cache_key("model-a", msgs, temperature=0.7)
    assert len({k1, k2, k3, k4}) == 4  # 模型/prompt/参数任一变化都产生新键


def test_cache_roundtrip_and_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "_CACHE_PATH", tmp_path / "c.sqlite")
    monkeypatch.setenv("DOC_RAG_LLM_CACHE", "1")
    key = llm._cache_key("m", [{"role": "user", "content": "q"}], temperature=0)
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
